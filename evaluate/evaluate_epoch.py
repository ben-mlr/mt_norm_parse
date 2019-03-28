
import sys
sys.path.insert(0,"..")
sys.path.insert(0,"/scratch/bemuller/mt_norm_parse")
#TODO  why is it necessary fo rioc ? 
from predict.prediction_batch import greedy_decode_batch#, decode_seq_str, decode_interacively
import pdb
from model.seq2seq import LexNormalizer
from model.generator import Generator
from io_.data_iterator import data_gen_conllu, readers_load, data_gen_multi_task_sampling_batch
from io_.dat import conllu_data
from io_.info_print import printing
import os
import json
import sys
import numpy as np
import torch
import re

from scipy.stats import hmean
from env.project_variables import PROJECT_PATH, TRAINING, DEV,EWT_DEV,EWT_PRED_TOKEN_UDPIPE, EWT_TEST, TEST, DEMO, DEMO2, LIU, LEX_TEST, REPO_DATASET, CHECKPOINT_DIR, SEED_TORCH, SEED_NP, LEX_TRAIN, LIU_TRAIN, LIU_DEV, CP_WR_PASTE_TEST, MTNT_EN_FR_TEST,MTNT_EN_FR_TEST_DEMO
from toolbox.gpu_related import use_gpu_
sys.path.insert(0, os.path.join(PROJECT_PATH, "..", "experimental_pipe"))
from reporting.write_to_performance_repo import report_template, write_dic
from evaluate.normalization_errors import score_auxiliary
from env.project_variables import SCORE_AUX

np.random.seed(SEED_NP)
torch.manual_seed(SEED_TORCH)


def evaluate(batch_size, data_path, task,
             write_report=True, dir_report=None,
             dict_path=None, model_full_name=None,
             score_to_compute_ls=None, mode_norm_ls=None, get_batch_mode_evaluate=True,
             overall_label="ALL_MODELS",overall_report_dir=CHECKPOINT_DIR, bucket = False,
             model_specific_dictionary=True, label_report="", print_raw=False, model=None,
             compute_mean_score_per_sent=False,write_output=False,
             word_decoding=False, char_decoding=True,
             extra_arg_specific_label="", scoring_func_sequence_pred="BLUE",
             max_char_len=None,
             normalization=True, debug=False, force_new_dic=False, use_gpu=None, verbose=0):
    assert model_specific_dictionary, "ERROR : only model_specific_dictionary = True supported now"
    # NB : now : you have to load dictionary when evaluating (cannot recompute) (could add in the LexNormalizer ability)
    use_gpu = use_gpu_(use_gpu)
    hardware_choosen = "GPU" if use_gpu else "CPU"
    printing("{} mode ", var=([hardware_choosen]), verbose_level=0, verbose=verbose)
    printing("EVALUATION : evaluating with compute_mean_score_per_sent {}".format(compute_mean_score_per_sent), verbose=verbose, verbose_level=1)

    if mode_norm_ls is None:
        mode_norm_ls = ["all", "NORMED", "NEED_NORM"]
    if write_report:
        assert dir_report is not None
    if model is not None:
        assert model_full_name is None and dict_path is None, \
            "ERROR as model is provided : model_full_name and dict_path should be None"
    else:
        assert model_full_name is not None and dict_path is not None,\
            "ERROR : model_full_name and dict_path required to load model "
    voc_size = None
    if not debug:
        pdb.set_trace = lambda: 1

    model = LexNormalizer(generator=Generator, load=True, model_full_name=model_full_name,
                          word_decoding=word_decoding, char_decoding=char_decoding, # added for dictionary purposes : might be other ways
                          voc_size=voc_size, use_gpu=use_gpu, dict_path=dict_path, model_specific_dictionary=True,
                          dir_model=os.path.join(PROJECT_PATH, "checkpoints", model_full_name + "-folder"), extra_arg_specific_label=extra_arg_specific_label,
                          verbose=verbose
                          ) if model is None else model

    if score_to_compute_ls is None:
        score_to_compute_ls = ["edit", "exact"]
        if model.auxilliary_task_norm_not_norm:
            score_to_compute_ls.extend(SCORE_AUX)

    printing("EVALUATION : Evaluating {} metric with details {}  ", var=[score_to_compute_ls, mode_norm_ls], verbose=verbose, verbose_level=3)

    readers_eval = readers_load(datasets=[data_path], tasks=[task], word_dictionary=model.word_dictionary,
                                word_dictionary_norm=model.word_nom_dictionary, char_dictionary=model.char_dictionary,
                                pos_dictionary=model.pos_dictionary, xpos_dictionary=model.xpos_dictionary,
                                type_dictionary=model.type_dictionary, use_gpu=use_gpu,
                                norm_not_norm=model.auxilliary_task_norm_not_norm, word_decoder=word_decoding,
                                bucket=bucket,max_char_len=max_char_len,
                                add_start_char=1, add_end_char=1, symbolic_end=model.symbolic_end, symbolic_root=model.symbolic_root,
                                verbose=verbose)
    batchIter = data_gen_multi_task_sampling_batch(tasks=[task], readers=readers_eval, batch_size=batch_size,
                                                   word_dictionary=model.word_dictionary,
                                                   char_dictionary=model.char_dictionary,
                                                   pos_dictionary=model.pos_dictionary,
                                                   get_batch_mode=get_batch_mode_evaluate,
                                                   extend_n_batch=1, dropout_input=0,
                                                   verbose=verbose)


    model.eval()
    # the formulas comes from normalization_erros functions
    score_dic_new, formulas = greedy_decode_batch(char_dictionary=model.char_dictionary, verbose=verbose, gold_output=True,
                                                  score_to_compute_ls=score_to_compute_ls, use_gpu=use_gpu,
                                                  write_output=write_output, eval_new=True,
                                                  stat="sum", mode_norm_score_ls=mode_norm_ls,
                                                  label_data=REPO_DATASET[data_path],
                                                  batchIter=batchIter, model=model,
                                                  scoring_func_sequence_pred=scoring_func_sequence_pred,
                                                  compute_mean_score_per_sent=compute_mean_score_per_sent,
                                                  batch_size=batch_size)

    for score_name, formula in formulas.items():
        if isinstance(formula, tuple) and len(formula) > 1:
            (num, denom) = formula
            score_value = score_dic_new[num]/score_dic_new[denom] if score_dic_new[denom]>0 else None
            #score_value_per_sent =
            if score_dic_new[denom] == 0:
                print("WARNING Score {} has denumerator {} null and numerator {} equal to  {}".format(score_name, denom,num, score_dic_new[num]))
            reg = re.match("([^-]+)-([^-]+)-.*", num)
            mode_norm = reg.group(1)
            task = reg.group(2)
            # report all in a dictionary
            report = report_template(metric_val=score_name,
                                     info_score_val=mode_norm,
                                     score_val=score_value,
                                     n_sents=score_dic_new["n_sents"],
                                     avg_per_sent=0,
                                     n_tokens_score=score_dic_new.get(mode_norm+"-"+task+"-gold-count",-1),
                                     model_full_name_val=model.model_full_name,
                                     task=task,
                                     report_path_val=model.arguments["checkpoint_dir"],
                                     evaluation_script_val="exact_match",
                                     model_args_dir=model.args_dir,
                                     data_val=REPO_DATASET[data_path])
            over_all_report_dir = os.path.join(dir_report, model.model_full_name + "NEW-report-" + label_report + ".json")
            over_all_report_dir_all_models = os.path.join(overall_report_dir, overall_label + "NEW-report.json")
            writing_mode = "w" if not os.path.isfile(over_all_report_dir) else "a"
            writing_mode_all_models = "w" if not os.path.isfile(over_all_report_dir_all_models) else "a"
            for dir, writing_mode in zip([over_all_report_dir, over_all_report_dir_all_models ], [writing_mode, writing_mode_all_models]):
                if writing_mode == "w":
                    json.dump([report], open(dir, writing_mode))
                    printing("REPORT : Creating new report  {} ".format(dir), verbose=verbose, verbose_level=1)
                else:
                    all_report = json.load(open(dir, "r"))
                    all_report.append(report)
                    json.dump(all_report, open(dir, "w"))
    printing("NEW REPORT metric : {} ", var=[" ".join(list(formulas.keys()))], verbose=verbose, verbose_level=1)
    printing("NEW REPORT : model specific report saved {} ".format(over_all_report_dir), verbose=verbose, verbose_level=1)
    printing("NEW REPORT : overall report saved {} ".format(over_all_report_dir_all_models), verbose=verbose,verbose_level=1)

    return None

#4538 , 4578

if __name__ == "__main__":
    list_all_dir = os.listdir(os.path.join(PROJECT_PATH, "checkpoints"))
    #for ablation_id in ["aaad"]:#,"bd55","0153","f178"]:
    #for ablation_id in ["42a20-WARMUP-unrolling-False0.1_scale_aux-True_aux-0.1do_char_dec-False_char_src_atten-model_13_db74-folder"]:
    #for ablation_id in ["5754c-bestbest"]:
    #for ablation_id in ["08661","d6960"]:
    #for ablation_id in ["f2f2-batchXdropout_char0.1-to_char_src-1_dir_sent-10_batch_size-model_18_aa04","8d9a0-new_data-batchXdropout_char0.2-to_char_src-1_dir_sent-20_batch_size-dir_word_encoder_1-model_1_33ca"]:
    #for ablation_id in ["e390","24f9d"]:
    #for ablation_id in ["01880"]:
    #for ablation_id in ["d6960-new_data-batchXdropout_char0-to_char_src-1_dir_sent-10_batch_size-model_2_be5d","08661-new_data-batchXdropout_char0-to_char_src-2_dir_sent-20_batch_size-model_5_c8b2"]:
    #for ablation_id in ["8d9a0-new_data-batchXdropout_char0.2-to_char_src-1_dir_sent-20_batch_size-dir_word_encoder_1-model_1_33ca","8d9a0-new_data-batchXdropout_char0.2-to_char_src-1_dir_sent-20_batch_size-dir_word_encoder_1-model_2_e437"]:
    #for ablation_id in ["21cc8-fixed_all_context-aux_dense1dir_word-200_aux-0do_char_dec-False_char_src-model_12_4d49"]:
    #for ablation_id in ["28aa3-schedule-policy_2"]:
      #for data in [DEMO,DEMO2]:
    #for ablation_id in ["97440_rioc-64c34-ATTbatch-aux-scale-shared_contex-Falseteach_Falseaux-model_2_61d6-folder"]:
    # tag : 101089-B-model_55_32bb-folder
    # norm  9122945-B-model_1_b5fe-folder
    PRED_AND_EVAL = True
    if PRED_AND_EVAL:
        for ablation_id in ["9122945-B-model_1_b5fe-folder"]:
          for get_batch_mode_evaluate in [False]:
            for batch_size in [2]:
              #for data in [LIU, DEV, LEX_TEST]:
              for data in [LIU_TRAIN]:
                list_ = [dir_ for dir_ in list_all_dir if dir_.startswith(ablation_id) and not dir_.endswith("log") and not dir_.endswith(".json") and not dir_.endswith("summary")]
                print("FOLDERS : ", list_)
                for folder_name in list_:
                  model_full_name = folder_name[:-7]
                  print("MODEL_FULL_NAME : ", model_full_name)
                  print("0Evaluating {} ".format(model_full_name))
                  import time
                  time_ = time.time()
                  evaluate(
                           model_full_name=model_full_name, data_path=data,#LIU,
                           dict_path=os.path.join(PROJECT_PATH, "checkpoints", folder_name, "dictionaries"),
                           label_report="eval_again", use_gpu=None,
                           overall_label=ablation_id+"-"+str(batch_size)+"-"+str(get_batch_mode_evaluate)+"_get_batch",#"f2f2-iterate+new_data-"+str(batch_size)+"-"+str(get_batch_mode_evaluate)+"_get_batch-validation_True",
                           mode_norm_ls=None, #score_to_compute_ls=["norm_not_norm-Recall"],
                           normalization=True, model_specific_dictionary=True, batch_size=batch_size,
                           debug=False, bucket=False,
                           compute_mean_score_per_sent=True,
                           word_decoding=False, char_decoding=True,
                           scoring_func_sequence_pred="exact_match", task="normalize",
                           get_batch_mode_evaluate=get_batch_mode_evaluate, write_output=True,
                           max_char_len=1000,
                           dir_report=os.path.join(PROJECT_PATH, "checkpoints", folder_name), verbose=1
                           )
                  print("TIME with max len {} on data {}s ".format(time.time()-time_, data))
    EVAL = False
    if EVAL:
        from evaluate import conll18_ud_eval
        data = EWT_DEV
        # predicted tags on gold tokens
        sys_pred = "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/../predictions/101089-B-model_55_32bb-lexnorm-normalized.conll"
        sys_pred = "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/../predictions/101089-B-model_55_32bb-ewt_dev-normalized.conll"
        #sys_pred = "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/../predictions/101089-B-model_55_32bb-ewt_test-normalized.conll"
        # predicted tags on pred tokens tokens
        #sys_pred = "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/../predictions/101089-B-model_55_32bb-ud_pred_tokens-ewt_dev-normalized.conll"
        gold = conll18_ud_eval.load_conllu(open(EWT_TEST,"r"))
        sys = conll18_ud_eval.load_conllu(open(sys_pred,"r"))
        results = conll18_ud_eval.evaluate(gold, sys)
        for score, val in results.items():
            print("METRIC {} is {} F1 ({} recall {} precision) ".format(score, results[score].f1, results[score].recall, results[score].precision))





#reporting = False
#              if reporting:
#                 report_path = ""
#                 report_generation_script = "normalizer_edit"
#                 dir_performance_json = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/experimental_pipe/model_repository/performancecopy.json"
#                 metric = "edit"
#                 write_dic(report_path, report_generation_script, dir_performance_json, metric, "None", model.model_full_name, test_path, 0)