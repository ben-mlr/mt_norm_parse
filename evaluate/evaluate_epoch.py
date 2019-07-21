from env.importing import *

sys.path.insert(0, "..")
sys.path.insert(0, "/scratch/bemuller/mt_norm_parse")
#TODO  why is it necessary for rioc ?
from predict.prediction_batch import greedy_decode_batch#, decode_seq_str, decode_interacively
from model.seq2seq import LexNormalizer
from model.generator import Generator
from io_.data_iterator import data_gen_conllu, readers_load, data_gen_multi_task_sampling_batch
from io_.dat import conllu_data
from io_.info_print import printing
import toolbox.report_tools as rep_tl

from env.project_variables import PROJECT_PATH, TRAINING, DEV,EWT_DEV,EWT_PRED_TOKEN_UDPIPE,\
    PERMUTATION_TRAIN, PERMUTATION_TEST, \
    EWT_TEST, TEST, DEMO, DEMO2, LIU, LEX_TEST, REPO_DATASET, CHECKPOINT_DIR, LEX_TRAIN, LIU_TRAIN, LIU_DEV, CP_WR_PASTE_TEST, MTNT_EN_FR_TEST,MTNT_EN_FR_TEST_DEMO
from toolbox.gpu_related import use_gpu_

sys.path.insert(0, os.environ.get("EXPERIENCE", os.path.join(PROJECT_PATH, "..", "experimental_pipe")))
try:
    from reporting.write_to_performance_repo import report_template, write_dic
    reportint_unavailable = False
except Exception as e:
    print("REPORTING modules not available")
    reportint_unavailable = True
from evaluate.normalization_errors import score_auxiliary
from env.project_variables import SCORE_AUX


def evaluate(batch_size, data_path, tasks, evaluated_task,
             write_report=True, dir_report=None,
             dict_path=None, model_full_name=None,
             score_to_compute_ls=None, mode_norm_ls=None, get_batch_mode_evaluate=True,
             overall_label="ALL_MODELS", overall_report_dir=CHECKPOINT_DIR, bucket=False,
             model_specific_dictionary=True, label_report="",
             print_raw=False,
             model=None,
             compute_mean_score_per_sent=False, write_output=False,
             word_decoding=False, char_decoding=True,
             extra_arg_specific_label="", scoring_func_sequence_pred="BLUE",
             max_char_len=None,
             normalization=True, debug=False,
             force_new_dic=False,
             use_gpu=None, verbose=0):
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
                          tasks=tasks,
                          word_decoding=word_decoding, char_decoding=char_decoding,
                          voc_size=voc_size, use_gpu=use_gpu, dict_path=dict_path, model_specific_dictionary=True,
                          dir_model=os.path.join(PROJECT_PATH, "checkpoints", model_full_name + "-folder"),
                          extra_arg_specific_label=extra_arg_specific_label,
                          loading_sanity_test=True,
                          verbose=verbose
                          ) if model is None else model

    if score_to_compute_ls is None:
        score_to_compute_ls = ["edit", "exact"]
        if model.auxilliary_task_norm_not_norm:
            score_to_compute_ls.extend(SCORE_AUX)

    printing("EVALUATION : Evaluating {} metric with details {}  ", var=[score_to_compute_ls, mode_norm_ls], verbose=verbose, verbose_level=3)

    #rep_tl.checkout_layer_name("encoder.seq_encoder.weight_ih_l0", model.named_parameters(), info_epoch="EVAL")

    readers_eval = readers_load(datasets=[data_path], tasks=[evaluated_task], word_dictionary=model.word_dictionary,
                                word_dictionary_norm=model.word_nom_dictionary, char_dictionary=model.char_dictionary,
                                pos_dictionary=model.pos_dictionary, xpos_dictionary=model.xpos_dictionary,
                                type_dictionary=model.type_dictionary, use_gpu=use_gpu,
                                norm_not_norm=model.auxilliary_task_norm_not_norm, word_decoder=word_decoding,
                                bucket=bucket,max_char_len=max_char_len,
                                add_start_char=1, add_end_char=1, symbolic_end=model.symbolic_end, symbolic_root=model.symbolic_root,
                                verbose=verbose)
    batchIter = data_gen_multi_task_sampling_batch(tasks=[evaluated_task], readers=readers_eval, batch_size=batch_size,
                                                   word_dictionary=model.word_dictionary,
                                                   char_dictionary=model.char_dictionary,
                                                   pos_dictionary=model.pos_dictionary,
                                                   get_batch_mode=get_batch_mode_evaluate,
                                                   word_dictionary_norm=model.word_nom_dictionary,
                                                   extend_n_batch=1, dropout_input=0,
                                                   verbose=verbose)

    model.eval()
    # the formulas comes from normalization_erros functions
    score_dic_new, formulas = greedy_decode_batch(char_dictionary=model.char_dictionary, verbose=verbose, gold_output=True,
                                                  score_to_compute_ls=score_to_compute_ls, use_gpu=use_gpu,
                                                  write_output=write_output, eval_new=True,
                                                  task_simultaneous_eval=[evaluated_task],
                                                  stat="sum", mode_norm_score_ls=mode_norm_ls,
                                                  label_data=REPO_DATASET[data_path],
                                                  batchIter=batchIter, model=model,
                                                  scoring_func_sequence_pred=scoring_func_sequence_pred,
                                                  compute_mean_score_per_sent=compute_mean_score_per_sent,
                                                  batch_size=batch_size)
    for score_name, formula in formulas.items():
        if isinstance(formula, tuple) and len(formula) > 1:
            (num, denom) = formula
            score_value = score_dic_new[num]/score_dic_new[denom] if score_dic_new[denom] > 0 else None
            #score_value_per_sent =
            if score_dic_new[denom] == 0:
                print("WARNING Score {} has denumerator {} null and numerator {} equal to  {}".format(score_name, denom,
                                                                                                      num,
                                                                                                      score_dic_new[num]
                                                                                                      ))
            reg = re.match("([^-]+)-([^-]+)-.*", num)
            mode_norm = reg.group(1)
            task = reg.group(2)
            # report all in a dictionary
            if not reportint_unavailable:
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
            else:
                report = {"report ":0}
            over_all_report_dir = os.path.join(dir_report, model.model_full_name + "-report-" + label_report + ".json")
            over_all_report_dir_all_models = os.path.join(overall_report_dir, overall_label + "-report.json")
            writing_mode = "w" if not os.path.isfile(over_all_report_dir) else "a"
            writing_mode_all_models = "w" if not os.path.isfile(over_all_report_dir_all_models) else "a"
            for dir, writing_mode in zip([over_all_report_dir, over_all_report_dir_all_models],
                                         [writing_mode, writing_mode_all_models]):
                if writing_mode == "w":
                    _all_report = [report]
                    json.dump([report], open(dir, writing_mode))
                    printing("REPORT : Creating new report  {} ".format(dir), verbose=verbose, verbose_level=1)
                else:
                    all_report = json.load(open(dir, "r"))
                    all_report.append(report)
                    json.dump(all_report, open(dir, "w"))
    printing("NEW REPORT metric : {} ", var=[" ".join(list(formulas.keys()))], verbose=verbose, verbose_level=1)
    try:
        printing("NEW REPORT : model specific report saved {} ".format(over_all_report_dir), verbose=verbose, verbose_level=1)
        printing("NEW REPORT : overall report saved {} ".format(over_all_report_dir_all_models), verbose=verbose,verbose_level=1)
    except Exception as e:
        print(Exception(e))
    if writing_mode == "w":
        all_report = _all_report
    return all_report


if __name__ == "__main__":
    list_all_dir = os.listdir(os.path.join(PROJECT_PATH, "checkpoints"))
    PRED_AND_EVAL = True
    if PRED_AND_EVAL:
        #
        for ablation_id in ["c47db-B0-model_1-model_1_e6a3"]:
          for get_batch_mode_evaluate in [False]:
            for batch_size in [2]:
              for data in [DEMO2]:
                list_ = [dir_ for dir_ in list_all_dir if dir_.startswith(ablation_id) and not dir_.endswith("log") and not dir_.endswith(".json") and not dir_.endswith("summary")]
                print("FOLDERS : ", list_)
                if len(list_) == 0:
                    raise(Exception("error empty list on {} id".format(ablation_id)))
                for folder_name in list_:
                  model_full_name = folder_name[:-7]
                  print("MODEL_FULL_NAME : ", model_full_name)
                  print("0Evaluating {} ".format(model_full_name))
                  time_ = time.time()
                  evaluate(
                           model_full_name=model_full_name, data_path=data,
                           dict_path=os.path.join(PROJECT_PATH, "checkpoints", folder_name, "dictionaries"),
                           label_report="eval_again", use_gpu=None,
                           overall_label=ablation_id+"-"+str(batch_size)+"-"+str(get_batch_mode_evaluate)+"_get_batch",
                           mode_norm_ls=None,
                           #score_to_compute_ls=["norm_not_norm-Recall"],
                           normalization=True, model_specific_dictionary=True, batch_size=batch_size,
                           debug=True, bucket=False,
                           compute_mean_score_per_sent=True,
                           word_decoding=False, char_decoding=True,
                           scoring_func_sequence_pred="exact_match",
                           tasks=["normalize"], evaluated_task="normalize",
                           get_batch_mode_evaluate=get_batch_mode_evaluate, write_output=True,
                           max_char_len=20,
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