
import sys
sys.path.insert(0,"..")
sys.path.insert(0,"/scratch/bemuller/mt_norm_parse")
#TODO  why is it necessary fo rioc ? 
from model.sequence_prediction import greedy_decode_batch, decode_seq_str, decode_interacively
import pdb
from model.seq2seq import LexNormalizer
from model.generator import Generator
from io_.data_iterator import data_gen_conllu
from io_.dat import conllu_data
from io_.info_print import printing
import os
import json
import sys
import numpy as np
import torch
from env.project_variables import PROJECT_PATH, TRAINING, DEV, TEST, DEMO, DEMO2, LIU, LEX_TEST, REPO_DATASET, CHECKPOINT_DIR, SEED_TORCH, SEED_NP, LEX_TRAIN
from toolbox.gpu_related import use_gpu_
sys.path.insert(0, os.path.join(PROJECT_PATH, "..", "experimental_pipe"))
from reporting.write_to_performance_repo import report_template, write_dic


np.random.seed(SEED_NP)
torch.manual_seed(SEED_TORCH)


def evaluate(batch_size, data_path, write_report=True, dir_report=None,
             dict_path=None, model_full_name=None,
             score_to_compute_ls=None, mode_norm_ls=None,get_batch_mode_evaluate=True,
             overall_label="ALL_MODELS",overall_report_dir=CHECKPOINT_DIR,
             model_specific_dictionary=True, label_report="", print_raw=False, model=None,
             compute_mean_score_per_sent=False,auxilliary_task_norm_not_norm=False,
             normalization=True, debug=False, force_new_dic=False, use_gpu=None, verbose=0):
    assert model_specific_dictionary, "ERROR : only model_specific_dictionary = True supported now"
    validation = True
    #if validation:
        #assert not get_batch_mode_evaluate, "ERROR : validation was set to true but get_batch_mode_evaluate is {} while it should be False".format(get_batch_mode_evaluate)

    # NB : now : you have to load dictionary when evaluating (cannot recompute) (could add in the LexNormalizer ability)
    use_gpu = use_gpu_(use_gpu)
    hardware_choosen = "GPU" if use_gpu else "CPU"
    printing("{} mode ", var=([hardware_choosen]), verbose_level=0, verbose=verbose)
    printing("EVALUATION : evaluating with compute_mean_score_per_sent {}".format(compute_mean_score_per_sent), verbose=verbose, verbose_level=1)
    if score_to_compute_ls is None:
        score_to_compute_ls = ["edit", "exact"]
    if mode_norm_ls is None:
        mode_norm_ls = ["all", "NORMED", "NEED_NORM"]
    printing("EVALUATION : Evaluating {} metric with details {}  ", var=[score_to_compute_ls, mode_norm_ls], verbose=verbose, verbose_level=3)
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
                          voc_size=voc_size, use_gpu=use_gpu, dict_path=dict_path, model_specific_dictionary=True,
                          dir_model=os.path.join(PROJECT_PATH, "checkpoints",
                                                 model_full_name + "-folder"),
                          verbose=verbose
                          ) if model is None else model

    data_read = conllu_data.read_data_to_variable(data_path, model.word_dictionary, model.char_dictionary,
                                                  model.pos_dictionary,
                                                  model.xpos_dictionary, model.type_dictionary,
                                                  use_gpu=use_gpu, symbolic_root=False,
                                                  norm_not_norm=auxilliary_task_norm_not_norm,
                                                  symbolic_end=False, dry_run=0, lattice=False, verbose=verbose,
                                                  normalization=normalization,
                                                  validation=validation,
                                                  add_start_char=1, add_end_char=1)
    batchIter = data_gen_conllu(data_read, model.word_dictionary, model.char_dictionary, model.pos_dictionary,
                                model.xpos_dictionary,
                                model.type_dictionary, batch_size=batch_size,  add_start_char=1,
                                add_end_char=1, get_batch_mode=get_batch_mode_evaluate,
                                normalization=normalization,
                                print_raw=print_raw,  verbose=verbose)

    model.eval()

    score_dic = greedy_decode_batch(char_dictionary=model.char_dictionary, verbose=verbose, gold_output=True,
                                    score_to_compute_ls=score_to_compute_ls,use_gpu=use_gpu,
                                    stat="sum", mode_norm_score_ls=mode_norm_ls,
                                    batchIter=batchIter, model=model, compute_mean_score_per_sent=compute_mean_score_per_sent,
                                    batch_size=batch_size)
    # NB : each batch should have the same size !! same number of words : otherwise averaging is wrong
    try:
      for score in score_to_compute_ls:
          for mode_norm in mode_norm_ls:
              print("MODEL Normalization {} on normalization {} score is {} in average out of {} tokens on evaluation based on {} "
                    .format(score, mode_norm,score_dic[score+"-"+mode_norm]/score_dic[score+"-"+mode_norm+"-total_tokens"], score_dic[score+"-"+mode_norm+"-total_tokens"], data_path))
    except ZeroDivisionError as e :
        print("ERROR catched {} ".format(e))
        raise Exception(e)
    for score in score_to_compute_ls:
        for mode_norm in mode_norm_ls:
            stat_type_ls = [""]
            if compute_mean_score_per_sent:
                stat_type_ls.append("-mean_per_sent")
            for stat_type in stat_type_ls:
                if stat_type == "":
                    score_value = score_dic[score+"-"+ mode_norm+stat_type]/score_dic[score+"-"+mode_norm+"-total_tokens"]
                elif stat_type == "-mean_per_sent":
                    score_value = score_dic[score + "-" + mode_norm + stat_type]/score_dic[score+"-"+mode_norm+"-n_sents"]
                report = report_template(metric_val=score+stat_type,
                                         info_score_val=mode_norm,
                                         score_val=score_value,
                                         model_full_name_val=model.model_full_name,
                                         task="normalization",
                                         report_path_val=model.arguments["checkpoint_dir"],
                                         evaluation_script_val="normalization_"+score,
                                         model_args_dir=model.args_dir,
                                         data_val=REPO_DATASET[data_path])
                _dir_report = os.path.join(dir_report, model.model_full_name+"-"+score+"-"+mode_norm+"-report-"+label_report+".json")
                over_all_report_dir = os.path.join(dir_report, model.model_full_name+"-report-"+label_report+".json")
                over_all_report_dir_all_models = os.path.join(overall_report_dir, overall_label+"-report.json")
                writing_mode = "w" if not os.path.isfile(over_all_report_dir) else "a"
                writing_mode_all_models = "w" if not os.path.isfile(over_all_report_dir_all_models) else "a"
                json.dump(report, open(_dir_report, "w"))
                if writing_mode_all_models == "w":
                  json.dump([report], open(over_all_report_dir_all_models, writing_mode_all_models))
                  print("Creating new over_all_report_dir_all_models {} ".format(over_all_report_dir_all_models))
                else:
                  all_report = json.load(open(over_all_report_dir_all_models, "r"))
                  all_report.append(report)
                  json.dump(all_report,open(over_all_report_dir_all_models, "w"))
                if writing_mode == "w":
                  print("Creating new over_all_report_dir {} ".format(over_all_report_dir))
                  json.dump([report], open(over_all_report_dir, writing_mode))
                else:
                  #print("Appending over_all_report_dir {} ".format(over_all_report_dir))
                  all_report = json.load(open(over_all_report_dir, "r"))
                  all_report.append(report)
                  json.dump(all_report,open(over_all_report_dir, "w"))
                printing("Report saved {} ".format(_dir_report), verbose=verbose, verbose_level=1)

        printing("Overall Model specific Report saved {} ".format(over_all_report_dir), verbose=verbose, verbose_level=1)
        printing("Overall  Report saved {} ".format(over_all_report_dir_all_models), verbose=verbose, verbose_level=1)

    return score_dic


if __name__ == "__main__":
    list_all_dir = os.listdir(os.path.join(PROJECT_PATH,"checkpoints"))
    #for ablation_id in ["aaad","bd55","0153","f178"]:
    for ablation_id in ["08661","d6960"]:
      #for data in [DEMO,DEMO2]:
      for get_batch_mode_evaluate in [False]:
        for batch_size in [10]:
          for data in [LEX_TRAIN, LEX_TEST, DEV, LIU]:
            list_ = [dir_ for dir_ in list_all_dir if dir_.startswith(ablation_id) and not dir_.endswith("log") and not dir_.endswith(".json") and not dir_.endswith("summary")]
            print("FOLDERS : ", list_)
            for folder_name in list_:
              model_full_name = folder_name[:-7]
              print("MODEL_FULL_NAME : ", model_full_name)
              print("0Evaluating {} ".format(model_full_name))
              evaluate(model_full_name=model_full_name, data_path=data,#LIU,
                       dict_path=os.path.join(PROJECT_PATH, "checkpoints", folder_name, "dictionaries"),
                       label_report="eval_again", use_gpu=None, 
                       overall_label="08661+d6960-"+str(batch_size)+"-"+str(get_batch_mode_evaluate)+"_get_batch",#"f2f2-iterate+new_data-"+str(batch_size)+"-"+str(get_batch_mode_evaluate)+"_get_batch-validation_True",
                       mode_norm_ls=None,
                       normalization=True,
                       model_specific_dictionary=True,
                       batch_size=batch_size,
                       debug=False,
                       compute_mean_score_per_sent=True,
                       get_batch_mode_evaluate=get_batch_mode_evaluate,
                       dir_report=os.path.join(PROJECT_PATH, "checkpoints", folder_name), verbose=1)





#reporting = False
#              if reporting:
#                 report_path = ""
#                 report_generation_script = "normalizer_edit"
#                 dir_performance_json = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/experimental_pipe/model_repository/performancecopy.json"
#                 metric = "edit"
#                 write_dic(report_path, report_generation_script, dir_performance_json, metric, "None", model.model_full_name, test_path, 0)