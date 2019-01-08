
import sys
sys.path.insert(0,"..")
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
import torch
from env.project_variables import PROJECT_PATH, TRAINING, DEV, TEST, DEMO, DEMO2
from toolbox.gpu_related import use_gpu_
sys.path.insert(0, os.path.join(PROJECT_PATH, "..", "experimental_pipe"))
from reporting.write_to_performance_repo import report_template, write_dic


def evaluate(batch_size, data_path, write_report=True, dir_report=None,
             dict_path=None, model_full_name=None,
             score_to_compute_ls=None, mode_norm_ls=None,
             model_specific_dictionary=True, label_report="", print_raw=False, model=None,
             normalization=True, debug=False, force_new_dic=False, use_gpu=None, verbose=0):
    # NB : now : you have to load dictionary when evaluating (cannot recompute) (could add in the LexNormalizer ability)
    use_gpu = use_gpu_(use_gpu)
    #print("WARNING use_gpu forced to False ")
    if score_to_compute_ls is None:
        score_to_compute_ls = ["edit", "exact"]
    if mode_norm_ls is None:
        mode_norm_ls = ["all", "NORMED", "NEED_NORM"]
    printing("EVALUATION : Evaluating {} metric with details {}  ",var=[score_to_compute_ls, mode_norm_ls], verbose=verbose, verbose_level=3)
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
                                                  symbolic_end=False, dry_run=0, lattice=False, verbose=verbose,
                                                  normalization=normalization,
                                                  add_start_char=1, add_end_char=1)
    batchIter = data_gen_conllu(data_read, model.word_dictionary, model.char_dictionary, model.pos_dictionary,
                                model.xpos_dictionary,
                                model.type_dictionary, batch_size=batch_size,  add_start_char=1,
                                add_end_char=1,
                                normalization=normalization,
                                print_raw=print_raw,  verbose=verbose)

    model.eval()

    score_dic = greedy_decode_batch(char_dictionary=model.char_dictionary, verbose=verbose, gold_output=True,
                                    score_to_compute_ls=score_to_compute_ls,use_gpu=use_gpu,
                                    stat="sum", mode_norm_score_ls=mode_norm_ls,
                                    batchIter=batchIter, model=model,
                                    batch_size=batch_size)
    print("-->", score_dic)
    # NB : each batch should have the same size !! same number of words : otherwise averaging is wrong
    try:
      for score in score_to_compute_ls:
          for mode_norm in mode_norm_ls:
              print("MODEL Normalization {} on normalization {} score is {} in average out of {} tokens on evaluation based on {} "
                    .format(score, mode_norm,score_dic[score+"-"+mode_norm]/score_dic[score+"-"+mode_norm+"-total_tokens"], score_dic[score+"-"+mode_norm+"-total_tokens"], data_path))
    except ZeroDivisionError as e :
        print("ERROR catched {} ".format(e))
        raise Exception(e)
    scoring_count = 0
    for score in score_to_compute_ls:
        for mode_norm in mode_norm_ls:
            scoring_count += 1
            writing_mode = "w" if scoring_count == 1 else "a"
            report = report_template(metric_val=score, info_score_val=mode_norm,
                                     score_val=score_dic[score+"-"+mode_norm]/score_dic[score+"-"+mode_norm+"-total_tokens"],
                                     model_full_name_val=model.model_full_name,
                                     task="normalization",
                                     report_path_val=model.arguments["checkpoint_dir"],
                                     evaluation_script_val="normalization_"+score,
                                     model_args_dir=model.args_dir,
                                     data_val=data_path)
            _dir_report = os.path.join(dir_report, model.model_full_name+"-"+score+"-"+mode_norm+"-report-"+label_report+".json")
            over_all_report_dir = os.path.join(dir_report, model.model_full_name+"-report-"+label_report+".json")
            json.dump(report, open(_dir_report, "w"))
            json.dump(report, open(over_all_report_dir, writing_mode))
            printing("Report saved {} ".format(_dir_report), verbose=verbose, verbose_level=1)

        printing("Overall Report saved {} ".format(over_all_report_dir), verbose=verbose, verbose_level=1)

    return score_dic


if __name__ == "__main__":
    evaluate(model_full_name="761f-TEST-model_1_6c01", data_path=DEMO,
             dict_path="../checkpoints/761f-TEST-model_1_6c01-folder/dictionaries",
             label_report="test",
             normalization=True, model_specific_dictionary=True, batch_size=50,
             dir_report="../checkpoints/comparison_ablation-big2_b8a1-folder", verbose=1)


#reporting = False
#              if reporting:
#                 report_path = ""
#                 report_generation_script = "normalizer_edit"
#                 dir_performance_json = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/experimental_pipe/model_repository/performancecopy.json"
#                 metric = "edit"
#                 write_dic(report_path, report_generation_script, dir_performance_json, metric, "None", model.model_full_name, test_path, 0)