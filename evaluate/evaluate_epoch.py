
import sys
sys.path.insert(0,"..")
#TODO  why is it necessary fo rioc ? 
from model.sequence_prediction import greedy_decode_batch, decode_seq_str, decode_interacively
import pdb
from model.seq2seq import LexNormalizer, Generator
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

use_old_code=False
if use_old_code:
    dict_path = "../dictionariesbackup/"
    train_path = DEV
    dev_path = TEST
    #test_path = DEMO2
    use_gpu = torch.cuda.is_available()
    printing("INFO : use_gpu is {}".format(use_gpu), verbose=0, verbose_level=0)
    debug = False
    normalization = True
    add_start_char = 1
    add_end_char = 1
    word_dictionary, char_dictionary, pos_dictionary,\
    xpos_dictionary, type_dictionary = \
            conllu_data.create_dict(dict_path=dict_path,
                                    train_path=train_path,
                                    dev_path=dev_path,
                                    test_path=None,
                                    add_start_char=add_start_char,
                                    word_embed_dict={},
                                    dry_run=False,
                                    vocab_trim=True)

    if not debug:
        pdb.set_trace = lambda: 1
    verbose = 0
    _dir = os.path.dirname(os.path.realpath(__file__))
    voc_size = len(char_dictionary.instance2index)+1

    # NORMALIZATION DEMO auto_encoder_TEST_70b7
    # autoencoder demo auto_encoder_TEST_f7ab
    # NORMALIZATION BIG : auto_encoder_TEST_21ac

    model_full_name = "compare_normalization_all_45e6"#"compare_normalization_all_879e"#compare_normalization_all_45e6

    model = LexNormalizer(generator=Generator, load=True, model_full_name=model_full_name,#"normalizer_lexnorm_ad6e",#"normalizer_lexnorm_12bf",
                          # "6437","#"auto_encoder_TEST_f7ab",#="normalizer_lexnorm_ad6e",#"6437",
                          voc_size=voc_size, use_gpu=use_gpu,
                          dir_model=os.path.join(PROJECT_PATH, "checkpoints", model_full_name+"-folder" ),
                          verbose=verbose)
    batch_size = 5
    nbatch = 30

    #data_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/lexnorm.integrated"
    data_path = TEST
    for data_path, label_report in zip([DEV, TEST],["owputi_train", "lexnorm_test"]):
      #label_report = "test_lexnorm"
      batchIter = data_gen_conllu(data_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                                  type_dictionary, batch_size=batch_size,  add_start_char=add_start_char,
                                  add_end_char=add_end_char,
                                  normalization=normalization,
                                  print_raw=True,  verbose=verbose)

      model.eval()
      batch_decoding = True
      #print("LOSS", loss)

      if batch_decoding:
          score_to_compute_ls = ["edit", "exact"]
          score_dic = greedy_decode_batch(char_dictionary=char_dictionary, verbose=verbose, gold_output=True,
                                          score_to_compute_ls=score_to_compute_ls,
                                          stat="sum",
                                          batchIter=batchIter, model=model, batch_size=batch_size)
          print("-->", score_dic)
          # NB : each batch should have the same size !! same number of words : otherwise averaging is wrong
          try:
              for score in score_to_compute_ls:
                  print("MODEL Normalization {} score is {} in average out of {} tokens on {} batches evaluation based on {} "
                        .format(score, score_dic[score]/score_dic[score+"total_tokens"], score_dic[score+"total_tokens"], nbatch, data_path))
          except ZeroDivisionError as e:
              print("ERROR catched {} ".format(e))

          for score in score_to_compute_ls:
              report = report_template(metric_val=score, info_score_val="None",
                                       score_val=score_dic[score]/score_dic[score+"total_tokens"],
                                       model_full_name_val=model.model_full_name,
                                       task="normalization",
                                       report_path_val=model.arguments["checkpoint_dir"],
                                       evaluation_script_val="normalization_"+score,
                                       model_args_dir=model.args_dir,
                                       data_val=data_path)
              dir_report = os.path.join("..", "checkpoints", model.model_full_name+"-folder",
                                        model.model_full_name+"-"+score+"-report-"+label_report+".json")

              json.dump(report, open(dir_report, "w"))
              print("Report saved {} ".format(dir_report))

              reporting = False
              if reporting:
                  report_path = ""
                  report_generation_script = "normalizer_edit"
                  dir_performance_json = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/experimental_pipe/model_repository/performancecopy.json"
                  metric = "edit"
                  write_dic(report_path, report_generation_script, dir_performance_json, metric, "None", model.model_full_name, test_path, 0)


def evaluate(dict_path, model_full_name, batch_size, data_path, write_report=True, dir_report=None,
             model_specific_dictionary=True, label_report="",
             normalization=True, debug=False, force_new_dic=False, use_gpu=None, verbose=0):
    use_gpu = use_gpu_(use_gpu)
    if write_report:
        assert dir_report is not None

    if not model_specific_dictionary:
        word_dictionary, char_dictionary, pos_dictionary, \
        xpos_dictionary, type_dictionary = \
            conllu_data.load_dict(dict_path=dict_path,
                                  train_path=train_path,force_new_dic=force_new_dic,
                                  dev_path=dev_path, test_path=None, word_embed_dict={},
                                  dry_run=False, vocab_trim=True,
                                  add_start_char=add_start_char, verbose=1)
        voc_size = len(char_dictionary.instance2index) + 1
    else:
        voc_size = None
    if not debug:
        pdb.set_trace = lambda: 1


    model = LexNormalizer(generator=Generator, load=True, model_full_name=model_full_name,
                          voc_size=voc_size, use_gpu=use_gpu, dict_path=dict_path,model_specific_dictionary=True,
                          dir_model=os.path.join(PROJECT_PATH, "checkpoints", model_full_name + "-folder"),
                          verbose=verbose)
    batchIter = data_gen_conllu(data_path, model.word_dictionary, model.char_dictionary, model.pos_dictionary, model.xpos_dictionary,
                                model.type_dictionary, batch_size=batch_size,  add_start_char=1,
                                add_end_char=1,
                                normalization=normalization,
                                print_raw=True,  verbose=verbose)

    model.eval()
    batch_decoding = True

    if batch_decoding:
        score_to_compute_ls = ["edit", "exact"]
        score_dic = greedy_decode_batch(char_dictionary=model.char_dictionary, verbose=verbose, gold_output=True,
                                      score_to_compute_ls=score_to_compute_ls,
                                          stat="sum",
                                          batchIter=batchIter, model=model, batch_size=batch_size)
        print("-->", score_dic)
        # NB : each batch should have the same size !! same number of words : otherwise averaging is wrong
        try:
          for score in score_to_compute_ls:
              print("MODEL Normalization {} score is {} in average out of {} tokens on {} batches evaluation based on {} "
                    .format(score, score_dic[score]/score_dic[score+"total_tokens"], score_dic[score+"total_tokens"], nbatch, data_path))
        except ZeroDivisionError as e:
          print("ERROR catched {} ".format(e))

        for score in score_to_compute_ls:
          report = report_template(metric_val=score, info_score_val="None",
                                   score_val=score_dic[score]/score_dic[score+"total_tokens"],
                                   model_full_name_val=model.model_full_name,
                                   task="normalization",
                                   report_path_val=model.arguments["checkpoint_dir"],
                                   evaluation_script_val="normalization_"+score,
                                   model_args_dir=model.args_dir,
                                   data_val=data_path)
          dir_report = os.path.join("..", "checkpoints", model.model_full_name+"-folder",
                                    model.model_full_name+"-"+score+"-report-"+label_report+".json")

          json.dump(report, open(dir_report, "w"))
          print("Report saved {} ".format(dir_report))


if __name__=="__main__":
    evaluate(model_full_name="test_dbc4", data_path=DEV, dict_path="../checkpoints/test_dbc4-folder/dictionaries", label_report="test",
             normalization=True, model_specific_dictionary=True, batch_size=2, dir_report="../checkpoints/test_dbc4-folder",
             verbose=1)