from model.sequence_prediction import greedy_decode_batch, decode_seq_str, decode_interacively
import pdb
from model.loss import LossCompute
import os
from io_.info_print import printing
from model.seq2seq import LexNormalizer
from model.generator import Generator
import torch.nn as nn
import torch
import sys
from io_.data_iterator import data_gen_conllu
from io_.batch_generator import MaskBatch
#sys.path.insert(0,"/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/ELMoLex_sosweet/")
from io_.dat import conllu_data
from training.epoch_train import run_epoch
from env.project_variables import PROJECT_PATH, TRAINING, DEV, TEST, DEMO, DEMO2

MAX_LEN = 20
#dict_path = "../dictionariesbackup/"


def interact(dic_path, model_full_name, dir_model, debug=False,
             model_specific_dictionary=True, verbose=2):
    assert model_specific_dictionary
    char_dictionary = None
    voc_size = None

    if not debug:
        pdb.set_trace = lambda: 1
    model = LexNormalizer(generator=Generator,
                          voc_size=voc_size,
                          load=True, model_full_name=model_full_name,
                          model_specific_dictionary=model_specific_dictionary,
                          dict_path=dic_path,
                          dir_model=dir_model,
                          verbose=verbose)
    model.eval()
    decode_interacively(max_len=MAX_LEN, model=model, char_dictionary=char_dictionary, sent_mode=True,
                        verbose=verbose)


if __name__ == "__main__":
    
    debug = False
    model_specific_dictionary = True
    script_dir = os.path.dirname(os.path.realpath(__file__))
    list_all_dir = os.listdir("../checkpoints/")
    #for ablation_id in ["aaad","bd55","0153","f178"]:
    ablation_id="89fa"
    #for data in [LIU, DEV]:
    list_ = [dir_ for dir_ in list_all_dir if dir_.startswith(ablation_id) and not dir_.endswith("log")]
    print("FOLDERS : ", list_)
    for folder_name in list_:
        model_full_name = folder_name[:-7]
        print("MODEL_FULL_NAME : ", model_full_name)
        print("0Evaluating {} ".format(model_full_name))
        dic_path = os.path.join(script_dir, "..", "checkpoints", model_full_name + "-folder", "dictionaries")
        model_dir = os.path.join(script_dir, "..", "checkpoints", model_full_name + "-folder")
        interact(dic_path=dic_path, dir_model=model_dir, model_full_name=model_full_name, debug=False)
        break