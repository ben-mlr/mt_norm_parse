from model.sequence_prediction import greedy_decode_batch, decode_seq_str, decode_interacively
import pdb
from model.loss import LossCompute
import os
from io_.info_print import printing
from model.seq2seq import LexNormalizer, Generator
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


def interact(dic_path, model_full_name, dir_model,
             model_specific_dictionary=True, verbose=2):

    if not model_specific_dictionary:
        assert train_path is not None and dev_path is not None
    if not model_specific_dictionary:
        word_dictionary, char_dictionary, pos_dictionary,\
        xpos_dictionary, type_dictionary = \
            conllu_data.create_dict(dict_path=dict_path,
                                    train_path=train_path,
                                    dev_path=dev_path,
                                    test_path=None,
                                    add_start_char=1,
                                    word_embed_dict={},
                                    dry_run=False,
                                    vocab_trim=True)
        voc_size = len(char_dictionary.instance2index) + 1,
    else:
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
    decode_interacively(max_len=MAX_LEN, model=model, char_dictionary=char_dictionary,
                        verbose=verbose)


if __name__=="__main__":
    debug = False
    model_specific_dictionary = True
    script_dir = os.path.dirname(os.path.realpath(__file__))

    model_full_name = "TEST2_37be"
    dic_path = os.path.join(script_dir, "..", "checkpoints", model_full_name + "-folder", "dictionaries")
    model_dir = os.path.join(script_dir, "..", "checkpoints", model_full_name + "-folder")

    interact(dic_path=dic_path, dir_model=model_dir,
             model_full_name=model_full_name)