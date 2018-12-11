from model.seq2seq import LexNormalizer, Generator
import torch.nn as nn
import torch
import sys
from io_.data_iterator import data_gen_conllu
from io_.batch_generator import MaskBatch
#sys.path.insert(0,"/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/ELMoLex_sosweet/")
from io_.dat import conllu_data
from io_.info_print import print_char_seq
from training.epoch_train import run_epoch
from model.loss import LossCompute
from torch.autograd import Variable
from model.sequence_prediction import greedy_decode, decode_seq_begins_with, decode_interacively

import numpy as np
import pdb

import time
from io_.info_print import printing
from io_.from_array_to_text import output_text

pdb.set_trace = lambda: 1


dict_path = "./dictionaries/"
train_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/en-ud-train.conllu"
dev_pat = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/owoputi.integrated"
test_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/lexnorm.integrated"

train_run = True

if train_run:
    add_start_char = 1
    verbose = 5
    nbatches = 10


    word_dictionary, char_dictionary, pos_dictionary, \
    xpos_dictionary, type_dictionary = conllu_data.create_dict(dict_path=dict_path,
                                                               train_path=test_path,
                                                               dev_path=test_path,
                                                               test_path=test_path,
                                                               word_embed_dict={},
                                                               dry_run=False,
                                                               add_start_char=add_start_char,
                                                               vocab_trim=True)

    V = len(char_dictionary.instance2index)+1
    print("Character vocabulary is {} length".format(V))

    model = LexNormalizer(generator=Generator, char_embedding_dim=5, voc_size=V,
                          hidden_size_encoder=11, output_dim=10,
                          hidden_size_decoder=11, verbose=verbose)

    batchIter = data_gen_conllu(test_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary,
                                normalization=True,
                                add_start_char=add_start_char, add_end_char=0,
                                batch_size=2, nbatch=nbatches, print_raw=True, verbose=verbose)

    printing("Starting training", verbose=verbose, verbose_level=0)
    loss = run_epoch(batchIter, model, LossCompute(model.generator, verbose=verbose),
                     n_epochs=1, i_epoch=1, n_batches=None, empty_run=False, verbose=verbose)
    printing("END training loss is {} ".format(loss), verbose=verbose, verbose_level=0)
    #print("input seq {} \n  input mask {} \n  output seq {} \n output mask {} \n ".format(batch.input_seq,batch.input_seq_mask, batch.output_seq, batch.output_mask ))

predict_run = False

if predict_run:

    dict_path = "./dictionaries/"
    train_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/en-ud-train.conllu"
    dev_pat = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/owoputi.integrated"
    test_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/lexnorm.integrated.demo"

    word_dictionary, char_dictionary, pos_dictionary,\
    xpos_dictionary, type_dictionary = \
            conllu_data.create_dict(dict_path=dict_path,
                                    train_path=test_path,
                                    dev_path=test_path,
                                    test_path=None,
                                    add_start_char=1,
                                    word_embed_dict={},
                                    dry_run=False,
                                    vocab_trim=True)

    verbose = 2

    #verbose = 2
    #3b87
    #1782
    #cd05

    model = LexNormalizer(generator=Generator, load=True, model_full_name="07d2", dir_model="./test/test_models",
                          verbose=verbose)
    batch_size = 10
    nbatch = 1
    verbose = 1
    batchIter = data_gen_conllu(test_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                                type_dictionary, batch_size=batch_size, nbatch=nbatch, add_start_char=1, add_end_char=0,
                                print_raw=True,  verbose=verbose)

    V = model.arguments["voc_size"]
    hidden_size_decoder = model.arguments["hidden_size_decoder"]
    model.eval()

    batch_decoding, sequence_decoding, interactive_mode = False, False, True
    loss = run_epoch(batchIter, model, LossCompute(model.generator, verbose=verbose),
                         i_epoch=0, n_epochs=1,
                         verbose=verbose,
                         log_every_x_batch=100)
    batchIter = data_gen_conllu(test_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                                type_dictionary, batch_size=batch_size, nbatch=nbatch, add_start_char=1, add_end_char=0,
                                print_raw=True,  verbose=verbose)
    print("LOSS", loss)
    if batch_decoding:
        greedy_decode(generator=Generator(hidden_size_decoder=hidden_size_decoder, voc_size=V, verbose=verbose),
                      char_dictionary=char_dictionary, verbose=3,
                      batchIter=batchIter, model=model, batch_size=batch_size)

    if sequence_decoding:
        decode_seq_begins_with(seq_string="eabf", dictionary=char_dictionary, max_len=10, model=model, char_dictionary=char_dictionary,
                           generator=Generator(hidden_size_decoder=hidden_size_decoder, voc_size=V, verbose=verbose),
                           )
    if interactive_mode:
        decode_interacively(dictionary=char_dictionary, max_len=10, model=model, char_dictionary=char_dictionary,
                            generator=Generator(hidden_size_decoder=hidden_size_decoder, voc_size=V, verbose=verbose, output_dim=50),
                            verbose=2)



