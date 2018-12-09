from model.seq2seq import LexNormalizer, Generator
import torch.nn as nn
import torch
import sys
from io_.data_iterator import data_gen_conllu
from io_.batch_generator import MaskBatch
#sys.path.insert(0,"/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/ELMoLex_sosweet/")
from io_.dat import conllu_data
from io_.info_print import print_char_seq
from training.train import run_epoch
from model.loss import LossCompute
from torch.autograd import Variable
from model.sequence_prediction import greedy_decode, decode_seq_begins_with, decode_interacively

import numpy as np
import pdb

import time
from io_.info_print import printing
from io_.from_array_to_text import output_text

pdb.set_trace = lambda: 1

# verbose typology :
## 0 is only starting , end with final loss
## 1 includes 0 + epoch-wise information : loss, + info about the epochs
## 2 includes 0 + 1 + batch wise information like loss êr batch + summary info on each batch
## 3 includes 0 + 1 + 2 + dimensions information of each tensors of the input, output the model, the loss
## 4 : add masking info + packed_sequence info
## 5 : printint data
dict_path = "./dictionaries/"
train_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/en-ud-train.conllu"
dev_pat = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/owoputi.integrated"
test_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/lexnorm.integrated"

train_run = False

if train_run:

    word_dictionary, char_dictionary, pos_dictionary, \
    xpos_dictionary, type_dictionary = conllu_data.create_dict(dict_path=dict_path,
                                                               train_path=train_path,
                                                               dev_path=dev_pat,
                                                               test_path=test_path,
                                                               word_embed_dict={},
                                                               dry_run=False,
                                                               vocab_trim=True)

    V = len(char_dictionary.instance2index)
    print("Character vocabulary is {} length".format(V))

    verbose = 5
    model = LexNormalizer(generator=Generator, char_embedding_dim=5, voc_size=V,
                          hidden_size_encoder=11,
                          hidden_size_decoder=11, verbose=verbose)
    nbatches = 5

    batchIter = data_gen_conllu(test_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary,
                                batch_size=2, nbatch=nbatches, print_raw=True, verbose=verbose)

    printing("Starting training", verbose=verbose, verbose_level=0)
    loss = run_epoch(batchIter, model, LossCompute(model.generator, verbose=verbose),
                     n_epochs=1, i_epoch=1, n_batches=None, empty_run=False, verbose=verbose)
    printing("END training loss is {} ".format(loss), verbose=verbose, verbose_level=0)
    #print("input seq {} \n  input mask {} \n  output seq {} \n output mask {} \n ".format(batch.input_seq,batch.input_seq_mask, batch.output_seq, batch.output_mask ))

predict_run = True

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

    model = LexNormalizer(generator=Generator, load=True, model_full_name="84d7", dir_model="./checkpoints",
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



# TODO :
# 1 add WORD ERROR RATE + CHARACTER ERROR RATE + EDIT DISTANCE RATE comparison with gold
# 2 evaluate on reproduction
# 3 quick experiments on lexnorm on predicting NormPar + LexNorm + Owuputi + EWT: Norm=
# TODO : when you do packed sequence you do teacher force : add inference-like training
# TODO : add stop symbol in the data + at decoding time
# TODO : add quick training report

# TODO :
# then build code to play with the model (write a noisy code --> gives you the prediction)
# plug tensorboard



# ressources : https://bastings.github.io/annotated_encoder_decoder/
#              https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

# NB : batch_size is in data_gen_conllu relates to the collu format setence level !!
## NB you can't have batch_size == 1 WHY ??