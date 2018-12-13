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


dict_path = "../dictionaries/"
train_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/en-ud-train.conllu"
dev_pat = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/owoputi.integrated"
test_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/lexnorm.integrated"

normalization = True
add_start_char = 1

word_dictionary, char_dictionary, pos_dictionary,\
xpos_dictionary, type_dictionary = \
        conllu_data.create_dict(dict_path=dict_path,
                                train_path=train_path,
                                dev_path=dev_pat,
                                test_path=None,
                                add_start_char=add_start_char,
                                word_embed_dict={},
                                dry_run=False,
                                vocab_trim=True)

verbose = 2

#verbose = 2
#3b87
#1782
#cd05
script_dir = os.path.dirname(os.path.realpath(__file__))

model = LexNormalizer(generator=Generator, load=True, model_full_name="auto_encoder_TEST_93a3", dir_model=os.path.join(script_dir,"..","checkpoints"),
                      verbose=verbose)
batch_size = 2
nbatch = 50
verbose = 2
batchIter = data_gen_conllu("/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/lexnorm.integrated", word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                            type_dictionary, batch_size=batch_size, nbatch=nbatch, add_start_char=add_start_char,
                            add_end_char=0,
                            normalization=normalization,
                            print_raw=True,  verbose=verbose)

V = model.arguments["hyperparameters"]["voc_size"]
hidden_size_decoder = model.arguments["hyperparameters"]["hidden_size_decoder"]
model.eval()

batch_decoding, sequence_decoding, interactive_mode = False, False, True

#loss = run_epoch(batchIter, model, LossCompute(model.generator, verbose=verbose),
#                     i_epoch=0, n_epochs=1,
#                     verbose=verbose,
#                     log_every_x_batch=100)
#print("LOSS", loss)
if batch_decoding:
    greedy_decode_batch(char_dictionary=char_dictionary, verbose=2, gold_output=True,evaluation_metric="mean",
                               batchIter=batchIter, model=model, batch_size=batch_size)

if sequence_decoding:
    decode_seq_str(seq_string="eabf", dictionary=char_dictionary, max_len=10, model=model, char_dictionary=char_dictionary,
                   generator=Generator(hidden_size_decoder=hidden_size_decoder, voc_size=V, verbose=verbose),)
if interactive_mode:
    decode_interacively(dictionary=char_dictionary, max_len=10, model=model, char_dictionary=char_dictionary,
                        generator=Generator(hidden_size_decoder=hidden_size_decoder, voc_size=V, verbose=verbose, output_dim=50),
                        verbose=2)
