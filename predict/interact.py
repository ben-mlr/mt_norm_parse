from model.sequence_prediction import greedy_decode, decode_seq_begins_with, decode_interacively
import pdb
from model.loss import LossCompute

from io_.info_print import printing
from model.seq2seq import LexNormalizer, Generator
import torch.nn as nn
import torch
import sys
from io_.data_iterator import data_gen_conllu
from io_.batch_generator import MaskBatch
#sys.path.insert(0,"/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/ELMoLex_sosweet/")
from io_.dat import conllu_data
from training.train import run_epoch


dict_path = "../dictionaries/"
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

model = LexNormalizer(generator=Generator, load=True, model_full_name="6131", dir_model="../checkpoints",
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
