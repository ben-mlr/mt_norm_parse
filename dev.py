from model.seq2seq import LexNormalizer, Generator
import torch.nn as nn
import torch
import sys
from io_.data_iterator import data_gen_conllu
from io_.batch_generator import MaskBatch
sys.path.insert(0,"/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/ELMoLex_sosweet/")
from dat import conllu_data

from training.train import run_epoch
from model.loss import LossCompute
from torch.autograd import Variable
import numpy as np
import pdb
import time

#pdb.set_trace = lambda: 1


dict_path = "./dictionaries/"
train_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/en-ud-train.conllu"
dev_pat = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/owoputi.integrated"
test_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/lexnorm.integrated"

word_dictionary, char_dictionary, pos_dictionary, \
xpos_dictionary, type_dictionary = conllu_data.create_dict(dict_path=dict_path,
                                                           train_path=train_path,
                                                           dev_path=dev_pat,
                                                           test_path=test_path,
                                                           word_embed_dict={},
                                                           dry_run=False,
                                                           vocab_trim=True)
print("char_dictionary", char_dictionary.instance2index)
V = len(char_dictionary.instance2index)
print("Character vocabulary is {} length".format(V))
#data_train = conllu_data.read_data_to_variable(test_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary,    use_gpu=0, symbolic_root=True, dry_run=0, lattice=False)
#word, char, pos, xpos, heads, types, masks, lengths, morph = conllu_data.get_batch_variable(data_train,batch_size= 1, unk_replace=0)
#print("-->", char.size(), word.size(), char)
model = LexNormalizer(generator=Generator, char_embedding_dim=5, voc_size=V, hidden_size_encoder=11, hidden_size_decoder=11, verbose=2)
#batch = MaskBatch(input_seq=char[:, 1,:],output_seq=char[:, 1, :], pad=1)


batchIter = data_gen_conllu(test_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary, batch_size=2, nbatch=5)
run_epoch(batchIter, model, LossCompute(model.generator))



#print("input seq {} \n  input mask {} \n  output seq {} \n output mask {} \n ".format(batch.input_seq,batch.input_seq_mask, batch.output_seq, batch.output_mask ))





# TODO :
# HANDLE MASKING in the input and output : within RNN then SOFTMAX (what else ?) same for encoder and decoder !!
# then build code to play with the model (write a noisy code --> gives you the prediction)
# plug tensorboard
from torchtext import datasets, data
#TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
#LABEL = data.Field(sequential=False)

