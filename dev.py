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
from io_.info_print import printing
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

train_run = True

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
    model = LexNormalizer(generator=Generator, char_embedding_dim=5, voc_size=V, hidden_size_encoder=11, hidden_size_decoder=11, verbose=verbose)
    nbatches = 5

    batchIter = data_gen_conllu(test_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary,
                                batch_size=2, nbatch=nbatches, print_raw=True, verbose=verbose)

    printing("Starting training", verbose=verbose, verbose_level=0)
    loss = run_epoch(batchIter, model, LossCompute(model.generator, verbose=verbose),
                     n_epochs=1, i_epoch=1, n_batches=None, empty_run=False, verbose=verbose)
    printing("END training loss is {} ".format(loss), verbose=verbose, verbose_level=0)
    #print("input seq {} \n  input mask {} \n  output seq {} \n output mask {} \n ".format(batch.input_seq,batch.input_seq_mask, batch.output_seq, batch.output_mask ))

predict_run = False

if predict_run:
    def greedy_decode(generator, model, src_seq, src_mask, src_len, verbose=0):
        with torch.no_grad():
            model.forward()
        decoding_states = model.forward(input_seq=src_seq, output_seq=None, input_mask=src_mask, input_word_len=src_len, output_mask=None, output_word_len=None)
        # [batch, seq_len, V] ? (TODO --> copy it to Generator also
        scores = generator.forward(decoding_states)
        # eacj time step predict the most likely
        prediction = scores.argmax(dim=2)






# TODO :
# I think in the way you coded it : you can't have sequence of character that are not padded twice at the end --> bugs in the argmin --> you have to correct it !!
# then build code to play with the model (write a noisy code --> gives you the prediction)
# plug tensorboard
from torchtext import datasets, data
#TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
#LABEL = data.Field(sequential=False)



# ressources : https://bastings.github.io/annotated_encoder_decoder/
