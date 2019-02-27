import torch.nn as nn
import torch
import pdb
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import time
EPSILON = 1e-4


class Attention(nn.Module):

    def __init__(self,  hidden_size_word_decoder,
                 char_embedding_dim, hidden_size_src_word_encoder, time=False,
                 method="general",use_gpu=False):

        super(Attention, self).__init__()
        self.time = time
        self.hidden_size_word_decoder = hidden_size_word_decoder
        self.attn = nn.Linear(hidden_size_word_decoder ,#+ char_embedding_dim,
                              hidden_size_src_word_encoder)#+hidden_size, hidden_size) # CHANGE--> (compared to example) we (hidden_size * 2+hidden_size because we have the embedding size +  ..
        self.v = nn.Parameter(torch.FloatTensor(self.hidden_size_word_decoder))
        self.use_gpu = use_gpu
        self.method = method

    def score(self, char_state_decoder, encoder_output):
        if self.method == "concat":
            print("WARNING : Do not understand the self.v.dot + will cause shape error  ")
            energy = self.attn(torch.cat((char_state_decoder, encoder_output), 0))#CHANGE 0 instead of 1
            energy = self.v.dot(energy)
        elif self.method == "general":
            energy = self.attn(char_state_decoder)
            energy = energy.unsqueeze(-1)
            encoder_output = encoder_output.squeeze(-1)#.unsqueeze(1)
            #energy = encoder_output.matmul(energy)
            energy = torch.bmm(encoder_output, energy)
            #energy = energy.squeeze(1).squeeze(1)
            energy = energy.squeeze(-1)
        elif self.method == "bahadanu":
            #TODO
            pass
            #energy = encoder_output.dot(energy)
        return energy

    def forward(self, char_state_decoder, encoder_outputs, word_src_sizes=None):
        max_word_len_src = encoder_outputs.size(1)
        this_batch_size = encoder_outputs.size(0)
        attn_energies = Variable(torch.zeros(this_batch_size, max_word_len_src)) # B x S
        # we loop over all the source encoded sequence (of character) to compute the attention weight
        # is the loop on the batch necessary
        #for batch in range(this_batch_size):
        # index of src word for masking
        batch_diag = torch.empty(encoder_outputs.size(1), len(word_src_sizes),len(word_src_sizes))
        #for word in range(len(encoder_outputs.size(1))):
            #score_index = np.array([i for i in range(len(word)) > word_src_sizes[word]])
            #diag = torch.diag(score_index).float()
            #batch_diag[word,:,:] = diag
        #
        #scores_energy = diag.matmul(scores_energy)
        attn_energies = self.score(char_state_decoder[:, :], encoder_outputs.squeeze(1))
        # scores_energy shaped : number of decoded word (batch x len_sent max) times n_character max src
        # we have a attention energy for the current decoding character for each src word target word pair
        #attn_energies[:, char_src] = diag.matmul(scores_energy)
        # DO WE NEED TO SET THE ENERGY TO -inf to force the softmax to be zero to all padded vector
        softmax = F.softmax(attn_energies, dim=1)
        try:
            ones = torch.ones(softmax.size(0)).cuda() if softmax.is_cuda else torch.ones(softmax.size(0))
            assert ((softmax.sum(dim=1) - ones) < EPSILON).all(), "ERROR : softmax not softmax"
        except:
            print("SOFTMAX0 is not softmax : softmax.size(0)")
            print(softmax.sum(dim=1))
        #  we do masking here : word that are only 1 len (START character) :
        #  we set their softmax to 0 so that their context vector is
        #  Q? is it useful
        # masking of softmax weights : we set the weights to 0 if padded value
        diag_sotm = torch.diag(word_src_sizes != 1).float()

        if softmax.is_cuda:
            diag_sotm = diag_sotm.cuda()
        # masking empty words
        softmax_ = diag_sotm.matmul(softmax) # equivalent to softmax[word_src_sizes == 1, :] = 0. #assert (softmax_2==softmax).all()
        softmax_ = softmax_.unsqueeze(1)
        return softmax_

