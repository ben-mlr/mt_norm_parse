import torch.nn as nn
import torch
import pdb
from torch.autograd import Variable
import torch.nn.functional as F
import time
EPSILON = 1e-6

class Attention(nn.Module):

    def __init__(self,  hidden_size_word_decoder,
                 char_embedding_dim, hidden_size_src_word_encoder, method="general",use_gpu=False):

        super(Attention, self).__init__()
        pdb.set_trace()
        self.hidden_size_word_decoder = hidden_size_word_decoder
        self.attn = nn.Linear(hidden_size_word_decoder + char_embedding_dim, hidden_size_src_word_encoder)#+hidden_size, hidden_size) # CHANGE--> (compared to example) we (hidden_size * 2+hidden_size because we have the embedding size +  ..
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
            energy = encoder_output.dot(energy)
        return energy

    def forward(self, char_state_decoder, encoder_outputs):
        max_word_len_src = encoder_outputs.size(1)
        this_batch_size = encoder_outputs.size(0)
        attn_energies = Variable(torch.zeros(this_batch_size, max_word_len_src)) # B x S

        # we loop over all the source encoded sequence (of character) to compute the attention weight
        # is the loop on the batch necessary
        for batch in range(this_batch_size):
            for char_src in range(max_word_len_src):
                # encoder_outputs[batch, char_src] : contextual character embedding of character ind char_src at batch (word level context) of the source word
                # char_state_decoder[batch, :] : state of the decoder for batch ind (embedding)
                attn_energies[batch, char_src] = self.score(char_state_decoder[batch, :], encoder_outputs[batch, char_src]) # CHANGE : no need of unsquueze ?
        softmax = F.softmax(attn_energies)
        assert ((softmax.sum(dim=1) - torch.ones(F.softmax(attn_energies).size(0))) < EPSILON).all(), "ERROR : softmax not softmax"

        return softmax.unsqueeze(1)

