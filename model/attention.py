from env.importing import *

import torch.nn.functional as F

EPSILON = 1e-3


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
        attn_energies = self.score(char_state_decoder=char_state_decoder,
                                   encoder_output=encoder_outputs.squeeze(1))

        # WARNING : we use encoder_outputs as our masking :
        # it means that we assume encoder_outputs is equal to 0 (at first index) # FINE because we have word_encoder_source which provides pad sequence
        attn_energies[encoder_outputs[:, :, 0] == 0] = -float("Inf")
        softmax = F.softmax(attn_energies, dim=1)
        #try:
        # TOD is kind of costly
        ones = torch.ones(softmax.size(0)).cuda() if softmax.is_cuda else torch.ones(softmax.size(0))
        if not ((softmax.sum(dim=1) - ones) < EPSILON).all():
            print("ERROR : softmax not softmax")
        #except:
            #print("SOFTMAX0 is not softmax : softmax.size(0)")
            #print(softmax.sum(dim=1))
        return softmax.unsqueeze(1)

