from env.importing import *
from env.project_variables import EPSILON


class Attention(nn.Module):

    def __init__(self,  hidden_size_word_decoder,
                 char_embedding_dim, hidden_size_src_word_encoder, time=False,
                 method="general",use_gpu=False):

        super(Attention, self).__init__()
        self.time = time
        self.hidden_size_word_decoder = hidden_size_word_decoder
        self.attn = nn.Linear(hidden_size_word_decoder+char_embedding_dim,#+ char_embedding_dim,
                              hidden_size_src_word_encoder)#+hidden_size, hidden_size) # CHANGE--> (compared to example) we (hidden_size * 2+hidden_size because we have the embedding size +  ..
        self.v = nn.Parameter(torch.FloatTensor(self.hidden_size_word_decoder))
        self.use_gpu = use_gpu
        self.method = method

    def score(self, char_state_decoder, encoder_output, char_embedding_current):
        if self.method == "concat":
            print("WARNING : Do not understand the self.v.dot + will cause shape error  ")
            energy = self.attn(torch.cat((char_state_decoder, encoder_output), 0))#CHANGE 0 instead of 1
            energy = self.v.dot(energy)
            raise(Exception("{} not supported so far ".format(self.method)))
        elif self.method == "general":
            # PROJECT DECODER STATE
            char_embedding_current = char_embedding_current.squeeze(1)
            char_state_decoder = torch.cat((char_embedding_current, char_state_decoder), dim=1)
            # [batch size x word max len, dim char embedding + dime char decoder ] :
            #  it's our overall decoder current state
            energy = self.attn(char_state_decoder)
            # [batch size x SENT max len target, hidden size decoder) : it's our query vector of the decoder state
            energy = energy.unsqueeze(-1)
            # [batch sier x sent max len source, max word len source, output dim encoder ] :
            #  1 vector per character for each source word
            encoder_output = encoder_output.squeeze(-1)
            pdb.set_trace()
            # For each pairs (source word,target words) we do encoder.energy
            energy = torch.bmm(encoder_output, energy)
            #energy = energy.squeeze(1).squeeze(1)
            energy = energy.squeeze(-1)
        elif self.method == "bahadanu":
            raise (Exception("{} not supported so far ".format(self.method)))

        return energy

    def forward(self, char_state_decoder, encoder_outputs, char_embedding_current,word_src_sizes=None):
        attn_energies = self.score(char_state_decoder=char_state_decoder,
                                   char_embedding_current=char_embedding_current,
                                   encoder_output=encoder_outputs.squeeze(1))

        # WARNING : we use encoder_outputs as our masking signal :
        # we assume encoder_output has been correctly masked (with pad_sequence) :
        #  when the vector is == 0 it's a masked index : we don't want the attention to focus on it
        attn_energies[encoder_outputs[:, :, 0] == 0] = -float("Inf")
        softmax = F.softmax(attn_energies, dim=1)

        ones = torch.ones(softmax.size(0)).cuda() if softmax.is_cuda else torch.ones(softmax.size(0))
        if not ((softmax.sum(dim=1) - ones) < EPSILON).all():
            print("WARNING : softmax not softmax")
        return softmax.unsqueeze(1)

