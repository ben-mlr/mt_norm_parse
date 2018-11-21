import torch.nn as nn
import pdb
import torch.nn.functional as F
class CharEncoder(nn.Module):
    def __init__(self, char_embedding, input_dim, hidden_size_encoder, verbose=2):
        super(CharEncoder, self).__init__()

        self.char_embedding_ = char_embedding
        self.verbose = verbose
        self.seq_encoder = nn.RNN(input_size=input_dim, hidden_size=hidden_size_encoder,
                                  num_layers=1, nonlinearity='relu', bias=True, batch_first=True,
                                  bidirectional=False)

    def forward(self, input, input_mask):
        # [batch, seq_len] , batch of sequences of indexes (that corresponds to character 1-hot encoded)
        if self.verbose>=2:
            print("SOURCE", input.size())
        # TODO DEAL WITH MASKING (padding and prediction oriented ?)
        char_vecs = self.char_embedding_(input)
        if self.verbose>=2:
            print("SOURCE EMBEDDING ", char_vecs .size())
        # [batch, seq_len, input_dim] n batch of sequences of embedded character
        output, h_n = self.seq_encoder(char_vecs)
        if self.verbose>=2:
            print("SOURCE ENCODED (output (includes all the hidden states of last layers), "
              "last hidden hidden for each dir+layers)", output.size(), h_n.size())
        return h_n


class CharDecoder(nn.Module):
    def __init__(self, char_embedding, input_dim, hidden_size_decoder, verbose=0):
        super(CharDecoder, self).__init__()
        self.char_embedding_decoder = char_embedding
        self.seq_decoder = nn.RNN(input_size=input_dim, hidden_size=hidden_size_decoder,
                                  num_layers=1, nonlinearity='relu',
                                  bias=True, batch_first=True, bidirectional=False)
        self.verbose = verbose

    def forward_step(self, output_seq, hidden):
        char_vecs = self.char_embedding_decoder(output_seq)
        output, h_n = self.seq_decoder(char_vecs, hidden)
        return h_n

    def forward(self, output, conditioning, output_mask):
        # TODO DEAL WITH MASKING (padding and prediction oriented ?)
        if self.verbose>2:
            print("TARGET ", output.size())
        char_vecs = self.char_embedding_decoder(output)
        if self.verbose>=2:
            print("TARGET EMBEDDING ", char_vecs.size())

        # conditioning is the output of the encoder (work as the first initial state of the decoder)
        if self.verbose>=2:
            print("BEFORE INPUT ", char_vecs.size(), conditioning.size())
        output, h_n = self.seq_decoder(char_vecs, conditioning)
        if self.verbose>=2:
            print("TARGET ENCODED {} output {} h_n (output (includes all the hidden states of last layers), "
              "last hidden hidden for each dir+layers)".format(output.size(), h_n.size()))
        #return output[:, -1, :]
        return output, h_n


class LexNormalizer(nn.Module):

    def __init__(self, generator, char_embedding_dim, hidden_size_encoder, hidden_size_decoder,voc_size, verbose=0):
        super(LexNormalizer, self).__init__()
        # 1 share character embedding layer
        assert hidden_size_decoder == hidden_size_encoder, "Warning : For now {} should equal {} because pf the " \
                                                           "init hidden state (cf. TODO for more flexibility )".format(hidden_size_encoder, hidden_size_decoder)
        #TODO : add projection of hidden_encoder for getting more flexibility
        self.char_embedding = nn.Embedding(num_embeddings=voc_size, embedding_dim=char_embedding_dim)
        self.encoder = CharEncoder(self.char_embedding, input_dim=char_embedding_dim, hidden_size_encoder=hidden_size_encoder, verbose=verbose)
        self.decoder = CharDecoder(self.char_embedding, input_dim=char_embedding_dim, hidden_size_decoder=hidden_size_decoder, verbose=verbose)
        self.generator = generator(hidden_size_decoder=hidden_size_decoder, voc_size=voc_size)
        self.verbose = verbose
        #self.output_predictor = nn.Linear(in_features=hidden_size_decoder, out_features=voc_size)

    def forward(self, input_seq, output_seq, input_mask, output_mask):
        # [batch, seq_len ] , batch of sequences of indexes (that corresponds to character 1-hot encoded)
        #char_vecs_input = self.char_embedding(input_seq)
        # [batch, seq_len, input_dim] n batch of sequences of embedded character

        if self.verbose>=2:
            print("INFO -- ENCODE SOURCE SEQUENCE")
        h = self.encoder.forward(input_seq, input_mask)
        # [] [batch, , hiden_size_decoder]

        #char_vecs_output = self.char_embedding(output_seq)
        if self.verbose>=2:
            print("INFO -- DECODE TARGET SEQUENCE given source code")
        output, h_n = self.decoder.forward(output_seq, h, output_mask)
        #output_score = nn.ReLU()(self.output_predictor(h_out))
        # [batch, output_voc_size], one score per output character token
        # return output
        if self.verbose>=2:
            print("RETURN full  output sequence encoded of size {} ".format(output.size()))
        return output

    # REMOVE FROM HERE

    def loss(self, input_seq, output_seq):
        # compute token
        output = self.forward(input_seq, output_seq)
        loss = nn.LogSoftmax()(output)
        return loss


class Generator(nn.Module):
    " Define standard linear + softmax generation step."
    def __init__(self, hidden_size_decoder, voc_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size_decoder, voc_size)
    # TODO : check if relu is needed or not
    # Is not masking needed here ?

    def forward(self, x):
        # return F.log_softmax(self.proj(x), dim=-1)
        # the log_softmax is done within the loss
        return self.proj(x)

