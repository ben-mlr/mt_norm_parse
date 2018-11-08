import torch.nn as nn
import pdb


class CharEncoder(nn.Module):
    def __init__(self, char_embedding, input_dim, verbose=2):
        super(CharEncoder, self).__init__()
        hidden_size_encoder = 10

        self.char_embedding_ = char_embedding
        self.seq_encoder = nn.RNN(input_size=input_dim, hidden_size= hidden_size_encoder,
                                  num_layers=1, nonlinearity='relu',
                                  bias=True, batch_first=True, bidirectional=False)

    def forward(self, input):
        # [batch, seq_len] , batch of sequences of indexes (that corresponds to character 1-hot encoded)
        char_vecs = self.char_embedding_(input)
        # [batch, seq_len, input_dim] n batch of sequences of embedded character
        output, h_n = self.seq_encoder(char_vecs)
        return h_n


class CharDecoder(nn.Module):
    def __init__(self, char_embedding, input_dim):
        super(CharDecoder, self).__init__()
        hidden_size_decoder = 10
        self.char_embedding_decoder = char_embedding
        self.seq_decoder = nn.RNN(input_size=input_dim, hidden_size= hidden_size_decoder,
                                  num_layers=1, nonlinearity='relu',
                                  bias=True, batch_first=True, bidirectional=False)
    def forward_step(self, output_seq, hidden):
        char_vecs = self.char_embedding_decoder(output_seq)
        output, h_n = self.seq_decoder(char_vecs, hidden)
        return h_n
    def forward(self, output, conditioning):
        char_vecs = self.char_embedding_decoder(output)
        # conditioning is the output of the encoder (work as the first initial state of the decoder)
        print(char_vecs.size(), conditioning.size())
        output, h_n = self.seq_decoder(char_vecs, conditioning)
        return output[:,-1,:]


class LexNormalizer(nn.Module):

    def __init__(self):
        super(LexNormalizer, self).__init__()
        input_dim = 12
        hidden_size_decoder = 10
        voc_size = 9
        self.char_embedding = nn.Embedding(num_embeddings=voc_size, embedding_dim=input_dim)
        self.encoder = CharEncoder(self.char_embedding, input_dim)
        self.decoder = CharDecoder(self.char_embedding, input_dim)
        self.output_predictor = nn.Linear(in_features=hidden_size_decoder, out_features=voc_size)

    def forward(self, input_seq, output_seq):
        # [batch, seq_len] , batch of sequences of indexes (that corresponds to character 1-hot encoded)
        #char_vecs_input = self.char_embedding(input_seq)
        # [batch, seq_len, input_dim] n batch of sequences of embedded character
        h = self.encoder.forward(input_seq)
        # [] [batch, , hiden_size_decoder]

        #char_vecs_output = self.char_embedding(output_seq)
        h_out = self.decoder.forward(output_seq, h)
        pdb.set_trace()
        output_score = nn.ReLU()(self.output_predictor(h_out))
        # [batch, output_voc_size], one score per output character token
        return output_score

    def loss(self,input_seq, output_seq):
        # compute token
        output = self.forward(input_seq, output_seq)
        loss = nn.LogSoftmax()(output)
        return loss









