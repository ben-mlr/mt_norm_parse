import torch.nn as nn
import pdb
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

DEV = True

DEV_2 = False
class CharEncoder(nn.Module):

    def __init__(self, char_embedding, input_dim, hidden_size_encoder, verbose=2):
        super(CharEncoder, self).__init__()

        self.char_embedding_ = char_embedding
        self.verbose = verbose
        self.seq_encoder = nn.RNN(input_size=input_dim, hidden_size=hidden_size_encoder,
                                  num_layers=1, nonlinearity='relu', bias=True, batch_first=True,
                                  bidirectional=False)

    def forward(self, input, input_mask, input_word_len=None):
        # [batch, seq_len] , batch of (already) padded sequences of indexes (that corresponds to character 1-hot encoded)
        if self.verbose>=2:
            print("SOURCE", input.size())
            print("SOURCE", input)
            print("MASK", input_mask)
        if DEV:
            print("{} word lengh of dim {} \n ".format(input_word_len, input_word_len.size()))

            input_word_len, perm_idx = input_word_len.squeeze().sort(0, descending=True)
            print("{} input_word_len new \n {} index \n of dim {} and {} resp ".format(input_word_len, perm_idx, input_word_len.size(), perm_idx.size()))
            # reordering by sequence len
            # [batch, seq_len]
            input = input[perm_idx,:]
            print("SOURCE permuted ", input, input.size())
            #char_vecs = pack_padded_sequence(char_vecs,input_word_len, batch_first=True)
            #padded = pad_packed_sequence()
            #print("PACKING : ",padded, padded.size())

        # [batch, max seq_len, dim char embedding]
        char_vecs = self.char_embedding_(input)

        if self.verbose >= 2:
            print("SOURCE EMBEDDING ", char_vecs.size(), char_vecs)
        if DEV:
            print("0ouptu", input_word_len, input_word_len.squeeze(), input_word_len.size())

            packed_char_vecs = pack_padded_sequence(char_vecs, input_word_len.squeeze().cpu().numpy(), batch_first=True)
            print(" packed_char_vecs ", packed_char_vecs.data.shape, packed_char_vecs)#.size(), packed_char_vecs)
        # all sequence encoding [batch, max seq_len, n_dir x encoding dim] ,
        # last complete hidden state: [dir*n_layer, batch, dim encoding dim]
        if DEV:
            output, h_n = self.seq_encoder(packed_char_vecs)
        else:
            output, h_n = self.seq_encoder(char_vecs)
        if self.verbose >= 2:
            print("LEN", input_word_len, input_word_len.size())
            print(h_n)
            print(h_n[-1])
            print(h_n.size())
            print("SOURCE ENCODED (output (includes all the hidden states of last layers), "
                  "last hidden hidden for each dir+layers)", output.data.shape, h_n.size())
        # TODO : check that usinh packed sequence indded privdes the last state of the sequence (not the end of the padded one ! )
        # + check this dimension ? why are we loosing a dimension
        return h_n[-1]


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

    def forward(self, output, conditioning, output_mask, output_word_len):
        # TODO DEAL WITH MASKING (padding and prediction oriented ?)
        if self.verbose >= 2:
            print("TARGET ", output.size())
        if DEV and DEV_2:
            output_word_len, perm_idx_output = output_word_len.squeeze().sort(0, descending=True)
            if self.verbose >= 2:
                print("{} input_word_len new \n {} index \n of dim {} and {} resp ".format(output_word_len, perm_idx_output, output_word_len.size(), perm_idx_output.size()))
            output = output[perm_idx_output,:]
            if self.verbose >= 2:
                print("TARGET permuted ", output, output.size())

        char_vecs = self.char_embedding_decoder(output)

        if self.verbose>=2:
            print("TARGET EMBEDDING ", char_vecs.size())

        if DEV and DEV_2:
            # TODO : decoding problem here  : Pb in the loss !
            # THe shapes are fine !! -->
            print("word len 1ouptu", output_word_len, output_word_len.squeeze(), output_word_len.size())
            packed_char_vecs_output = pack_padded_sequence(char_vecs, output_word_len.squeeze().cpu().numpy(), batch_first=True)
            print(" packed_char_vecs TARGET ", packed_char_vecs_output.data.shape, packed_char_vecs_output)#.size(), packed_char_vecs)


        # conditioning is the output of the encoder (work as the first initial state of the decoder)
        if self.verbose >= 2:
            print("BEFORE INPUT ", char_vecs.data.shape, conditioning.data.shape)
        if DEV and DEV_2:
            output, h_n = self.seq_decoder(packed_char_vecs_output, conditioning.unsqueeze(0))
            print("DECODING", output, output.data.shape)
            output, output_sizes = pad_packed_sequence(output, batch_first=True)
        else:
            output, h_n = self.seq_decoder(char_vecs, conditioning.unsqueeze(0))#conditioning)
        if self.verbose >= 2:

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

    def forward(self, input_seq, output_seq, input_mask, input_word_len, output_mask, output_word_len):
        # [batch, seq_len ] , batch of sequences of indexes (that corresponds to character 1-hot encoded)
        #char_vecs_input = self.char_embedding(input_seq)
        # [batch, seq_len, input_dim] n batch of sequences of embedded character

        if self.verbose >= 2:
            print("INFO -- ENCODE SOURCE SEQUENCE")
        h = self.encoder.forward(input_seq, input_mask, input_word_len)
        # [] [batch, , hiden_size_decoder]
        #char_vecs_output = self.char_embedding(output_seq)
        if self.verbose >= 2:
            print("INFO -- DECODE TARGET SEQUENCE given source code")
        output, h_n = self.decoder.forward(output_seq, h, output_mask, output_word_len)
        # output_score = nn.ReLU()(self.output_predictor(h_out))
        # [batch, output_voc_size], one score per output character token
        # return output
        if self.verbose >= 2:
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

