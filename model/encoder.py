import torch.nn as nn
import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from io_.info_print import printing

DEV = True
DEV_2 = True
DEV_4 = True
DEV_5 = True


class CharEncoder(nn.Module):

    def __init__(self, char_embedding, input_dim, hidden_size_encoder, hidden_size_sent_encoder,
                 word_recurrent_cell=None,
                 verbose=2):
        super(CharEncoder, self).__init__()
        self.char_embedding_ = char_embedding
        self.sent_encoder = nn.LSTM(input_size=hidden_size_encoder,
                                    hidden_size=hidden_size_sent_encoder,
                                    num_layers=1, bias=True, batch_first=True,
                                    bidirectional=False)
        self.verbose = verbose
        word_recurrent_cell = nn.GRU if word_recurrent_cell is None else nn.GRU
        printing("MODEL Encoder : word_recurrent_cell has been set to {} ".format(str(word_recurrent_cell)),
                 verbose=verbose, verbose_level=0)
        self.word_encoder = word_recurrent_cell(input_size=input_dim, hidden_size=hidden_size_encoder,
                                                num_layers=1, #nonlinearity='tanh',
                                                bias=True, batch_first=True, bidirectional=False)

    def word_encoder_source(self, input, input_word_len=None):
        # input : [word batch dim, max character length],  input_word_len [word batch dim]
        printing("SOURCE dim {} ".format(input.size()), self.verbose, verbose_level=3)
        printing("SOURCE DATA {} ".format(input), self.verbose, verbose_level=5)
        printing("SOURCE Word lenght size {} ".format(input_word_len.size()), self.verbose, verbose_level=5)
        printing("SOURCE : Word  length  {}  ".format(input_word_len), self.verbose, verbose_level=3)
        _input_word_len = input_word_len.clone()
        input_word_len, perm_idx = input_word_len.squeeze().sort(0, descending=True)
        # [batch, seq_len]
        _inp = input.clone()
        # reordering by sequence len
        input = input[perm_idx, :]
        inverse_perm_idx = torch.from_numpy(np.argsort(perm_idx.cpu().numpy()))
        assert torch.equal(input[inverse_perm_idx, :], _inp), " ERROR : two tensors should be equal but are not " # TODO : to remove when code stable enough
        char_vecs = self.char_embedding_(input)
        # char_vecs : [word batch dim, dim char_embedding]
        printing("SOURCE embedding dim {} ".format(char_vecs.size()), self.verbose, verbose_level=3)
        printing("SOURCE  word lengths after  {} dim".format(input_word_len.size()), self.verbose, verbose_level=3)
        # As the target sequence : if the word is empty we still encode the sentence (the first PAD symbol)
        #  WARNING : We will be cautious not to take it as input of our SENTENCE ENCODER !
        input_word_len[input_word_len == 0] = 1
        packed_char_vecs = pack_padded_sequence(char_vecs, input_word_len.squeeze().cpu().numpy(), batch_first=True)
        # packed_char_vecs.data : [dim batch x word lenghtS ] with .batch_sizes
        printing("SOURCE Packed data shape {} ".format(packed_char_vecs.data.shape), self.verbose, verbose_level=4)
        # all sequence encoding [batch, max seq_len, n_dir x encoding dim] ,
        # last complete hidden state: [dir*n_layer, batch, dim encoding dim]
        output, h_n = self.word_encoder(packed_char_vecs)
        # TODO add attention out of the output (or maybe output the all output and define attention later)
        printing("SOURCE ENCODED all {}  , hidden {}  (output (includes all the "
                 "hidden states of last layers), last hidden hidden for each dir+layers)".format(output.data.shape,
                                                                                                 h_n.size()),
                 self.verbose, verbose_level=3)
        output, _ = pad_packed_sequence(output, batch_first=True)
        # output : [batch, max word len, dim hidden_size_encoder]
        output = output[inverse_perm_idx, :]
        h_n = h_n[:, inverse_perm_idx, :]

        printing("SOURCE ENCODED UNPACKED {}  , hidden {}  (output (includes all the "
                 "hidden states of last layers), last hidden hidden for each dir+layers)".format(output.data.shape, h_n.size()),
                 self.verbose, verbose_level=3)
        # TODO: check that using packed sequence provides the last state of the sequence(not the end of the padded one!)
        return h_n

    def sent_encoder_source(self, input, input_mask, input_word_len=None, verbose=0):
        # input should be a batach of sentences
        # input : [batch, max sent len, max word len], input_word_len [batch, max_sent_len]
        printing("SOURCE : input size {}  length size {}" .format(input.size(), input_word_len.size()),
                 verbose=verbose, verbose_level=4)
        _input_word_len = input_word_len.clone()
        # handle sentence that take the all sequence
        _input_word_len[:, -1, :] = 0
        # when input_word_len is 0 means we reached end of sentence
        sent_len = torch.argmin(_input_word_len, dim=1)
        # sort batch based on sentence length
        sent_len, perm_idx_input_sent = sent_len.squeeze().sort(0, descending=True)
        inverse_perm_idx_input_sent = torch.from_numpy(np.argsort(perm_idx_input_sent.cpu().numpy()))
        # we pack and padd the sentence to shorten and pad sentences
        # [batch x sent_len , dim hidden word level] # this remove empty words
        packed_char_vecs_input = pack_padded_sequence(input[perm_idx_input_sent, :, :],
                                                      sent_len.squeeze().cpu().numpy(), batch_first=True)
        # unpacked for the word level representation
        input_char_vecs, input_sizes = pad_packed_sequence(packed_char_vecs_input, batch_first=True,
                                                           padding_value=1.0)
        # [batch, sent_len max, dim encoder] reorder the sequence
        input_char_vecs = input_char_vecs[inverse_perm_idx_input_sent, :, :]
        # cut input_word_len so that it fits packed_padded sequence
        input_word_len = input_word_len[:, :input_char_vecs.size(1), :]
        sent_len_max_source = input_char_vecs.size(1)
        input_word_len = input_word_len.contiguous()
        # reshape word_len and word_char_vecs matrix --> for feeding to word level encoding
        input_word_len = input_word_len.view(input_word_len.size(0) * input_word_len.size(1))
        shape_sent_seq = input_char_vecs.size()
        input_char_vecs = input_char_vecs.view(input_char_vecs.size(0) * input_char_vecs.size(1),
                                               input_char_vecs.size(2))

        # input_char_vecs : [batch x max sent_len , MAX_CHAR_LENGTH or bucket max_char_length]
        # input_word_len  [batch x max sent_len]
        h_w = self.word_encoder_source(input=input_char_vecs, input_word_len=input_word_len)
        # [batch x max sent_len , packed max_char_length, hidden_size_encoder]
        h_w = h_w.view(shape_sent_seq[0], shape_sent_seq[1], -1)
        # [batch,  max sent_len , packed max_char_length, hidden_size_encoder]
        sent_encoded, hidden = self.sent_encoder(h_w)
        # sent_encoded : [batch, max sent len ,hidden_size_sent_encoder]
        printing("SOURCE sentence encoder output dim sent : {} ".format(sent_encoded.size()),
                 verbose=verbose, verbose_level=3)
        # concatanate
        source_context_word_vector = torch.cat((sent_encoded, h_w), dim=2)
        source_context_word_vector = source_context_word_vector.view(1, source_context_word_vector.size(0)*source_context_word_vector.size(1), -1)
        # source_context_word_vector : [1, batch x sent len, hidden_size_sent_encoder + hidden_size_encoder]
        printing("SOURCE contextual last representation : {} ".format(source_context_word_vector.size()),
                 verbose=verbose, verbose_level=3)

        return source_context_word_vector, sent_len_max_source
