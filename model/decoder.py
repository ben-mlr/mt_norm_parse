import torch.nn as nn
import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from io_.info_print import printing


class CharDecoder(nn.Module):
    def __init__(self, char_embedding, input_dim, hidden_size_decoder, word_recurrent_cell=None,
                 verbose=0):
        super(CharDecoder, self).__init__()
        self.char_embedding_decoder = char_embedding
        word_recurrent_cell = nn.GRU if word_recurrent_cell is None else nn.GRU
        self.seq_decoder = word_recurrent_cell(input_size=input_dim, hidden_size=hidden_size_decoder,
                                               num_layers=1,  # nonlinearity='tanh',
                                               bias=True, batch_first=True, bidirectional=False)
        printing("MODEL Decoder : word_recurrent_cell has been set to {} ".format(str(word_recurrent_cell)),
                 verbose=verbose, verbose_level=0)
        self.verbose = verbose

    def word_encoder_target(self, output, conditioning, output_mask, output_word_len):
        # TODO DEAL WITH MASKING (padding and prediction oriented ?)
        printing("TARGET size {} ".format(output.size()), verbose=self.verbose, verbose_level=3)
        printing("TARGET data {} ".format(output), verbose=self.verbose, verbose_level=5)
        printing("TARGET mask data {} mask {} ".format(output_mask, output_mask.size()), verbose=self.verbose,
                 verbose_level=6)
        printing("TARGET  : Word  length  {}  ".format(output_word_len), self.verbose, verbose_level=5)
        output_word_len, perm_idx_output = output_word_len.squeeze().sort(0, descending=True)

        output = output[perm_idx_output, :]
        inverse_perm_idx_output = torch.from_numpy(np.argsort(perm_idx_output.cpu().numpy()))
        # output : [ ]
        char_vecs = self.char_embedding_decoder(output)

        printing("TARGET EMBEDDING size {} ".format(char_vecs.size()), verbose=self.verbose, verbose_level=3)
        printing("TARGET EMBEDDING data {} ".format(char_vecs), verbose=self.verbose, verbose_level=5)

        conditioning = conditioning[:, perm_idx_output, :]

        #  USING PACKED SEQUENCE
        # THe shapes are fine !! -->
        printing("TARGET  word lengths after  {} dim".format(output_word_len.size()), self.verbose, verbose_level=4)
        # same as target sequence and source ..
        output_word_len[output_word_len == 0] = 1
        # pdb.set_trace()
        packed_char_vecs_output = pack_padded_sequence(char_vecs, output_word_len.squeeze().cpu().numpy(), batch_first=True)
        printing("TARGET packed_char_vecs {}  dim".format(packed_char_vecs_output.data.shape), verbose=self.verbose, verbose_level=3  )  # .size(), packed_char_vecs)
        # conditioning is the output of the encoder (work as the first initial state of the decoder)
        output, h_n = self.seq_decoder(packed_char_vecs_output, conditioning)
        printing("TARGET ENCODED {} output {} h_n (output (includes all the hidden states of last layers), "
                 "last hidden hidden for each dir+layers)".format(output, h_n), verbose=self.verbose,
                 verbose_level=5)
        printing("TARGET ENCODED  SIZE {} output {} h_n (output (includes all the hidden states of last layers), "
                 "last hidden hidden for each dir+layers)".format(output.data.shape, h_n.size()), verbose=self.verbose, verbose_level=3)
        output, output_sizes = pad_packed_sequence(output, batch_first=True)
        output = output[inverse_perm_idx_output, :, :]

        printing("TARGET ENCODED UNPACKED  {} output {} h_n (output (includes all the hidden states of last layers), "
                 "last hidden hidden for each dir+layers)".format(output, h_n), verbose=self.verbose,
                 verbose_level=5)

        printing("TARGET ENCODED UNPACKED SIZE {} output {} h_n (output (includes all "
                 "the hidden states of last layers),"
                 "last hidden hidden for each dir+layers)".format(output.size(), h_n.size()),
                 verbose=self.verbose, verbose_level=3)
        # TODO : output is not shorted in regard to max sent len --> how to handle gold sequence ?
        return output

    def sent_encoder_target(self, output, conditioning, output_mask, output_word_len, perm_encoder=None, sent_len_max_source=None ,verbose=0):

        # WARNING conditioning is for now the same for every decoded token
        printing("TARGET output_mask size {}  mask  {} size length size {} ".format(output_mask.size(), output_mask.size(),
                                                                             output_mask.size()), verbose=verbose,
                 verbose_level=3)
        _output_word_len = output_word_len.clone()
        # handle sentence that take the all sequence (
        printing("TARGET SIZE : output_word_len length (before 0 last) : size {} data {} ".format(_output_word_len.size(),_output_word_len), verbose=verbose,
                 verbose_level=4)
        printing("TARGET  : output  (before 0 last) : size {} data {} ".format(output.size(), output), verbose=verbose,
                 verbose_level=4)
        _output_word_len[:, -1, :] = 0
        # when input_word_len is 0 means we reached end of sentence
        sent_len = torch.argmin(_output_word_len, dim=1)
        # sort batch at the sentence length
        sent_len, perm_idx_input_sent = sent_len.squeeze().sort(0, descending=True)
        inverse_perm_idx_input_sent = torch.from_numpy(np.argsort(perm_idx_input_sent.cpu().numpy()))
        # [batch x sent_len , dim hidden word level] # this remove empty words
        packed_char_vecs_output = pack_padded_sequence(output[perm_idx_input_sent, :, :],
                                                       sent_len.squeeze().cpu().numpy(), batch_first=True)
        # unpacked for the word level representation
        # packed_char_vecs_output .data : [batch x shorted sent_lenS , word lens ] + .batch_sizes
        output_char_vecs, output_sizes = pad_packed_sequence(packed_char_vecs_output, batch_first=True,
                                                             padding_value=1.0)
        # output_char_vecs : [batch ,  shorted sent_len, word len ] + .batch_sizes
        # output_char_vecs : [batch, sent_len max, dim encoder] reorder the sequence
        output_char_vecs = output_char_vecs[inverse_perm_idx_input_sent, :, :]
        # cut input_word_len so that it fits packed_padded sequence
        output_word_len = output_word_len[:, :output_char_vecs.size(1), :]
        # cut again (should be done in one step I guess) to fit sent len source
        output_word_len = output_word_len[:, :sent_len_max_source, :]
        output_seq = output_char_vecs.view(output_char_vecs.size(0) * output_char_vecs.size(1), output_char_vecs.size(2))
        # output_seq : [ batch x max sent len, max word len  ]
        output_word_len = output_word_len.contiguous()
        output_word_len = output_word_len.view(output_word_len.size(0) * output_word_len.size(1))
        output_w_decoder = self.word_encoder_target(output_seq, conditioning, output_mask, output_word_len)
        # output_w_decoder : [ batch x max sent len, max word len , hidden_size_decoder ]
        output_w_decoder = output_w_decoder.view(output_char_vecs.size(0),
                                                 output_w_decoder.size(0 )/output_char_vecs.size(0), -1,
                                                 output_w_decoder.size(2))
        # output_w_decoder : [ batch , max sent len, max word len , hidden_size_decoder ]
        return output_w_decoder
