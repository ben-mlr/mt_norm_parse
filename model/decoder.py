import torch.nn as nn
import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from io_.info_print import printing
import pdb
import time
from toolbox.sanity_check import get_timing
from collections import OrderedDict
from env.project_variables import SUPPORED_WORD_ENCODER


class CharDecoder(nn.Module):
    def __init__(self, char_embedding, input_dim, hidden_size_decoder, word_recurrent_cell=None,
                 drop_out_word_cell=0,timing=False, drop_out_char_embedding_decoder=0,
                 verbose=0):
        super(CharDecoder, self).__init__()
        self.timing = timing
        self.char_embedding_decoder = char_embedding
        self.drop_out_char_embedding_decoder = nn.Dropout(drop_out_char_embedding_decoder)
        if word_recurrent_cell is not None:
            assert word_recurrent_cell in SUPPORED_WORD_ENCODER, \
                "ERROR : word_recurrent_cell should be in {} ".format(SUPPORED_WORD_ENCODER)
        word_recurrent_cell = nn.GRU if word_recurrent_cell is None else eval("nn."+word_recurrent_cell)
        if isinstance(word_recurrent_cell, nn.LSTM):
            printing("WARNING : in the case of LSTM : inital states defined as "
                     " h_0, c_0 = (zero tensor, source_conditioning) so far (cf. row 70 decoder.py) ",
                     verbose=self.verbose, verbose_level=0)
            printing("WARNING : LSTM only using h_0 for prediction not the  cell", verbose=self.verbose, verbose_level=0)

        self.seq_decoder = word_recurrent_cell(input_size=input_dim, hidden_size=hidden_size_decoder,
                                               num_layers=1,  # nonlinearity='tanh',
                                               dropout=drop_out_word_cell,
                                               bias=True, batch_first=True, bidirectional=False)
        printing("MODEL Decoder : word_recurrent_cell has been set to {} ".format(str(word_recurrent_cell)),
                 verbose=verbose, verbose_level=0)
        self.verbose = verbose

    def word_encoder_target(self, output, conditioning, output_word_len):
        # TODO DEAL WITH MASKING (padding and prediction oriented ?)
        printing("TARGET size {} ", var=output.size(), verbose=self.verbose, verbose_level=3)
        printing("TARGET data {} ", var=output, verbose=self.verbose, verbose_level=5)
        #printing("TARGET mask data {} mask {} ", var=(output_mask, output_mask.size()), verbose=self.verbose, verbose_level=5)
        printing("TARGET  : Word  length  {}  ".format(output_word_len), self.verbose, verbose_level=5)
        output_word_len, perm_idx_output = output_word_len.squeeze().sort(0, descending=True)

        output = output[perm_idx_output, :]
        inverse_perm_idx_output = torch.from_numpy(np.argsort(perm_idx_output.cpu().numpy()))
        # output : [ ]
        start = time.time() if self.timing else None
        char_vecs = self.char_embedding_decoder(output)
        char_vecs = self.drop_out_char_embedding_decoder(char_vecs)
        char_embedding, start = get_timing(start)

        printing("TARGET EMBEDDING size {} ", var=char_vecs.size(), verbose=self.verbose, verbose_level=3) #if False else None
        printing("TARGET EMBEDDING data {} ", var=char_vecs, verbose=self.verbose, verbose_level=5)
        not_printing, start = get_timing(start)
        conditioning = conditioning[:, perm_idx_output, :]
        reorder_conditioning, start = get_timing(start)

        #  USING PACKED SEQUENCE
        # THe shapes are fine !! -->
        printing("TARGET  word lengths after  {} dim", var = output_word_len.size(), verbose=self.verbose, verbose_level=4)
        # same as target sequence and source ..
        output_word_len[output_word_len == 0] = 1
        # pdb.set_trace()
        zero_last, start = get_timing(start)
        packed_char_vecs_output = pack_padded_sequence(char_vecs, output_word_len.squeeze().cpu().numpy(), batch_first=True)
        pack, start = get_timing(start)
        printing("TARGET packed_char_vecs {}  dim", var=packed_char_vecs_output.data.shape, verbose=self.verbose,
                 verbose_level=3)  # .size(), packed_char_vecs)
        # conditioning is the output of the encoder (work as the first initial state of the decoder)
        if isinstance(self.seq_decoder, nn.LSTM):
            conditioning = (torch.zeros_like(conditioning), conditioning)

        output, h_n = self.seq_decoder(packed_char_vecs_output, conditioning)
        h_n = h_n[0] if isinstance(self.seq_decoder, nn.LSTM) else h_n
        recurrent_cell, start = get_timing(start)
        printing("TARGET ENCODED {} output {} h_n (output (includes all the hidden states of last layers), "
                 "last hidden hidden for each dir+layers)", var=(output, h_n), verbose=self.verbose, verbose_level=5)
        printing("TARGET ENCODED  SIZE {} output {} h_n (output (includes all the hidden states of last layers), "
                 "last hidden hidden for each dir+layers)", var=(output.data.shape, h_n.size()), verbose=self.verbose, verbose_level=3)
        output, output_sizes = pad_packed_sequence(output, batch_first=True)
        padd, start = get_timing(start)
        output = output[inverse_perm_idx_output, :, :]

        printing("TARGET ENCODED UNPACKED  {} output {} h_n (output (includes all the hidden states of last layers), "
                 "last hidden hidden for each dir+layers)", var=(output, h_n), verbose=self.verbose,
                 verbose_level=5)

        printing("TARGET ENCODED UNPACKED SIZE {} output {} h_n (output (includes all "
                 "the hidden states of last layers),"
                 "last hidden hidden for each dir+layers)", var=(output.size(), h_n.size()),
                 verbose=self.verbose, verbose_level=3)
        # TODO : output is not shorted in regard to max sent len --> how to handle gold sequence ?
        if self.timing:
             print("WORD TARGET {} ".format(OrderedDict([('char_embedding', char_embedding),
                  ("reorder_conditioning", reorder_conditioning), ("zero_last", zero_last),("not_printing",not_printing),
                  ("pack",pack), ("recurrent_cell", recurrent_cell), ("padd", padd)])))

        return output

    def sent_encoder_target(self, output, conditioning, output_word_len, perm_encoder=None, sent_len_max_source=None ,verbose=0):

        # WARNING conditioning is for now the same for every decoded token
        #printing("TARGET output_mask size {}  mask  {} size length size {} ", var=(output_mask.size(), output_mask.size(),
        #                                                                           output_mask.size()), verbose=verbose,
        #         verbose_level=3)

        start = time.time() if self.timing else None
        _output_word_len = output_word_len.clone()
        clone_len, start = get_timing(start)
        # handle sentence that take the all sequence (
        printing("TARGET SIZE : output_word_len length (before 0 last) : size {} data {} ", var=(_output_word_len.size(),_output_word_len), verbose=verbose,
                 verbose_level=4)
        printing("TARGET  : output  (before 0 last) : size {} data {} ", var=(output.size(), output), verbose=verbose,
                 verbose_level=4)
        _output_word_len[:, -1, :] = 0
        # when input_word_len is 0 means we reached end of sentence
        # TODO : WARNING : is +1 required : as sent with 1 ??
        sent_len = torch.argmin(_output_word_len, dim=1)
        # WARNING : forcint sent_len to be one

        if (sent_len == 0).any():
            printing("WARNING : WE ARE FORCING SENT_LEN in the SOURCE SIDE", verbose=verbose, verbose_level=0)
            sent_len[sent_len==0] += 1

        # sort batch at the sentence length
        sent_len, perm_idx_input_sent = sent_len.squeeze().sort(0, descending=True)
        argmin_squeeze, start = get_timing(start)
        inverse_perm_idx_input_sent = torch.from_numpy(np.argsort(perm_idx_input_sent.cpu().numpy()))
        sorting, start = get_timing(start)
        # [batch x sent_len , dim hidden word level] # this remove empty words
        pdb.set_trace()
        packed_char_vecs_output = pack_padded_sequence(output[perm_idx_input_sent, :, :], sent_len.squeeze().cpu().numpy(), batch_first=True)
        packed_sent, start = get_timing(start)
        # unpacked for the word level representation
        # packed_char_vecs_output .data : [batch x shorted sent_lenS , word lens ] + .batch_sizes
        output_char_vecs, output_sizes = pad_packed_sequence(packed_char_vecs_output, batch_first=True, padding_value=1.0)
        padd_sent, start = get_timing(start)
        # output_char_vecs : [batch ,  shorted sent_len, word len ] + .batch_sizes
        # output_char_vecs : [batch, sent_len max, dim encoder] reorder the sequence
        output_char_vecs = output_char_vecs[inverse_perm_idx_input_sent, :, :]
        # cut input_word_len so that it fits packed_padded sequence
        output_word_len = output_word_len[:, :output_char_vecs.size(1), :]
        # cut again (should be done in one step I guess) to fit sent len source
        output_word_len = output_word_len[:, :sent_len_max_source, :]
        output_seq = output_char_vecs.view(output_char_vecs.size(0) * output_char_vecs.size(1), output_char_vecs.size(2))
        reshape_sent, start = get_timing(start)
        # output_seq : [ batch x max sent len, max word len  ]
        output_word_len = output_word_len.contiguous()
        output_word_len = output_word_len.view(output_word_len.size(0) * output_word_len.size(1))
        reshape_len, start = get_timing(start)
        output_w_decoder = self.word_encoder_target(output_seq, conditioning, output_word_len)
        word_encoders, start = get_timing(start)
        # output_w_decoder : [ batch x max sent len, max word len , hidden_size_decoder ]
        output_w_decoder = output_w_decoder.view(output_char_vecs.size(0),
                                                 output_w_decoder.size(0)/output_char_vecs.size(0), -1,
                                                 output_w_decoder.size(2))
        reshape, start = get_timing(start)
        # output_w_decoder : [ batch , max sent len, max word len , hidden_size_decoder ]
        if self.timing:
            print("SENT TARGET : {}".format(OrderedDict([("clone_len", clone_len), ("argmin_squeeze", argmin_squeeze),("sorting", sorting),
                                                         ("packed_sent", packed_sent), ("padd_sent",padd_sent), ("reshape_sent",reshape_sent),
                                                         ("reshape_len",reshape_len),("word_encoders", word_encoders), ("reshape",reshape)])))
        return output_w_decoder
