import torch.nn as nn
import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from io_.info_print import printing
import time
import pdb
from env.project_variables import SUPPORED_WORD_ENCODER
from io_.dat.constants import PAD_ID_CHAR

class CharEncoder(nn.Module):

    def __init__(self, char_embedding, input_dim, hidden_size_encoder, hidden_size_sent_encoder,
                 word_recurrent_cell=None, dropout_sent_encoder_cell=0, dropout_word_encoder_cell=0,
                 n_layers_word_cell=1, timing=False, bidir_sent=True,context_level="all",
                 drop_out_word_encoder_out=0, drop_out_sent_encoder_out=0,
                 n_layers_sent_cell=1,
                 dir_word_encoder=1, add_word_level=False,
                 word_embedding_dim_inputed=0,
                 verbose=2):
        super(CharEncoder, self).__init__()
        self.char_embedding_ = char_embedding
        self.timing = timing
        self.add_word_level = add_word_level
        word_embedding_dim_inputed = 0 if word_embedding_dim_inputed is None else word_embedding_dim_inputed
        # context level shared to the decoder (should prune a lot if context level word/or sent )
        self.context_level = context_level
        if dir_word_encoder == 2:
            assert hidden_size_encoder % 2 == 0, "ERROR = it will be divided by two and remultipy so need even number for simplicity"
        if bidir_sent:
            assert hidden_size_sent_encoder % 2 == 0, "ERROR = it will be divided by two and remultipy so need even number for simplicity"
        self.sent_encoder = nn.LSTM(input_size=hidden_size_encoder*n_layers_word_cell*dir_word_encoder+word_embedding_dim_inputed,
                                    hidden_size=hidden_size_sent_encoder,
                                    num_layers=n_layers_sent_cell, bias=True, batch_first=True,
                                    dropout=dropout_word_encoder_cell,
                                    bidirectional=bidir_sent)
        self.drop_out_word_encoder_out = nn.Dropout(drop_out_word_encoder_out)
        self.drop_out_sent_encoder_out = nn.Dropout(drop_out_sent_encoder_out)
        self.verbose = verbose
        if word_recurrent_cell is not None:
            assert word_recurrent_cell in SUPPORED_WORD_ENCODER, \
                "ERROR : word_recurrent_cell should be in {} ".format(SUPPORED_WORD_ENCODER)
        word_recurrent_cell = nn.GRU if word_recurrent_cell is None else eval("nn."+word_recurrent_cell)
        self.word_recurrent_cell = word_recurrent_cell
        printing("MODEL Encoder : word_recurrent_cell has been set to {} ", var=([str(word_recurrent_cell)]),
                 verbose=verbose, verbose_level=1)
        self.seq_encoder = word_recurrent_cell(input_size=input_dim, hidden_size=hidden_size_encoder,
                                               dropout=dropout_sent_encoder_cell,
                                               num_layers=n_layers_word_cell,#nonlinearity='tanh',
                                               bias=True, batch_first=True, bidirectional=bool(dir_word_encoder-1))

    def word_encoder_source(self, input, input_word_len=None):
        # input : [word batch dim, max character length],  input_word_len [word batch dim]
        printing("SOURCE dim {} ", var=(input.size()),verbose= self.verbose, verbose_level=3)
        printing("SOURCE DATA {} ", var=(input), verbose=self.verbose, verbose_level=5)
        printing("SOURCE Word lenght size {} ", var=(input_word_len.size()),verbose= self.verbose, verbose_level=5)
        printing("SOURCE : Word  length  {}  ", var=(input_word_len), verbose=self.verbose, verbose_level=3)
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
        printing("SOURCE embedding dim {} ", var=(char_vecs.size()),verbose= self.verbose, verbose_level=3)
        printing("SOURCE  word lengths after  {} dim", var=(input_word_len.size()), verbose=self.verbose, verbose_level=3)
        # As the target sequence : if the word is empty we still encode the sentence (the first PAD symbol)
        #  WARNING : We will be cautious not to take it as input of our SENTENCE ENCODER !
        input_word_len[input_word_len == 0] = 1

        packed_char_vecs = pack_padded_sequence(char_vecs, input_word_len.squeeze().cpu().numpy(), batch_first=True)
        # packed_char_vecs.data : [dim batch x word lenghtS ] with .batch_sizes
        printing("SOURCE Packed data shape {} ", var=(packed_char_vecs.data.shape), verbose=self.verbose, verbose_level=4)
        # all sequence encoding [batch, max seq_len, n_dir x encoding dim] ,

        # last complete hidden state: [dir*n_layer, batch, dim encoding dim]
        output, h_n = self.seq_encoder(packed_char_vecs)
        # see if you really want that
        h_n = h_n[0] if isinstance(self.seq_encoder, nn.LSTM) else h_n
        # TODO add attention out of the output (or maybe output the all output and define attention later)
        printing("SOURCE ENCODED all {}  , hidden {}  (output (includes all the "
                 "hidden states of last layers), last hidden hidden for each dir+layers)", var=(output.data.shape,
                                                                                                h_n.size()),
                 verbose=self.verbose, verbose_level=3)
        output, word_src_sizes = pad_packed_sequence(output, batch_first=True)
        # output : [batch, max word len, dim hidden_size_encoder]
        output = output[inverse_perm_idx, :]
        h_n = h_n[:, inverse_perm_idx, :]

        printing("SOURCE ENCODED UNPACKED {}  , hidden {}  (output (includes all the "
                 "hidden states of last layers), last hidden hidden for each dir+layers)", var=(output.data.shape, h_n.size()),
                 verbose=self.verbose, verbose_level=3)
        # TODO: check that using packed sequence provides the last state of the sequence(not the end of the padded one!)
        return h_n, output, word_src_sizes

    def forward(self, input, input_word_len=None, word_embed_input=None, verbose=0):
        # input should be a batach of sentences
        # input : [batch, max sent len, max word len], input_word_len [batch, max_sent_len]
        context_level = self.context_level
        assert context_level in ["all","word", "sent"], 'context_level : should be in ["all","word", "sent"]'
        printing("SOURCE : input size {}  length size {}",
                 var=(input.size(), input_word_len.size()),
                 verbose=verbose, verbose_level=4)

        _input_word_len = input_word_len.clone()
        # handle sentence that take the all sequence
        # TODO : I think this case problem for sentence that take the all sequence : we are missing a word ! ??
        _input_word_len[:, -1, :] = 0
        # when input_word_len is 0 means we reached end of sentence
        # I think +1 is required : we want the lenght !! so if argmin --> 0 lenght should be 1 right
        sent_len = torch.argmin(_input_word_len, dim=1)
        # we add to sent len if the original src word was filling the entire sequence (i.e last len is not 0)
        pdb.set_trace()
        sent_len += (input_word_len[:, -1, :] != 0).long()
        # sort batch based on sentence length
        sent_len, perm_idx_input_sent = sent_len.squeeze().sort(0, descending=True)
        inverse_perm_idx_input_sent = torch.from_numpy(np.argsort(perm_idx_input_sent.cpu().numpy()))
        # we pack and padd the sentence to shorten and pad sentences
        # [batch x sent_len , dim hidden word level] # this remove empty words
        packed_char_vecs_input = pack_padded_sequence(input[perm_idx_input_sent, :, :],
                                                      sent_len.squeeze().cpu().numpy(), batch_first=True)
        # unpacked for the word level representation
        pdb.set_trace()
        input_char_vecs, input_sizes = pad_packed_sequence(packed_char_vecs_input, batch_first=True,
                                                           padding_value=PAD_ID_CHAR)
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
        pdb.set_trace()
        h_w, char_seq_hidden, word_src_sizes = self.word_encoder_source(input=input_char_vecs, input_word_len=input_word_len)
        # [batch x max sent_len , packed max_char_length, hidden_size_encoder]

        h_w = h_w.transpose(1, 0)

        # n_dir x dim hidden
        hidden_dim = h_w.size(2) * h_w.size(1)
        h_w = h_w.contiguous().view(shape_sent_seq[0], shape_sent_seq[1], hidden_dim)
        # [batch,  max sent_len , packed max_char_length, hidden_size_encoder]
        printing("SOURCE word encoding reshaped dim sent : {} ", var=[h_w.size()],
                 verbose=verbose, verbose_level=3)
        printing("SOURCE char_seq_hidden encoder reshaped dim sent : {} ", var=[char_seq_hidden.size()],
                 verbose=verbose, verbose_level=3)
        if self.add_word_level:
            assert word_embed_input is not None, "ERROR word_embed_input required as self.add_word_level"
            # we trust h_w for padding
            word_embed_input = word_embed_input[:, :h_w.size(1),:]
            h_w = torch.cat((word_embed_input.float(),
                             h_w), dim=-1)
        sent_encoded, hidden = self.sent_encoder(h_w)
        # sent_encoded : upper layer only but all time step, to get all the layers states of the last state get hidden
        # sent_encoded : [batch, max sent len ,hidden_size_sent_encoder]
        printing("SOURCE sentence encoder output dim sent : {} ", var=[sent_encoded.size()],
                 verbose=verbose, verbose_level=3)
        # concatanate
        sent_encoded = self.drop_out_sent_encoder_out(sent_encoded)
        h_w = self.drop_out_word_encoder_out(h_w)
        if context_level == "all":
            #" 'all' means word and sentence level "
            source_context_word_vector = torch.cat((sent_encoded, h_w), dim=2)
        elif context_level == "sent":
            source_context_word_vector = sent_encoded
        elif context_level == "word":
            source_context_word_vector = h_w
        printing("SOURCE contextual for decoding: {} ", var=[source_context_word_vector.size() if source_context_word_vector is not None else 0],
                 verbose=verbose, verbose_level=3)
        pdb.set_trace()
        return source_context_word_vector, sent_len_max_source, char_seq_hidden, word_src_sizes

