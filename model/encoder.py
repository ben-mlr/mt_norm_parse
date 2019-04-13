from env.importing import *


from io_.info_print import printing
from env.project_variables import SUPPORED_WORD_ENCODER
from io_.dat.constants import PAD_ID_CHAR
from io_.dat.constants import MAX_CHAR_LENGTH


class CharEncoder(nn.Module):

    def __init__(self, char_embedding, input_dim, hidden_size_encoder, hidden_size_sent_encoder,
                 word_recurrent_cell=None, dropout_sent_encoder_cell=0, dropout_word_encoder_cell=0,
                 n_layers_word_cell=1, timing=False, bidir_sent=True,context_level="all",
                 drop_out_word_encoder_out=0, drop_out_sent_encoder_out=0,
                 n_layers_sent_cell=1, attention_tagging=False,
                 char_level_embedding_projection_dim=0, word_embedding_dim_inputed=0,
                 dir_word_encoder=1, add_word_level=False,
                 mode_word_encoding="cat",
                 verbose=2):
        super(CharEncoder, self).__init__()
        assert mode_word_encoding in ["cat", "sum"], "ERROR : only cat sum supported for aggregating word representation"
        self.mode_word_encoding = mode_word_encoding
        self.char_embedding_ = char_embedding
        self.timing = timing
        self.add_word_level = add_word_level
        word_embedding_dim_inputed = 0 if word_embedding_dim_inputed is None else word_embedding_dim_inputed
        # context level shared to the decoder (should prune a lot if context level word/or sent )
        self.context_level = context_level
        # attention query is based on the last layer
        self.attention_query = nn.Linear(hidden_size_encoder*1*dir_word_encoder, 1, bias=False) if attention_tagging else None
        # attention projection project
        dim_last_layer_state = hidden_size_encoder * n_layers_word_cell * dir_word_encoder
        dim_attention = (hidden_size_encoder * dir_word_encoder) * int(attention_tagging)
        if n_layers_word_cell>1:
            printing("WARNING : ENCODER : n_layers_word_cell is {} : all layers of cell/hidden state "
                     "are used", var=n_layers_word_cell, verbose=verbose, verbose_level=1)
        if n_layers_sent_cell >1:
            printing("WARNING : ENCODER : n_layers_sent_cell is {} : all layers of cell/hidden state "
                     "are used", var=n_layers_sent_cell, verbose=verbose, verbose_level=1)
        self.char_level_encoding_projection = nn.Linear(dim_attention+dim_last_layer_state,
                                                        char_level_embedding_projection_dim) if char_level_embedding_projection_dim >0 else None
        if dir_word_encoder == 2:
            assert hidden_size_encoder % 2 == 0, "ERROR = it will be divided " \
                                                 "by two and remultipy so need even number for simplicity"
        if bidir_sent:
            assert hidden_size_sent_encoder % 2 == 0, "ERROR = it will be divided by " \
                                                      "two and remultipy so need even number for simplicity"

        # if attention the state is concatanated with the cell state (*2)
        char_level_rnn_output = hidden_size_encoder*n_layers_word_cell*dir_word_encoder if not attention_tagging else 2 * hidden_size_encoder * n_layers_word_cell*dir_word_encoder
        char_level_embedding_input_dim = char_level_embedding_projection_dim if char_level_embedding_projection_dim > 0 else char_level_rnn_output

        if mode_word_encoding == "sum":
            assert char_level_embedding_input_dim == word_embedding_dim_inputed, \
                "ERROR : in sum : they should be same dimension but are char:{} and word:{} ".format(
                    char_level_embedding_input_dim, word_embedding_dim_inputed)
            dim_input_sentence_encoder = char_level_embedding_projection_dim
        elif mode_word_encoding == "cat":
            dim_input_sentence_encoder = char_level_embedding_input_dim + word_embedding_dim_inputed

        self.drop_out_word_encoder_out = nn.Dropout(drop_out_word_encoder_out)
        self.drop_out_sent_encoder_out = nn.Dropout(drop_out_sent_encoder_out)
        self.verbose = verbose
        if word_recurrent_cell is not None:
            assert word_recurrent_cell in SUPPORED_WORD_ENCODER, \
                "ERROR : word_recurrent_cell should be in {} ".format(SUPPORED_WORD_ENCODER)
        if word_recurrent_cell is None:
            word_recurrent_cell = nn.GRU
            printing("MODEL word_recurrent_cell as no argument passed was set to default {} ", var=[word_recurrent_cell], verbose_level=1, verbose=verbose)
        elif word_recurrent_cell == "WeightDropLSTM":
            word_recurrent_cell = eval(word_recurrent_cell)
        else:
            word_recurrent_cell = eval("nn."+word_recurrent_cell)

        self.word_recurrent_cell = word_recurrent_cell

        printing("MODEL Encoder : word_recurrent_cell has been set to {} ", var=([str(word_recurrent_cell)]),
                 verbose=verbose, verbose_level=1)

        self.seq_encoder = word_recurrent_cell(input_size=input_dim, hidden_size=hidden_size_encoder,
                                               dropout=dropout_word_encoder_cell,
                                               num_layers=n_layers_word_cell,
                                               bias=True, batch_first=True, bidirectional=bool(dir_word_encoder-1))
        self.output_encoder_dim = 0
        if self.context_level in ["sent", "all"]:
            self.output_encoder_dim += hidden_size_sent_encoder*(int(bidir_sent)+1)
            self.sent_encoder = nn.LSTM(input_size=dim_input_sentence_encoder,
                                        hidden_size=hidden_size_sent_encoder,
                                        num_layers=n_layers_sent_cell, bias=True, batch_first=True,
                                        dropout=dropout_sent_encoder_cell,
                                        bidirectional=bool(bidir_sent))
        else:
            self.sent_encoder = None
            printing("WARNING : parameters hidden_size_sent_encoder, n_layers_sent_cell and bidir_sent will be ignored "
                     "as self.context_level is {}".format(self.context_level),
                     verbose=verbose, verbose_level=1)
        if self.context_level in ["word", "all"]:
            self.output_encoder_dim += dim_input_sentence_encoder

    def word_encoder_source(self, input, input_word_len=None):
        # input : [word batch dim, max character length],  input_word_len [word batch dim]
        printing("SOURCE dim {} ", var=(input.size()),verbose=self.verbose, verbose_level=3)
        printing("SOURCE DATA {} ", var=(input), verbose=self.verbose, verbose_level=5)
        printing("SOURCE Word lenght size {} ", var=(input_word_len.size()),verbose= self.verbose, verbose_level=5)
        printing("SOURCE : Word  length  {}  ", var=(input_word_len), verbose=self.verbose, verbose_level=3)
        _input_word_len = input_word_len.clone() # sanity check purpose
        input_word_len, perm_idx = input_word_len.squeeze().sort(0, descending=True)
        # [batch, seq_len]
        _inp = input.clone() # sanity check purpose
        # reordering by sequence len
        input = input[perm_idx, :]
        char_vecs = self.char_embedding_(input)
        # char_vecs : [word batch dim, dim char_embedding]
        printing("SOURCE embedding dim {} ", var=(char_vecs.size()),verbose= self.verbose, verbose_level=3)
        printing("SOURCE  word lengths after  {} dim", var=(input_word_len.size()), verbose=self.verbose, verbose_level=3)
        # As the target sequence : if the word is empty we still encode the sentence (the first PAD symbol)
        #  WARNING : We will be cautious not to take it as input of our SENTENCE ENCODER !
        # DEPRECIATED we might have empty words (only pad symbol) because we packed and padded at sentence level
        # We take care of cutting empty words here
        char_vecs = char_vecs[input_word_len != 0, :, :]
        perm_idx = perm_idx[input_word_len != 0]
        input = input[input_word_len != 0, :] # needed within attention only (and for sanity check)
        input_word_len = input_word_len[input_word_len != 0]
        packed_char_vecs = pack_padded_sequence(char_vecs, input_word_len.squeeze().cpu().numpy(), batch_first=True)
        # we can now inverse permutation(we permuted the sequence, we removed, now we want the inverse permutation
        inverse_perm_idx = torch.from_numpy(np.argsort(perm_idx.cpu().numpy()))
        # SANITY CHECKING
        assert torch.equal(input[inverse_perm_idx, :], _inp[_input_word_len!=0,:]), \
            " ERROR : two tensors should be equal but are not {} and {}  ".format(_inp.size(), input.size())  # TODO : to remove when code stable enough
        # packed_char_vecs.data : [dim batch x word lenghtS ] with .batch_sizes
        printing("SOURCE Packed data shape {} ", var=(packed_char_vecs.data.shape), verbose=self.verbose, verbose_level=4)
        # all sequence encoding [batch, max seq_len, n_dir x encoding dim] ,

        # last complete hidden state: [dir*n_layer, batch, dim encoding dim]
        output, h_n = self.seq_encoder(packed_char_vecs)
        # see if you really want that
        # NB WeightDropLSTM is also a nn.LSTM instance
        c_n = None
        if isinstance(self.seq_encoder, nn.LSTM):
            c_n = h_n[1]
            h_n = h_n[0]

        # TODO add attention out of the output (or maybe output the all output and define attention later)
        printing("SOURCE ENCODED all {}  , hidden {}  (output (includes all the "
                 "hidden states of last layers), last hidden hidden for each dir+layers)", var=(output.data.shape,
                                                                                                h_n.size()),
                 verbose=self.verbose, verbose_level=3)
        output, word_src_sizes = pad_packed_sequence(output, batch_first=True)
        # output : [batch, max word len, dim hidden_size_encoder]
        output = output[inverse_perm_idx, :, :]
        # TODO -> is the size correct here
        input = input[inverse_perm_idx, :] # needed because we use it for padding
        word_src_sizes = word_src_sizes[inverse_perm_idx]
        h_n = h_n[:, inverse_perm_idx, :]
        h_n = h_n.transpose(1, 0)
        h_n = h_n.contiguous().view(h_n.size(0), h_n.size(1)*h_n.size(2))

        attention_weights_char_tag = None
        if self.attention_query is not None:
            assert c_n is not None, "ERROR attention only supported when LSTM used (for layziness)"
            # we don't use c_n otherise so don't need to do this
            c_n = c_n[:, inverse_perm_idx, :]
            c_n = c_n.transpose(1, 0)
            c_n = c_n.contiguous().view(c_n.size(0), c_n.size(1) * c_n.size(2))
            # dim = 1 cause along sequence dimension
            proj = self.attention_query(output)
            # take care of padding for shorter sequence
            index_to_pad = (input[:, :output.size(1)] == 1)
            proj[index_to_pad, :] = -float("Inf")
            attention_weights_char_tag = nn.Softmax(dim=1)(proj)
            ## SHOULD REMOVE THE PAD ENCODINg
            #attention_weights_char_tag[attention_weights_char_tag!=attention_weights_char_tag] = 0
            _output = output.transpose(2, 1)
            new_h_n = torch.bmm(_output, attention_weights_char_tag)
            new_h_n = new_h_n.squeeze()
            #c_n = c_n.view(c_n.size(1), c_n.size(0)*c_n.size(2))
            h_n = torch.cat((c_n, new_h_n), dim=1)
            # TODO : could add projection to word_dim (or word_dim projected) so that we can sum and not concatanate to do Dozat
            #h_n = self.attention_projection(h_n)
        if self.char_level_encoding_projection is not None:
            h_n = self.char_level_encoding_projection(h_n)
        printing("SOURCE ENCODED UNPACKED {}  , hidden {}  (output (includes all the "
                 "hidden states of last layers), last hidden hidden for each dir+layers)", var=(output.data.shape, h_n.size()),
                 verbose=self.verbose, verbose_level=3)
        # TODO: check that using packed sequence provides the last state of the sequence(not the end of the padded one!)
        return h_n, output, word_src_sizes, attention_weights_char_tag

    def forward(self, input, input_word_len=None, word_embed_input=None, verbose=0):
        # input should be a batach of sentences
        # input : [batch, max sent len, max word len], input_word_len [batch, max_sent_len]
        context_level = self.context_level
        assert context_level in ["all", "word", "sent"], 'context_level : should be in ["all","word", "sent"]'
        printing("SOURCE : input size {}  length size {}",
                 var=(input.size(), input_word_len.size()),
                 verbose=verbose, verbose_level=4)
        _input_word_len = input_word_len.clone()
        # handle sentence that take the all sequence
        _input_word_len[:, -1, :] = 0
        # when input_word_len is 0 means we reached end of sentence
        # I think +1 is required : we want the lenght !! so if argmin --> 0 lenght should be 1 right
        #sent_len = torch.argmin(_input_word_len, dim=1) # PYTORCH 0.4
        sent_len = torch.Tensor(np.argmin(np.array(_input_word_len), axis=1)).long() ## PYTORCH 1.0 (or O.4)
        if _input_word_len.is_cuda:
            sent_len = sent_len.cuda()
        # we add to sent len if the original src word was filling the entire sequence (i.e last len is not 0)
        sent_len += (input_word_len[:, -1, :] != 0).long() # #handling (I guess) ODO : I think this case problem for sentence that take the all sequence : we are missing a word ! ??
        # sort batch based on sentence length
        sent_len, perm_idx_input_sent = sent_len.squeeze().sort(0, descending=True)
        # get inverse permutation to reorder
        inverse_perm_idx_input_sent = torch.from_numpy(np.argsort(perm_idx_input_sent.cpu().numpy()))
        # we pack and padd the sentence to shorten and pad sentences
        # [batch x sent_len , dim hidden word level] # this remove empty words
        # --PERMUTE input so that it's sorted
        try:
            ## WARNING / CHANGED sent_len.squeeze().cpu().numpy() into sent_len.cpu().numpy()
            packed_char_vecs_input = pack_padded_sequence(input[perm_idx_input_sent, :, :], sent_len.cpu().numpy(), batch_first=True)
        except:
            #print("EXCEPT ENCODER PACKING", [perm_idx_input_sent])
            if len(perm_idx_input_sent.size()) == 0:
                perm_idx_input_sent = [perm_idx_input_sent]
                inverse_perm_idx_input_sent = [inverse_perm_idx_input_sent]
                sent_len = sent_len.view(-1)
            packed_char_vecs_input = pack_padded_sequence(input[perm_idx_input_sent, :, :], sent_len.cpu().numpy(), batch_first=True)
        # unpacked for computing the word level representation
        input_char_vecs, input_sizes = pad_packed_sequence(packed_char_vecs_input, batch_first=True,
                                                           padding_value=PAD_ID_CHAR)
        # [batch, sent_len max, dim encoder] reorder the sequence
        # permutation test
        # assert (input[perm_idx_input_sent, :, :][inverse_perm_idx_input_sent,:,:] == input).all()
        #input_char_vecs = input_char_vecs[inverse_perm_idx_input_sent, :, :]
        # --PERMUTE : input_word_len : we align it with input_char that has been permuted
        input_word_len = input_word_len[perm_idx_input_sent,:,:]
        # cut input_word_len so that it fits packed_padded sequence
        input_word_len = input_word_len[:, :input_char_vecs.size(1), :]
        sent_len_max_source = input_char_vecs.size(1)
        # reshape word_len and word_char_vecs matrix --> for feeding to word level encoding
        input_word_len = input_word_len.contiguous().view(input_word_len.size(0) * input_word_len.size(1))
        #DEPRECIATED : shape_sent_seq = input_char_vecs.size()
        input_char_vecs = input_char_vecs.contiguous().view(input_char_vecs.size(0) * input_char_vecs.size(1), input_char_vecs.size(2))
        # input_char_vecs : [batch x max sent_len , MAX_CHAR_LENGTH or bucket max_char_length]
        # input_word_len  [batch x max sent_len]
        h_w, char_seq_hidden, word_src_sizes, attention_weights_char_tag = self.word_encoder_source(input=input_char_vecs,
                                                                                                    input_word_len=input_word_len)
        # [batch x max sent_len , packed max_char_length, hidden_size_encoder]
        # n_dir x dim hidden
        # [batch,  max sent_len , packed max_char_length, hidden_size_encoder]
        printing("SOURCE word encoding reshaped dim sent : {} ", var=[h_w.size()],
                 verbose=verbose, verbose_level=3)
        printing("SOURCE char_seq_hidden encoder reshaped dim sent : {} ", var=[char_seq_hidden.size()],
                 verbose=verbose, verbose_level=3)
        if self.add_word_level:
            assert word_embed_input is not None, "ERROR word_embed_input required as self.add_word_level"
            # we trust h_w for padding
            # TODO / IS this concatanation CORRECT ??
            #word_embed_input = word_embed_input[:, :h_w.size(1), :]
            if self.mode_word_encoding == "cat":
                h_w = torch.cat((word_embed_input, #.float(),
                                 h_w), dim=-1)
            elif self.mode_word_encoding == "sum":
                h_w = word_embed_input+h_w
        sent_len_cumulated = get_cumulated_list(sent_len)
        # we want to pack the sequence so we tranqform it as a list
        # NB ; sent_len and sent_len_cumulated are aligned with permuted input and therefore input_char_vec and h_w
        h_w_ls = [h_w[sent_len_cumulated[i]:sent_len_cumulated[i + 1], :] for i in range(len(sent_len_cumulated) - 1)]
        h_w = pack_sequence(h_w_ls)
        # sent_encoded last layer for each t (word) of the last layer
        if context_level != "word":
            sent_encoded, _ = self.sent_encoder(h_w)
            # add contitioning
            sent_encoded, length_sent = pad_packed_sequence(sent_encoded, batch_first=True)

        h_w, lengh_2 = pad_packed_sequence(h_w, batch_first=True)
        # now we reorder it one time to get the original order
        # --PERMUTE / reorder to original ordering so that it's consistent with output
        h_w = h_w[inverse_perm_idx_input_sent, :, :]
        if context_level != "word":
            sent_encoded = sent_encoded[inverse_perm_idx_input_sent, :, :]
            # sent_encoded : upper layer only but all time step, to get all the layers states of the last state get hidden
            # sent_encoded : [batch, max sent len ,hidden_size_sent_encoder]
            printing("SOURCE sentence encoder output dim sent : {} ", var=[sent_encoded.size()],
                     verbose=verbose, verbose_level=3)
        # concatanate
        if context_level != "word":
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
        return source_context_word_vector, sent_len_max_source, char_seq_hidden, word_src_sizes, attention_weights_char_tag

