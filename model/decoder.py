from env.importing import *

from io_.info_print import printing
from toolbox.deep_learning_toolbox import get_cumulated_list
from toolbox.sanity_check import get_timing
from env.project_variables import SUPPORED_WORD_ENCODER
from io_.dat.constants import PAD_ID_WORD
from model.attention import Attention

EPSILON = 1e-6


class CharDecoder(nn.Module):

    def __init__(self, char_embedding, input_dim, hidden_size_decoder, shared_context, word_recurrent_cell=None,
                 drop_out_word_cell=0, timing=False, drop_out_char_embedding_decoder=0,
                 char_src_attention=False, unrolling_word=False, init_context_decoder=True,
                 hidden_size_src_word_encoder=None, generator=None, stable_decoding_state=False,
                 verbose=0):
        super(CharDecoder, self).__init__()
        self.generator = generator
        self.timing = timing
        self.char_embedding_decoder = char_embedding
        self.shared_context = shared_context
        self.unrolling_word = unrolling_word
        self.stable_decoding_state = stable_decoding_state
        self.init_context_decoder = init_context_decoder

        printing("WARNING : stable_decoding_state is {}", var=[stable_decoding_state], verbose_level=0, verbose=verbose)
        printing("WARNING : init_context_decoder is {}", var=[init_context_decoder], verbose_level=0, verbose=verbose)
        printing("WARNING : DECODER unrolling_word is {}", var=[unrolling_word], verbose_level=0, verbose=verbose)
        printing("WARNING : DECODER char_src_attention is {}", var=[char_src_attention], verbose_level=0, verbose=verbose)
        self.drop_out_char_embedding_decoder = nn.Dropout(drop_out_char_embedding_decoder)
        if word_recurrent_cell is not None:
            assert word_recurrent_cell in SUPPORED_WORD_ENCODER, \
                "ERROR : word_recurrent_cell should be in {} ".format(SUPPORED_WORD_ENCODER)
        if word_recurrent_cell is None:
            word_recurrent_cell = nn.GRU
        else:
            word_recurrent_cell = eval("nn."+word_recurrent_cell)

        if isinstance(word_recurrent_cell, nn.LSTM):
            printing("WARNING : in the case of LSTM : inital states defined as "
                     " h_0, c_0 = (zero tensor, source_conditioning) so far (cf. row 70 decoder.py) ",
                     verbose=self.verbose, verbose_level=0)
            printing("WARNING : LSTM only using h_0 for prediction not the  cell", verbose=self.verbose, verbose_level=0)
        if char_src_attention:
            assert hidden_size_src_word_encoder is not None, "ERROR : need hidden_size_src_word_encoder for attention "
            # we need to add dimension because of the context vector that is hidden_size encoder projected
            #input_dim += hidden_size_src_word_encoder : # NO NEED anymire as we project the all context : same size as currnt
            printing("WARNING : DECODER word_recurrent_cell hidden dim will be {} "
                     "(we added hidden_size_decoder) because of attention", verbose=verbose, verbose_level=0)
        # if stable_decoding_state : we add a projection of the attention context vector + the stable one
        # TODO : try to project the cat of those three vectors (char, attention context, stable context)
        self.context_proj = nn.Linear(hidden_size_decoder*int(stable_decoding_state)+hidden_size_src_word_encoder*int(char_src_attention), char_embedding.embedding_dim) if stable_decoding_state or char_src_attention else None
        input_dim = 2*input_dim if stable_decoding_state or char_src_attention else input_dim # because we concat with projected context
        self.seq_decoder = word_recurrent_cell(input_size=input_dim, hidden_size=hidden_size_decoder,
                                               num_layers=1,
                                               dropout=drop_out_word_cell,
                                               bias=True, batch_first=True, bidirectional=False)
        printing("MODEL Decoder : word_recurrent_cell has been set to {} ".format(str(word_recurrent_cell)),
                 verbose=verbose, verbose_level=1)
        #self.attn_param = nn.Linear(hidden_size_decoder*1) if char_src_attention else None
        self.dropout_char_in = nn.Dropout(0.3)
        self.attn_layer = Attention(hidden_size_word_decoder=hidden_size_decoder,
                                    char_embedding_dim=self.char_embedding_decoder.embedding_dim,
                                    time=self.timing,
                                    hidden_size_src_word_encoder=hidden_size_src_word_encoder) if char_src_attention else None
        self.verbose = verbose

    def word_encoder_target_step(self, char_vec_current_batch,  state_decoder_current,
                                 char_vecs_sizes, step_char, word_stable_context,
                                 char_seq_hidden_encoder=None):
        # char_vec_current_batch is the new input character read, state_decoder_current
        # is the state of the cell (h and possibly cell)
        # should torch.cat() char_vec_current_batch with attention based context computed on char_seq_hidden_encoder

        state_hiden, state_cell = state_decoder_current[0], state_decoder_current[1] if isinstance(self.seq_decoder, nn.LSTM) else (state_decoder_current, None)
        printing("DECODER STEP : target char_vec_current_batch {} size and state_decoder_current {} and {} size",
                 var=[char_vec_current_batch.size(), state_hiden.size(), state_cell.size()],
                 verbose_level=3, verbose=self.verbose)

        printing("DECODER STEP : context  char_vec_current_batch {}, state_decoder_current {} ",
                 var=[char_vec_current_batch.size(), state_hiden.size()], verbose_level=3,
                 verbose=self.verbose)
        # attention weights computed based on character embedding + state of the decoder recurrent state
        #current_state = torch.cat((char_vec_current_batch, state_hiden.squeeze(0)), dim=1)
        char_vec_current_batch = char_vec_current_batch.unsqueeze(1)
        # current_state : for each word (in sentence batches)  1 character local target context
        # (char embedding + previous recurrent state of the decoder))
        # current_state  : dim batch x sentence max len , char embedding + hidden_dim decoder
        start_atten = time.time()

        if self.attn_layer is not None:
            #TODO : make it generic (is there no problem also if not attention ?? (bug fix)
            # we align what we decode
            state_hiden = state_hiden[:, :char_seq_hidden_encoder.size(0), :]
            attention_weights = self.attn_layer(char_state_decoder=state_hiden.squeeze(0),word_src_sizes=char_vecs_sizes,encoder_outputs=char_seq_hidden_encoder)
            printing("DECODER STEP : attention context {} char_seq_hidden_encoder {} ", var=[attention_weights.size(), char_seq_hidden_encoder.size()],
                     verbose_level=3, verbose=self.verbose)

            # we multiply for each batch attention matrix by our source sequence
            if char_seq_hidden_encoder.is_cuda:
                # don't know why we need to do that 
                attention_weights = attention_weights.cuda()
            # TODO HOW IS MASKING TAKEN CARE OF IN THE TARGET ? WE PACKED AND PADDED SO SHORTED THE SEQUENCE
            attention_context = attention_weights.bmm(char_seq_hidden_encoder)
            # was context
        else:
            attention_context = None
            attention_weights = None
        if self.stable_decoding_state:
            word_stable_context = word_stable_context.transpose(1, 0)
        else:
            word_stable_context = None
        if attention_context is not None or word_stable_context is not None:
            if attention_context is None:
                context = word_stable_context
            elif word_stable_context is None:
                context = attention_context
            else:
                context = torch.cat((word_stable_context, attention_context), dim=2)
            context = self.context_proj(context)
            # MIS ALGINEMNT BETWEEN SOURCE CHAR LEVEL CONTEXT PER WORD AND WORD THAT WE DECODE PER CHAR
            # output_word_len is incorrect --> char_vec_current_batch incorrect to at test time were
            try:
                char_vec_current_batch = torch.cat((context, char_vec_current_batch), dim=2)
            except:
                pdb.set_trace()
                char_vec_current_batch = torch.cat((context, char_vec_current_batch), dim=2)

        else:
            # no word level context passed so --> char_vec_current is only the current character vector  
            pass

            # compute product attention_weights with  char_seq_hidden_encoder (updated for each character)
            # this provide a new character context that we concatanate
            #  with char_vec_current_batch + possibly conditioning_other
            #  as they do
            ##https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
            # the context is goes as input as the character embedding : we add the tranditional conditioning_other
        time_atten, start = get_timing(start_atten)
        try:
            output, state = self.seq_decoder(char_vec_current_batch, state_decoder_current)
        except Exception as a:
            print(Exception(a))

            output, state = self.seq_decoder(char_vec_current_batch, state_decoder_current)
        time_step_decoder, _ = get_timing(start)
        if self.timing:
            print("Attention time {} ".format(OrderedDict([('time_attention', time_atten),
                                                           ("time_step_decoder", time_step_decoder)])))
        # output and state are equal because we call the GRU step by step (no the entire sequence)
        return output, state, attention_weights

    def word_encoder_target(self, output, conditioning, output_word_len,
                            word_src_sizes=None,
                            proportion_pred_train=None,
                            char_seq_hidden_encoder=None):
        # TODO DEAL WITH MASKING (padding and prediction oriented ?)

        printing("TARGET size {} ", var=output.size(), verbose=self.verbose, verbose_level=3)
        printing("TARGET data {} ", var=output, verbose=self.verbose, verbose_level=5)
        printing("TARGET  : Word  length  {}  ".format(output_word_len), self.verbose, verbose_level=5)
        start = time.time() if self.timing else None
        output_word_len, perm_idx_output = output_word_len.squeeze().sort(0, descending=True)
        output = output[perm_idx_output]
        # we made the choice to mask again the
        conditioning = conditioning.view(1, conditioning.size(0) * conditioning.size(1), -1)
        conditioning = conditioning[:, perm_idx_output, :]
        reorder_conditioning, start = get_timing(start)
        perm_idx_output = perm_idx_output[output_word_len != 0]
        inverse_perm_idx_output = torch.from_numpy(np.argsort(perm_idx_output.cpu().numpy()))
        # output : [  ]
        # we remove empty token from the output_sequence and th input conditioning vector () (as we did in the input) ,
        #pdb.set_trace()
        output = output[output_word_len != 0]
        conditioning = conditioning[:, output_word_len !=0, :]
        output_word_len = output_word_len[output_word_len != 0]
        char_vecs = self.char_embedding_decoder(output)
        char_vecs = self.drop_out_char_embedding_decoder(char_vecs)
        char_embedding, start = get_timing(start)
        printing("TARGET EMBEDDING size {} ", var=[char_vecs.size()], verbose=self.verbose, verbose_level=3) #if False else None
        printing("TARGET EMBEDDING data {} ", var=char_vecs, verbose=self.verbose, verbose_level=5)
        not_printing, start = get_timing(start)
        printing("TARGET  word lengths after  {} dim",
                 var=[output_word_len.size()], verbose=self.verbose, verbose_level=3)
        # same as target sequence and source ..
        #output_word_len[output_word_len == 0] = 1
        zero_last, start = get_timing(start)
        packed_char_vecs_output = pack_padded_sequence(char_vecs, output_word_len.squeeze().cpu().numpy(), batch_first=True)
        pack_time, start = get_timing(start)
        _start_recurrence = start
        printing("TARGET packed_char_vecs {}  dim", var=[packed_char_vecs_output.data.shape], verbose=self.verbose,
                 verbose_level=3) # .size(), packed_char_vecs)
        # conditioning is the output of the encoder (work as the first initial state of the decoder)
        if isinstance(self.seq_decoder, nn.LSTM):
            stable_decoding_word_state = conditioning.clone() if self.stable_decoding_state else None
            # TODO : ugly because we have done a projection and reshaping for nothing on conditioning
            conditioning = (torch.zeros_like(conditioning), conditioning) if self.init_context_decoder else (torch.zeros_like(conditioning), torch.zeros_like(conditioning))
        # attention
        if self.attn_layer is not None:
            assert char_seq_hidden_encoder is not None, 'ERROR sent_len_max_source is None'
            assert word_src_sizes is not None
        # start new unrolling by had
        # we initiate with our same original context conditioning
        if self.unrolling_word:
            # we start with context sent + word as before
            state_i = conditioning
            _output = []
            attention_weight_all = []
            # we repad it straight away cause unrolling by hand
            char_vecs, char_vecs_sizes_target = pad_packed_sequence(packed_char_vecs_output, batch_first=True)
            #pdb.set_trace()
            printing("DECODER char_vecs re-paded {} ", var=[char_vecs.data.size()], verbose=self.verbose,
                     verbose_level=3)
            max_word_len = char_vecs.size(1)
            for char_i in range(max_word_len):
                if proportion_pred_train is not None:
                    teacher_force = True if np.random.randint(0, 100) > proportion_pred_train else False
                else:
                    teacher_force = True
                if teacher_force or char_i == 0:
                    emb_char = char_vecs[:, char_i, :]
                    printing("DECODER state_decoder_current {} ", var=[state_i[0].size()], verbose=self.verbose,
                             verbose_level=3)
                    printing("DECODER emb_char {} ", var=[emb_char.size()], verbose=self.verbose, verbose_level=3)
                    all_states, state_i, attention_weights = self.word_encoder_target_step(
                                                                    char_vec_current_batch=emb_char,
                                                                    state_decoder_current=state_i,
                                                                    char_vecs_sizes=word_src_sizes,
                                                                    step_char=char_i,
                                                                    word_stable_context=stable_decoding_word_state,
                                                                    char_seq_hidden_encoder=char_seq_hidden_encoder)
                else:
                    assert self.generator is not None, "Generator must be passed in decoder for decodibg if not teacher_force"
                    # TODO based on state_i compute as generator : get id : lookup character embedding and that's it
                    # TODO : not fir the first one that should be the STARTING_SYMBOL
                    # given the current emb_char, the states of the cell (inirialized with the conditoning source )
                    #  we compute the next states
                    # [batch x sent_max_len, len_words] ??
                    decoding_states, state_i, attention_weights = self.word_encoder_target_step(char_vec_current_batch=emb_char,
                                                                                                word_stable_context=stable_decoding_word_state,
                                                                                                state_decoder_current=state_i,
                                                                                                char_vecs_sizes=word_src_sizes,
                                                                                                step_char=char_i,
                                                                                                char_seq_hidden_encoder=char_seq_hidden_encoder)
                    printing("DECODING in schedule sampling {} ", var=[state_i[0].size()], verbose=self.verbose,
                             verbose_level=3)
                    # we feed to generator to get the score and the prediction
                    # [batch x sent_max_len, len_words, hidden_dim] ??
                    scores = self.generator.forward(x=decoding_states)

                    predictions = scores.argmax(dim=-1)

                    # TODO : to confirm the shapes here
                    pred = predictions[:,  -1]

                    # given the prediction we get the next character embedding
                    emb_char = self.char_embedding_decoder(pred)

                # no more pack sequence&
                # TODO : should shorted sequence output and state by setting them to 0 using step_char and char_vecs_sizes_target (but it should be fine with the loss outpu)
                #c_i = state_i[1] if isinstance(self.seq_decoder, nn.LSTM) else None
                h_i = state_i[0] if isinstance(self.seq_decoder, nn.LSTM) else h_i
                attention_weight_all.append(attention_weights)
                _output.append(h_i.transpose(0, 1)) # for LSTM the hidden is the output not the cell
                printing("DECODER hidden out {} ", var=[h_i.transpose(0, 1).size()], verbose=self.verbose, verbose_level=3)
                printing("DECODER all_states {} ", var=[all_states.size()], verbose=self.verbose, verbose_level=3)
                #assert (all_states == h_i.transpose(0, 1)).all() == 1
            output = torch.cat(_output, dim=1)
            # we reoder char_vecs so need to do it
            output = output[inverse_perm_idx_output, :, :]
            printing("DECODER : target unrolling : output {} size ", var=[output.size()], verbose=0, verbose_level=3)
            recurrent_cell_time, pack_time, padd_time = None, None, None
        else:

            output, h_n = self.seq_decoder(packed_char_vecs_output, conditioning)
            h_n = h_n[0] if isinstance(self.seq_decoder, nn.LSTM) else h_n
            recurrent_cell_time, start = get_timing(start)
            printing("TARGET ENCODED {} output {} h_n (output (includes all the hidden states of last layers), "
                     "last hidden hidden for each dir+layers)", var=(output, h_n), verbose=self.verbose, verbose_level=5)
            printing("TARGET ENCODED SIZE {} output {} h_n (output (includes all the hidden states of last layers), "
                     "last hidden hidden for each dir+layers)", var=(output.data.shape, h_n.size()), verbose=self.verbose,
                     verbose_level=3)
            output, output_sizes = pad_packed_sequence(output, batch_first=True)
            padd_time, start = get_timing(start)
            output = output[inverse_perm_idx_output, :, :]
            printing("TARGET ENCODED UNPACKED  {} output {} h_n (output (includes all the hidden states of last layers)"
                     "last hidden hidden for each dir+layers)", var=(output, h_n), verbose=self.verbose, verbose_level=5)

            printing("TARGET ENCODED UNPACKED SIZE {} output {} h_n (output (includes all "
                     "the hidden states of last layers),"
                     "last hidden hidden for each dir+layers)", var=(output.size(), h_n.size()),
                     verbose=self.verbose, verbose_level=3)
            attention_weight_all = None
        all_recurrent_time, _ = get_timing(_start_recurrence)
        if self.timing:
            print("WORD TARGET {} ".format(OrderedDict([('char_embedding', char_embedding),
                                                        ("reorder_all", reorder_conditioning),
                                                        ("zero_last", zero_last), ("not_printing", not_printing),
                                                        ("pack_time", pack_time), ("recurrent_cell_time", recurrent_cell_time),
                                                        ("all_recurrent_time", all_recurrent_time), ("pad_time", padd_time)])))
        return output, attention_weight_all

    def forward(self, output, conditioning, output_word_len,
                char_seq_hidden_encoder=None,
                word_src_sizes=None, proportion_pred_train=None,
                sent_len_max_source=None, verbose=0):

        start = time.time() if self.timing else None
        _output_word_len = output_word_len.clone()
        clone_len, start = get_timing(start)
        # handle sentence that take the all sequence ()
        printing("TARGET SIZE : output_word_len length (before 0 last) : size {} data {} ", var=(_output_word_len.size(),_output_word_len), verbose=verbose,
                 verbose_level=4)
        printing("TARGET : output  (before 0 last) : size {}", var=[output.size()], verbose=verbose, verbose_level=3)
        printing("TARGET : output  (before 0 last) :  data {} ", var=[output], verbose=verbose, verbose_level=5)
        _output_word_len[:, -1, :] = 0
        # when input_word_len is 0 means we reached end of sentence
        # TODO : WARNING : is +1 required : as sent with 1 ? WHY ALWAYS IS NOT WORKING
        sent_len = torch.Tensor(np.argmin(np.array(_output_word_len), axis=1)).long()  ## PYTORCH 1.0 (or O.4)
        if _output_word_len.is_cuda:
            sent_len = sent_len.cuda()
        #sent_len = torch.argmin(_output_word_len, dim=1) ## PYTORCH WARNING : THEY MIGH BE A PROBLEM HERE
        # WARNING : forcint sent_len to be one
        if (sent_len == 0).any() and False:
            printing("WARNING : WE ARE FORCING SENT_LEN in the SOURCE SIDE", verbose=verbose, verbose_level=3)
            sent_len[sent_len == 0] += 1
        # as encoder side : we handle words that take the all sequnence
        sent_len += (output_word_len[:, -1, :] != 0).long()
        # sort batch at the sentence length
        sent_len, perm_idx_input_sent = sent_len.squeeze().sort(0, descending=True)
        argmin_squeeze, start = get_timing(start)
        inverse_perm_idx_input_sent = torch.from_numpy(np.argsort(perm_idx_input_sent.cpu().numpy()))
        sorting, start = get_timing(start)
        # [batch x sent_len , dim hidden word level] # this remove empty words
        #reorder so that it aligns with input

        try:
            ## WARNING / CHANGED sent_len.squeeze().cpu().numpy() into sent_len.cpu().numpy()
            packed_char_vecs_output = pack_padded_sequence(output[perm_idx_input_sent, :, :],
                                                           sent_len.cpu().numpy(), batch_first=True)
        except:
            print("EXCEPT DECODER PACKING", [perm_idx_input_sent])
            if len(perm_idx_input_sent.size()) == 0:
                perm_idx_input_sent = [perm_idx_input_sent]
                inverse_perm_idx_input_sent = [inverse_perm_idx_input_sent]
                sent_len = sent_len.view(-1)

            packed_char_vecs_output = pack_padded_sequence(output[perm_idx_input_sent, :, :],
                                                           sent_len.cpu().numpy(), batch_first=True)
        # unpacked for computing the word level representation
        #packed_char_vecs_output = pack_padded_sequence(output[perm_idx_input_sent, :, :],
        #                                               sent_len.squeeze().cpu().numpy(), batch_first=True)
        conditioning = conditioning[perm_idx_input_sent,:,:]
        packed_sent, start = get_timing(start)
        # unpacked for the word level representation
        # packed_char_vecs_output .data : [batch x shorted sent_lenS , word lens ] + .batch_sizes

        output_char_vecs, output_sizes = pad_packed_sequence(packed_char_vecs_output, batch_first=True,
                                                             padding_value=PAD_ID_WORD) # padding_value
        padd_sent, start = get_timing(start)

        # output_char_vecs : [batch ,  shorted sent_len, word len ] + .batch_sizes
        # output_char_vecs : [batch, sent_len max, dim encoder] reorder the sequence
        #output_char_vecs = output_char_vecs[inverse_perm_idx_input_sent, :, :]
        # reorder sent_len also
        #sent_len = sent_len[inverse_perm_idx_input_sent]
        # cut input_word_len so that it fits packed_padded sequence (based on output sequence)
        output_word_len = output_word_len[:, :output_char_vecs.size(1), :]
        # cut again (should be done in one step I guess) to fit source sequence (important at test time)
        output_word_len = output_word_len[:, :sent_len_max_source, :]
        # we cut output_char_vec based on ??
        output_char_vecs = output_char_vecs[:, :sent_len_max_source, :]
        output_seq = output_char_vecs.contiguous().view(output_char_vecs.size(0) * output_char_vecs.size(1), output_char_vecs.size(2))
        reshape_sent, start = get_timing(start)
        # output_seq : [ batch x max sent len, max word len  ]
        output_word_len = output_word_len.contiguous()
        output_word_len = output_word_len.view(output_word_len.size(0) * output_word_len.size(1))
        reshape_len, start = get_timing(start)
        printing("TARGET output before word encoder {}", var=[output_seq.size()], verbose=verbose, verbose_level=3)
        output_w_decoder, attention_weight_all = self.word_encoder_target(output_seq, conditioning, output_word_len, word_src_sizes=word_src_sizes,proportion_pred_train=proportion_pred_train,char_seq_hidden_encoder=char_seq_hidden_encoder)

        # output_w_decoder
        word_encoders, start = get_timing(start)
        # we update sent len based on how it was cut (specifically useful at test time)
        sent_len = torch.min(torch.ones_like(sent_len) * sent_len_max_source, sent_len)
        sent_len_cumulated = get_cumulated_list(sent_len)
        output_w_decoder_ls = [output_w_decoder[sent_len_cumulated[i]:sent_len_cumulated[i + 1]] for i in range(len(sent_len_cumulated) - 1)]
        output_w_decoder = pack_sequence(output_w_decoder_ls)
        output_w_decoder, _ = pad_packed_sequence(output_w_decoder, batch_first=True)
        output_w_decoder = output_w_decoder[inverse_perm_idx_input_sent,:,:]
        # output_w_decoder : [ n_sents  x max sent len, max word len , hidden_size_decoder ]
        #max_word = output_w_decoder.size(0)/output_char_vecs.size(0)
        #output_w_decoder = output_w_decoder.view(output_char_vecs.size(0), max_word, -1, output_w_decoder.size(2))
        if self.attn_layer is not None:
            attention_weight_all = torch.cat(attention_weight_all, dim=1)
            attention_weight_all_ls = [attention_weight_all[sent_len_cumulated[i]:sent_len_cumulated[i + 1]] for i in range(len(sent_len_cumulated) - 1)]
            attention_weight_all = pack_sequence(attention_weight_all_ls)
            attention_weight_all, _ = pad_packed_sequence(attention_weight_all, batch_first=True)
            attention_weight_all = attention_weight_all[inverse_perm_idx_input_sent]
            #attention_weight_all = torch.cat(attention_weight_all, dim=1)
            #attention_weight_all = attention_weight_all.view(output_char_vecs.size(0), max_word, attention_weight_all.size(1),attention_weight_all.size(2))
        else:
            attention_weight_all = None
        reshape_attention, start = get_timing(start)
        # output_w_decoder : [ batch , max sent len, max word len , hidden_size_decoder ]
        if self.timing:
            print("SENT TARGET : {}".format(OrderedDict([("clone_len", clone_len), ("argmin_squeeze", argmin_squeeze),("sorting", sorting),
                                                         ("packed_sent", packed_sent), ("padd_sent",padd_sent), ("reshape_sent",reshape_sent),
                                                         ("reshape_len",reshape_len),("word_encoders", word_encoders), ("reshape_attention",reshape_attention)])))
        return output_w_decoder, attention_weight_all


class WordDecoder(nn.Module):
    def __init__(self,  input_dim,
                 voc_size, dense_dim, dense_dim_2, dense_dim_3=0,
                 activation=None,
                 verbose=2):
        super(WordDecoder, self).__init__()
        assert dense_dim is not None and dense_dim > 0, "ERROR dense_dim should be 0"
        n_layers = 1
        if dense_dim_2 is not None and dense_dim_2 > 0:
            n_layers += 1
            if dense_dim_3 > 0 and dense_dim_3 is not None:
                n_layers+=1
            else:
                dense_dim_3 = dense_dim_2
        else:
            dense_dim_3 = dense_dim

        self.activation_decoder = str("nn.ReLU") if activation is None or activation == str(None) else activation

        self.dense_output_1 = nn.Linear(input_dim, dense_dim)
        self.dense_output_2 = nn.Linear(dense_dim, dense_dim_2) if n_layers > 1 else None
        self.dense_output_3 = nn.Linear(dense_dim_2, dense_dim_3) if n_layers>2 else None
        self.dense_output_4 = nn.Linear(dense_dim_3, voc_size)
        printing("MODEL WordDecoder set with {} dense layers + softmax ", var=n_layers, verbose=verbose, verbose_level=1)

    def forward(self, context):
        activation = eval(self.activation_decoder)
        if self.dense_output_3 is not None:
            prediction_state = activation()(self.dense_output_4(activation()(self.dense_output_3(activation()(self.dense_output_2(activation()(self.dense_output_1(context))))))))
        elif self.dense_output_2 is not None:
            prediction_state = activation()(self.dense_output_4(activation()(self.dense_output_2(activation()(self.dense_output_1(context))))))
        else:
            prediction_state = activation()(self.dense_output_4(activation()(self.dense_output_1(context))))

        return prediction_state
        # TODO :
        #1 process output gold sequence at the word level pack and
        #2 : reshape conditioning so that it fits the cell