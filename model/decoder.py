import torch.nn as nn
import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from io_.info_print import printing
import pdb
import time
from torch.autograd import Variable
from toolbox.sanity_check import get_timing
from collections import OrderedDict
from env.project_variables import SUPPORED_WORD_ENCODER
import torch.nn.functional as F
from io_.dat.constants import PAD_ID_WORD

EPSILON = 0.00000001

class Attention(nn.Module):

    def __init__(self,  hidden_size, method="general",use_gpu=False):

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2,hidden_size)#+hidden_size, hidden_size) # CHANGE--> (compared to example) we (hidden_size * 2+hidden_size because we have the embedding size +  ..
        self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        self.use_gpu = use_gpu
        self.method = method

    def score(self, char_state_decoder, encoder_output):
        if self.method == "concat":
            print("WARNING : Do not understand the self.v.dot + will cause shape error  ")
            energy = self.attn(torch.cat((char_state_decoder, encoder_output), 0))#CHANGE 0 instead of 1
            energy = self.v.dot(energy)
        elif self.method == "general":
            energy = self.attn(char_state_decoder)
            energy = encoder_output.dot(energy)
        return energy

    def forward(self, char_state_decoder, encoder_outputs):
        max_word_len_src = encoder_outputs.size(1)
        this_batch_size = encoder_outputs.size(0)
        attn_energies = Variable(torch.zeros(this_batch_size, max_word_len_src)) # B x S

        # we loop over all the source encoded sequence (of character) to compute the attention weight
        # is the loop on the batch necessary
        for batch in range(this_batch_size):
            for char_src in range(max_word_len_src):
                # encoder_outputs[batch, char_src] : contextual character embedding of character ind char_src at batch (word level context) of the source word
                # char_state_decoder[batch, :] : state of the decoder for batch ind (embedding)
                attn_energies[batch, char_src] = self.score(char_state_decoder[batch, :], encoder_outputs[batch, char_src]) # CHANGE : no need of unsquueze ?
        softmax = F.softmax(attn_energies)
        pdb.set_trace()
        assert ((softmax.sum(dim=1) - torch.ones(F.softmax(attn_energies).size(0))) < EPSILON).all(), "ERROR : softmax not softmax"

        return softmax.unsqueeze(1)


class CharDecoder(nn.Module):

    def __init__(self, char_embedding, input_dim, hidden_size_decoder, word_recurrent_cell=None,
                 drop_out_word_cell=0,timing=False, drop_out_char_embedding_decoder=0,
                 char_src_attention=False, unrolling_word=False, teacher_force=True,
                 verbose=0):
        super(CharDecoder, self).__init__()
        self.timing = timing
        self.char_embedding_decoder = char_embedding
        self.unrolling_word = unrolling_word
        printing("WARNING : DECODER unrolling_word is {}", var=[unrolling_word], verbose_level=0, verbose=verbose)
        printing("WARNING : DECODER char_src_attention is {}", var=[char_src_attention], verbose_level=0, verbose=verbose)
        self.teacher_force = teacher_force
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
        #self.attn_param = nn.Linear(hidden_size_decoder*1) if char_src_attention else None
        self.attn_layer = Attention(hidden_size_decoder) if char_src_attention else None
        self.verbose = verbose

    def word_encoder_target_step(self, char_vec_current_batch,  state_decoder_current,
                                 conditioning_other, char_seq_hidden_encoder=None):
        # char_vec_current_batch is the new input character read, state_decoder_current
        # is the state of the cell (h and possibly cell)
        # should torch.cat() char_vec_current_batch with attention based context computed on char_seq_hidden_encoder
        state_hiden, state_cell = state_decoder_current[0], state_decoder_current[1] if isinstance(self.seq_decoder, nn.LSTM) else (state_decoder_current, None)
        printing("DECODER STEP : target char_vec_current_batch {} size and state_decoder_current {} and {} size",
                 var=[char_vec_current_batch.size(), state_hiden.size(), state_cell.size()],
                 verbose_level=3, verbose=self.verbose)

        printing("DECODER STEP : context  char_vec_current_batch {}, state_decoder_current {} ",
                 var=[char_vec_current_batch.size(), state_hiden.size()], verbose_level=0,
                 verbose=self.verbose)
        # attention weights computed based on character embedding + state of the decoder recurrent state
        current_state = torch.cat((char_vec_current_batch, state_hiden.squeeze(0)), dim=1)
        char_vec_current_batch = char_vec_current_batch.unsqueeze(1)
        # current_state : for each word (in sentence batches)  1 character local target context
        # (char embedding + previous recurrent state of the decoder))
        # current_state  : dim batch x sentence max len , char embedding + hidden_dim decoder
        if self.attn_layer is not None:
            attention_weights = self.attn_layer(char_state_decoder=current_state, encoder_outputs=char_seq_hidden_encoder)
            printing("DECODER STEP : attention context {} char_seq_hidden_encoder {} ",var=[attention_weights.size(), char_seq_hidden_encoder.size()],
                     verbose_level=0, verbose=self.verbose)
            # compute product attention_weights with  char_seq_hidden_encoder (updated for each character)
            # this provide a new character context that we concatanate with char_vec_current_batch + possibly conditioning_other as they do https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
            # the context is goes as input as the character embedding : we add the tranditional conditioning_other
        output, state = self.seq_decoder(char_vec_current_batch, state_decoder_current)
        # output and state are equal because we call the GRU step by step (no the entire sequence)
        return output, state

    def word_encoder_target(self, output, conditioning, output_word_len, char_seq_hidden_encoder=None):
        # TODO DEAL WITH MASKING (padding and prediction oriented ?)
        printing("TARGET size {} ", var=output.size(), verbose=self.verbose, verbose_level=3)
        printing("TARGET data {} ", var=output, verbose=self.verbose, verbose_level=5)
        printing("TARGET  : Word  length  {}  ".format(output_word_len), self.verbose, verbose_level=5)
        output_word_len, perm_idx_output = output_word_len.squeeze().sort(0, descending=True)
        output = output[perm_idx_output, :]
        inverse_perm_idx_output = torch.from_numpy(np.argsort(perm_idx_output.cpu().numpy()))
        # output : [  ]
        start = time.time() if self.timing else None
        char_vecs = self.char_embedding_decoder(output)
        char_vecs = self.drop_out_char_embedding_decoder(char_vecs)
        char_embedding, start = get_timing(start)
        printing("TARGET EMBEDDING size {} ", var=[char_vecs.size()], verbose=self.verbose, verbose_level=3) #if False else None
        printing("TARGET EMBEDDING data {} ", var=char_vecs, verbose=self.verbose, verbose_level=5)
        not_printing, start = get_timing(start)
        conditioning = conditioning[:, perm_idx_output, :]
        reorder_conditioning, start = get_timing(start)
        #  USING PACKED SEQUENCE
        # THe shapes are fine !! -->
        printing("TARGET  word lengths after  {} dim", var =[output_word_len.size()], verbose=self.verbose, verbose_level=3)
        # same as target sequence and source ..
        output_word_len[output_word_len == 0] = 1
        # pdb.set_trace()
        zero_last, start = get_timing(start)
        packed_char_vecs_output = pack_padded_sequence(char_vecs, output_word_len.squeeze().cpu().numpy(), batch_first=True)
        pack, start = get_timing(start)
        printing("TARGET packed_char_vecs {}  dim", var=[packed_char_vecs_output.data.shape], verbose=self.verbose,
                 verbose_level=3) # .size(), packed_char_vecs)
        # conditioning is the output of the encoder (work as the first initial state of the decoder)
        if isinstance(self.seq_decoder, nn.LSTM):
            conditioning = (torch.zeros_like(conditioning), conditioning)

        # attention
        import torch.nn.functional as F
        if self.attn_layer is not None:
            assert char_seq_hidden_encoder is not None, 'ERROR sent_len_max_source is None'
        # start new unrolling by habd
        # we initiate with our same original context conditioning
        if self.unrolling_word:
            state_i = conditioning
            _output = []
            # we repad it straight away cause unrolling by hand
            char_vecs, char_vecs_sizes = pad_packed_sequence(packed_char_vecs_output, batch_first=True)
            printing("DECODER char_vecs re-paded {} ", var=[char_vecs.data.size()], verbose=self.verbose,
                     verbose_level=3)
            max_word_len = char_vecs.size(1)
            for char_i in range(max_word_len):
                if self.teacher_force:
                    emb_char = char_vecs[:, char_i, :]
                else:
                    print("not supported yet")
                    raise(Exception)
                    pass
                # no more pack sequence
                printing("DECODER state_decoder_current {} ", var=[state_i[0].size()], verbose=self.verbose, verbose_level=3)
                printing("DECODER emb_char {} ", var=[emb_char.size()], verbose=self.verbose, verbose_level=2)
                all_states, state_i = self.word_encoder_target_step(char_vec_current_batch=emb_char,
                                                        state_decoder_current=state_i,
                                                        char_seq_hidden_encoder=char_seq_hidden_encoder, #char_seq_hidden_encoder,
                                                        conditioning_other=None) #conditioning)
                #c_i = state_i[1] if isinstance(self.seq_decoder, nn.LSTM) else None
                h_i = state_i[0] if isinstance(self.seq_decoder, nn.LSTM) else h_i
                _output.append(h_i.transpose(0, 1)) # for LSTM the hidden is the output not the cell
                printing("DECODER hidden out {} ", var=[h_i.transpose(0, 1).size()], verbose=self.verbose, verbose_level=0)
                printing("DECODER all_states {} ", var=[all_states.size()], verbose=self.verbose, verbose_level=0)
                #assert (all_states == h_i.transpose(0, 1)).all() == 1
            output = torch.cat(_output, dim=1)
            # we reoder char_vecs so need to do it
            output = output[inverse_perm_idx_output, :, :]
            printing("DECODER : target unrolling : output {} size ", var=[output.size()], verbose=0, verbose_level=3)

                # TODO : shape it in the right way and output : it should include all for each sentence batch ,for each word, for each character a hidden representation
        else:
            # state_char_decoder should be cat to char_vecs
            # attention_weights = F.softmax(self.attn_param(torch.cat((char_vecs,state_char_decoder))))
            # then product with char_seq_hidden_encoder
            # then cat in a way with conditioning (and should remove word level part of it also )
            # then feed as conditioning
            # back to old implementation
            output, h_n = self.seq_decoder(packed_char_vecs_output, conditioning)
            h_n = h_n[0] if isinstance(self.seq_decoder, nn.LSTM) else h_n
            recurrent_cell, start = get_timing(start)
            printing("TARGET ENCODED {} output {} h_n (output (includes all the hidden states of last layers), "
                     "last hidden hidden for each dir+layers)", var=(output, h_n), verbose=self.verbose, verbose_level=5)
            printing("TARGET ENCODED SIZE {} output {} h_n (output (includes all the hidden states of last layers), "
                     "last hidden hidden for each dir+layers)", var=(output.data.shape, h_n.size()), verbose=self.verbose,
                     verbose_level=3)
            output, output_sizes = pad_packed_sequence(output, batch_first=True)
            padd, start = get_timing(start)
            output = output[inverse_perm_idx_output, :, :]
            printing("TARGET ENCODED UNPACKED  {} output {} h_n (output (includes all the hidden states of last layers), "
                     "last hidden hidden for each dir+layers)", var=(output, h_n), verbose=self.verbose, verbose_level=5)

            printing("TARGET ENCODED UNPACKED SIZE {} output {} h_n (output (includes all "
                     "the hidden states of last layers),"
                     "last hidden hidden for each dir+layers)", var=(output.size(), h_n.size()),
                     verbose=self.verbose, verbose_level=3)
            # TODO : output is not shorted in regard to max sent len --> how to handle gold sequence ?
            if self.timing:
                 print("WORD TARGET {} ".format(OrderedDict([('char_embedding', char_embedding),
                      ("reorder_conditioning", reorder_conditioning), ("zero_last", zero_last),("not_printing",not_printing),
                      ("pack",pack), ("recurrent_cell", recurrent_cell), ("padd", padd)])))
        pdb.set_trace()
        return output

    def sent_encoder_target(self, output, conditioning, output_word_len,
                            char_seq_hidden_encoder=None,
                            sent_len_max_source=None,verbose=0):

        # WARNING conditioning is for now the same for every decoded token
        #printing("TARGET output_mask size {}  mask  {} size length size {} ", var=(output_mask.size(), output_mask.size(),
        #                                                                           output_mask.size()), verbose=verbose,
        #         verbose_level=3)

        conditioning = conditioning.view(1, conditioning.size(0) * conditioning.size(1), -1)
        start = time.time() if self.timing else None
        _output_word_len = output_word_len.clone()
        clone_len, start = get_timing(start)
        # handle sentence that take the all sequence (
        printing("TARGET SIZE : output_word_len length (before 0 last) : size {} data {} ", var=(_output_word_len.size(),_output_word_len), verbose=verbose,
                 verbose_level=4)
        printing("TARGET : output  (before 0 last) : size {}", var=[output.size()], verbose=verbose, verbose_level=3)
        printing("TARGET : output  (before 0 last) :  data {} ", var=[output], verbose=verbose, verbose_level=5)

        _output_word_len[:, -1, :] = 0
        # when input_word_len is 0 means we reached end of sentence
        # TODO : WARNING : is +1 required : as sent with 1 ? WHY ALWAYS IS NOT WORKING
        sent_len = torch.argmin(_output_word_len, dim=1)
        # WARNING : forcint sent_len to be one
        if (sent_len == 0).any():
            printing("WARNING : WE ARE FORCING SENT_LEN in the SOURCE SIDE", verbose=verbose, verbose_level=0)
            sent_len[sent_len == 0] += 1
        # sort batch at the sentence length
        sent_len, perm_idx_input_sent = sent_len.squeeze().sort(0, descending=True)
        argmin_squeeze, start = get_timing(start)
        inverse_perm_idx_input_sent = torch.from_numpy(np.argsort(perm_idx_input_sent.cpu().numpy()))
        sorting, start = get_timing(start)
        # [batch x sent_len , dim hidden word level] # this remove empty words
        packed_char_vecs_output = pack_padded_sequence(output[perm_idx_input_sent, :, :],
                                                       sent_len.squeeze().cpu().numpy(), batch_first=True)
        packed_sent, start = get_timing(start)
        # unpacked for the word level representation
        # packed_char_vecs_output .data : [batch x shorted sent_lenS , word lens ] + .batch_sizes
        output_char_vecs, output_sizes = pad_packed_sequence(packed_char_vecs_output, batch_first=True,
                                                             padding_value=PAD_ID_WORD) # padding_value
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
        printing("TARGET output before word encoder {}", var=[output_seq.size()], verbose=verbose, verbose_level=3)
        output_w_decoder = self.word_encoder_target(output_seq, conditioning, output_word_len, char_seq_hidden_encoder=char_seq_hidden_encoder)
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
