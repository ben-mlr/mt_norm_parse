

import torch
import torch.nn as nn
from torch.autograd import Variable
from io_.info_print import printing
from io_.dat.normalized_writer import write_conll
from io_.from_array_to_text import output_text, output_text_
from io_.dat.constants import PAD_ID_CHAR, CHAR_END_ID
import numpy as np
from evaluate.normalization_errors import score_norm_not_norm
from evaluate.normalization_errors import score_ls_, correct_pred_counter
from env.project_variables import WRITING_DIR
from io_.dat.constants import CHAR_START_ID
from toolbox.sanity_check import get_timing
import pdb
import os
from collections import OrderedDict
from toolbox.beam_related_reshape_ind import get_beam_ind_token_ind, update_output_seq


from evaluate.visualize_attention import show_attention
from toolbox.norm_not_norm import get_label_norm
from io_.dat.constants import PAD, ROOT, END, ROOT_CHAR, END_CHAR
# EPSILON for the test of edit distance
import time
EPSILON = 0.000001
TEST_SCORING_IN_CODE = False


def decode_sequence_beam(model, max_len, src_seq, src_mask, src_len, char_dictionary,
                         pad=1, target_seq_gold=None,
                         use_gpu=False, beam_size=2,input_word=None,showing_attention=False,single_sequence=True,
                         printout_while_decoding=False,
                         verbose=2):

    output_seq = pad*np.ones(src_seq.size(), dtype=np.int64)
    # we start with the _START symbol
    output_seq[:, :, 0] = src_seq[:, :, 0] #CHAR_START_ID
    src_text_ls = []
    target_seq_gold_ls = [] if target_seq_gold is not None else None
    output_mask = np.ones(src_mask.size(), dtype=np.int64)
    output_mask[:, :, 1:] = 0
    output_len = Variable(torch.from_numpy(np.ones((src_seq.size(0), src_seq.size(1), 1), dtype=np.int64)), requires_grad=False)
    output_mask = Variable(torch.from_numpy(output_mask), requires_grad=False)
    output_seq = Variable(torch.from_numpy(output_seq), requires_grad=False)
    printing("Data Start source {} {} ", var=(src_seq, src_seq.size()), verbose=verbose, verbose_level=5)
    output_str = True
    printing("WARNING : output_str = True hardcoded (decode_sequence)", verbose=verbose, verbose_level=2)
    printing("Data output sizes ", var=(output_seq.size(), output_len.size(), output_mask.size()), verbose=verbose, verbose_level=6)

    # for beam dim we add a dimension
    # the first before starting decoding is the same for all beam

    output_seq = output_seq.unsqueeze(-1).expand(output_seq.size(0),
                                                 output_seq.size(1),
                                                 output_seq.size(2),
                                                 beam_size)

    # is going to store the log probability for all decoding step of all best beams
    #log_scores_ranked_former = torch.zeros(output_seq.size(0), output_seq.size(1), beam_size)
    log_scores_ranked_former_all_seq = torch.zeros(output_seq.size(0), output_seq.size(1),output_seq.size(2), beam_size)
    for step, char_decode in enumerate(range(2,  max_len)):
        if use_gpu:
            src_seq = src_seq.cuda()
            output_seq = output_seq.cuda()
            src_len = src_len.cuda()
            output_len = output_len.cuda()

        # for each sentence, each word, the current decoding state
        # is going to store all the scores foe each possible decoding token

        log_scores_all_candidates = torch.ones(output_seq.size(0), output_seq.size(1), len(model.char_dictionary.instance2index)+1, beam_size)*(-float("inf"))

        for candidate_ind in range(beam_size):
            # we decode the sequence for each beam
            decoding_states, word_pred, pos_pred, norm_not_norm, attention, _ = model.forward(input_seq=src_seq,
                                                                                              word_embed_input=input_word,
                                                                                              output_seq=output_seq[:, :, :, candidate_ind],
                                                                                              input_word_len=src_len, output_word_len=output_len)
            #model.forward(input_seq=src_seq,word_embed_input=input_word,output_seq=output_seq[:, :, :, candidate_ind],input_word_len=src_len, output_word_len=output_len)
            scores = model.generator.forward(x=decoding_states)
            # we get the log sores for each beam
            output_len = (src_len[:, :, 0] != 0).unsqueeze(dim=2) * char_decode
            log_softmax_score = nn.LogSoftmax(dim=-1)(scores)
            # get the score of the last predicted tokens
            pdb.set_trace()
            log_softmax_score = log_softmax_score[:, :, char_decode-2, :]#squeeze(-2)
            # we remove padded scores
            log_softmax_score = log_softmax_score[:, :log_softmax_score.size(1), :]
            # we sum along the voc dimension by expanding the
            # we sum along the sequence scores the former scores
            expand_score_former = log_scores_ranked_former_all_seq.sum(dim=2)[:, :log_softmax_score.size(1), candidate_ind].unsqueeze(-1)
            expand_score_former = expand_score_former.expand(output_seq.size(0), log_softmax_score.size(1), log_softmax_score.size(-1))
            # we update the log score of all candidates with the new ones
            pdb.set_trace()
            log_scores_all_candidates[:, :log_softmax_score.size(1), :, candidate_ind] = torch.add(log_softmax_score, expand_score_former)
            pdb.set_trace()
        # we find the best scores of all beam x decoded tokens
        log_scores_all_candidates_reshaped = log_scores_all_candidates.view(log_scores_all_candidates.size(0), log_scores_all_candidates.size(1),log_scores_all_candidates.size(2) * log_scores_all_candidates.size(3))
        #pdb.set_trace()
        log_score_best, index_pred = log_scores_all_candidates_reshaped.sort(dim=-1, descending=True)
        index_pred_candidate = index_pred[:, :, :2*beam_size]
        #pdb.set_trace()
        token_pred_id_cand, beam_id_cand  = get_beam_ind_token_ind(index_pred_candidate, beam_size)
        token_pred_id_cand = token_pred_id_cand[:, :, [2*k for k in range(beam_size)]]
        beam_id_cand = beam_id_cand[:, :, [2*k for k in range(beam_size)]]
        # for each sent , each word , the current decoded step : we associate the prediction to its beam
        #output_seq[0, 0, char_decode - 1, beam_id_cand[0, 0, 0]] = token_pred_id_cand[0, 0, 0]
        pdb.set_trace()
        output_seq, log_scores_ranked_former_all_seq = update_output_seq(output_seq, token_pred_id_cand, beam_id_cand,
                                                                         log_scores_ranked_former_all_seq, char_decode,
                                                                         log_score_best)
        pdb.set_trace()
        if printout_while_decoding:
            for beam in range(beam_size):
                pred_word_count, text_decoded, decoded_ls = output_text_(output_seq[:, :, :, beam],  # predictions,
                                                                     char_dictionary,
                                                                     single_sequence=single_sequence,
                                                                     output_str=output_str,
                                                                     last=char_decode == (max_len - 1),
                                                                     debug=False)
            print("WHILE DECODING BEAM {} sequence is {}".format(beam, text_decoded))

    src_word_count, src_text, src_all_ls = output_text_(src_seq, char_dictionary, single_sequence=single_sequence,
                                                        showing_attention=showing_attention,
                                                        output_str=output_str)
    printing("SOURCE  : array text {} ", var=[src_text], verbose=verbose, verbose_level=5)
    src_text_ls.extend(src_text)

    # we have now beam_size sequence
    beam_text = []
    for beam in range(beam_size):
        pred_word_count, text_decoded, decoded_ls = output_text_(output_seq[:, :, :, beam],  # predictions,
                                                                 char_dictionary,
                                                                 single_sequence=True,
                                                                 output_str=output_str,
                                                                 last=char_decode == (max_len - 1),
                                                                 debug=False)
        print("BEAM {} sequence is {}".format(beam, text_decoded))
        beam_text.append(text_decoded)
        if beam == 0:
            text_decoded_beam_first = text_decoded
            decoded_ls_beam_first = decoded_ls
            pred_word_count_beam_first = pred_word_count


    return (text_decoded_beam_first, src_text_ls, target_seq_gold_ls, None), \
           {
           "src_word_count": None,
           "target_word_count": None,
           "pred_word_count": pred_word_count_beam_first, "beam_ls":beam_text},\
           (None, None,), \
           (None, output_seq, src_seq, target_seq_gold)