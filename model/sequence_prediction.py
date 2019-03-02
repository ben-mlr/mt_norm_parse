import torch
import torch.nn as nn
from torch.autograd import Variable
from io_.info_print import printing
from io_.dat.normalized_writer import write_normalization
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
#from toolbox.beam_related_reshape_ind import get_beam_ind_token_ind
from evaluate.visualize_attention import show_attention
from toolbox.norm_not_norm import get_label_norm
from io_.dat.constants import PAD, ROOT, END, ROOT_CHAR, END_CHAR
# EPSILON for the test of edit distance 
import time
EPSILON = 0.000001
TEST_SCORING_IN_CODE = False





def decode_sequence_beam(model, max_len, src_seq, src_mask, src_len, char_dictionary,
                         pad=1, target_seq_gold=None,
                         use_gpu=False, beam_size=2,
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
        log_scores_all_candidates = torch.ones(output_seq.size(0), output_seq.size(1), 109, beam_size)*(-float("inf"))

        for candidate_ind in range(beam_size):
            # we decode the sequence for each beam
            decoding_states, word_pred, pos_pred, norm_not_norm, attention = model.forward(input_seq=src_seq, output_seq=output_seq[:, :, :, candidate_ind],
                                                                                           input_word_len=src_len, output_word_len=output_len)
            scores = model.generator.forward(x=decoding_states)
            # we get the log sores
            output_len = (src_len[:, :, 0] != 0).unsqueeze(dim=2) * char_decode
            log_softmax_score = nn.LogSoftmax(dim=-1)(scores)
            # get the score of the last predicted tokens
            log_softmax_score = log_softmax_score[:, :, char_decode-2, :]#squeeze(-2)
            # we remove padded scores
            log_softmax_score = log_softmax_score[:, :log_softmax_score.size(1), :]
            # we sum along the voc dimension by expanding the
            #expand_score_former = log_scores_ranked_former[:, :log_softmax_score.size(1), candidate_ind].unsqueeze(-1)
            expand_score_former = log_scores_ranked_former_all_seq.sum(dim=2)[:, :log_softmax_score.size(1), candidate_ind].unsqueeze(-1)
            expand_score_former = expand_score_former.expand(output_seq.size(0),
                                                             log_softmax_score.size(1),
                                                             log_softmax_score.size(-1))
            # we update the log score of all candidates with the new ones

            log_scores_all_candidates[:, :log_softmax_score.size(1), :, candidate_ind] = torch.add(log_softmax_score, expand_score_former)
        # we find the best scores of all beam x decoded tokens
        log_scores_all_candidates_reshaped = log_scores_all_candidates.view(log_scores_all_candidates.size(0),
                                                                            log_scores_all_candidates.size(1),
                                                                            log_scores_all_candidates.size(2)*log_scores_all_candidates.size(3))
        log_score_best, index_pred = log_scores_all_candidates_reshaped.sort(dim=-1, descending=True)

        def get_beam_ind_token_ind(ind_flatted_ls, first_dim_in_view):
            first_ind = ind_flatted_ls / first_dim_in_view
            second_ind = ind_flatted_ls - (ind_flatted_ls / first_dim_in_view) * first_dim_in_view
            return first_ind, second_ind
        # get the predictions and update the output_seq foe each beam
        index_pred_candidate = index_pred[:, :, :beam_size]
        beam_id_cand, token_pred_id_cand = get_beam_ind_token_ind(index_pred_candidate, 109)
        # for each sent , each word , the current decoded step : we associate the prediction to its beam
        #output_seq[0, 0, char_decode - 1, beam_id_cand[0, 0, 0]] = token_pred_id_cand[0, 0, 0]

        def update_output_seq(output_seq_, token_pred_id_cand, beam_id_cand,log_scores_ranked_former_all_seq, char_decode_step):
            output_seq_1 = output_seq_.clone()
            log_scores_ranked_former_all_seq_1 = log_scores_ranked_former_all_seq.clone()
            for sent in range(output_seq_.size(0)):
                for word in range(output_seq_.size(1)):
                    for ind_new_beam in range(output_seq_.size(3)):
                        #beam_id_cand[sent, word, beam]
                        beam = beam_id_cand[sent, word, ind_new_beam]
                        # we set the new token prediction
                        if beam != ind_new_beam:
                            #pdb.set_trace()
                            output_seq_1[sent, word, char_decode_step - 2, ind_new_beam] = output_seq_1[sent, word, char_decode_step - 2, beam]
                            log_scores_ranked_former_all_seq_1[sent, word, char_decode_step - 2, ind_new_beam] = log_scores_ranked_former_all_seq_1[sent, word, char_decode_step - 2, beam]
                        # We update the former step of the new beam ind_new_beam with the ones of the beam we decode
                        output_seq_1[sent, word, char_decode_step - 1, ind_new_beam] = token_pred_id_cand[sent, word, ind_new_beam]
                        log_scores_ranked_former_all_seq_1[sent, word, char_decode_step - 1, ind_new_beam] = log_score_best[sent, word, ind_new_beam]
                        if word == 0 and sent == 0:
                            print("WORD SENT 00")
                            print(token_pred_id_cand[sent, word, ind_new_beam])
                            print(output_seq_[sent, word, char_decode_step - 1, ind_new_beam])
                            print(output_seq_[sent, word, char_decode_step - 1, :])
            return output_seq_1, log_scores_ranked_former_all_seq_1

        output_seq, log_scores_ranked_former_all_seq = update_output_seq(output_seq, token_pred_id_cand, beam_id_cand, log_scores_ranked_former_all_seq, char_decode)

        if False:
            for candidate_ind in range(beam_size):
                index_pred_top = index_pred[:, :, candidate_ind]
                beam_id, token_pred_id = get_beam_ind_token_ind(index_pred_top, 109)


                output_seq = output_seq[:, :log_scores_all_candidates.size(1), :, :]
                pdb.set_trace()
                # is it the right order of token_pred_id
                for update_beam in range(beam_size):
                    output_seq[:, :, char_decode-2, update_beam] = token_pred_id
                    output_seq[:, :, ]
                print(token_pred_id)
                log_scores_ranked_former[:, :, candidate_ind] = log_score_best[:, :, candidate_ind]
            # so on until we end decoding
    # we have now beam_size sequence
    for beam in range(beam_size):
        pred_word_count, text_decoded, decoded_ls = output_text_(output_seq[:, :, :, beam],  # predictions,
                                                                 char_dictionary,
                                                                 single_sequence=True,
                                                                 output_str=output_str,
                                                                 last=char_decode == (max_len - 1),
                                                                 debug=False)
        print("BEAM {} sequence is {}".format(beam, text_decoded))


def decode_word(model, src_seq, src_len,
                pad=1, target_word_gold=None,
                use_gpu=False, target_pos_gold=None,
                mode="word", input_word=None,
                single_sequence=False, verbose=2):
    """
    NB : could be more factorized (its exactly the same prediction the only difference is the dictionary
    """
    _, word_pred, pos_pred, norm_not_norm, _ = model.forward(input_seq=src_seq,
                                                             input_word_len=src_len,
                                                             word_embed_input=input_word,
                                                             word_level_predict=True)
    pred_norm_not_norm = None
    src_word_input = None
    if mode == "word":
        assert target_pos_gold is None, "Only target_word_gold should be provided"
        if target_word_gold is None:
            print("INFO decoding with no gold reference")
        prediction = word_pred.argmax(dim=-1)

        # we trust the predictor to do the padding !
        src_seq = src_seq[:, :prediction.size(1)]
        # should not be mandatory right ?
        if target_word_gold is not None:
            target_word_gold = target_word_gold[: , :prediction.size(1)]
        if norm_not_norm is not None:
            pred_norm_not_norm = norm_not_norm.argmax(dim=-1)
        if pred_norm_not_norm is not None:
            pred_norm_not_norm = pred_norm_not_norm[:, :src_seq.size(1)]  # followign what's done above

        src_word_count, src_text, src_all_ls = output_text_(src_seq, model.char_dictionary,
                                                            single_sequence=single_sequence, debug=False,
                                                            output_str=True)
        words_count_pred, text_decoded, _ = output_text_(prediction, word_decode=True, word_dic=model.word_dictionary,#word_nom_dictionary,
                                                         single_sequence=single_sequence, char_decode=False,
                                                         debug=False,
                                                         output_str=True)
        pdb.set_trace()
        if target_word_gold is not None:
            assert model.word_nom_dictionary is not None, "ERROR : word_nom_dictionary is required"
            words_count_gold, target_word_gold_text, _ = output_text_(target_word_gold,#input_word,#target_word_gold,
                                                                      word_decode=True,
                                                                      word_dic=model.word_nom_dictionary,
                                                                      debug=False,
                                                                      single_sequence=single_sequence, char_decode=False,
                                                                      output_str=True)
        else:
            words_count_gold, target_word_gold_text = None, None

        if input_word is not None:
            words_embed_count_src, src_word_input, _ = output_text_(input_word,
                                                                      word_decode=True,
                                                                      word_dic=model.word_dictionary,
                                                                      # model.word_nom_dictionary,
                                                                      debug=False, single_sequence=single_sequence,
                                                                      char_decode=False, output_str=True)

        # fix by hand  # TODO : check if it is corret
        # its based on src_text sequence length because we assumed word to word mapping and we want to predict without gold
        text_decoded = [sent[:len(ls_gold)] for sent, ls_gold in zip(text_decoded, src_text)]
        words_count_pred = sum([len(sent) for sent in text_decoded])
        printing("GOLD : array text {} ", var=[target_word_gold_text], verbose=verbose, verbose_level=5)
        printing("SRC : array text {} ", var=[src_text], verbose=verbose, verbose_level=5)
        printing("PRED : array text {} ", var=[text_decoded], verbose=verbose, verbose_level=5)

    elif mode == "pos":
        assert target_word_gold is None, "Only target_pos_gold should be provided"
        assert target_pos_gold is not None
        prediction_pos = pos_pred.argmax(dim=-1)
        src_seq = src_seq[:, :prediction_pos.size(1)]
        if target_pos_gold is not None:

            target_pos_gold = target_pos_gold[:, :prediction_pos.size(1)]

        src_word_count, src_text, src_all_ls = output_text_(src_seq, model.char_dictionary,
                                                            single_sequence=single_sequence,debug=False,
                                                            output_str=True)
        words_count_pred, text_decoded, _ = output_text_(prediction_pos, word_decode=True,
                                                         word_dic=model.pos_dictionary,
                                                         single_sequence=single_sequence, char_decode=False,
                                                         debug=False,
                                                         output_str=True)
        if target_pos_gold is not None:

            words_count_gold, target_word_gold_text, _ = output_text_(target_pos_gold, word_decode=True,
                                                                      word_dic=model.pos_dictionary,
                                                                      debug=False, single_sequence=single_sequence,
                                                                      char_decode=False, output_str=True)
            text_decoded = [sent[:len(ls_gold)] for sent, ls_gold in zip(text_decoded, src_text)]
            words_count_pred = sum([len(sent) for sent in text_decoded])
        else:
            target_word_gold_text = None
            words_count_gold = None
    if single_sequence:
        if pred_norm_not_norm is not None:
            pred_norm_not_norm = pred_norm_not_norm[0]
    return (text_decoded, src_text, target_word_gold_text, src_word_input), \
           {"src_word_count": src_word_count, "target_word_count": words_count_gold, "pred_word_count": words_count_pred}, \
           (None, None,), \
           (pred_norm_not_norm, None, src_seq, target_word_gold)


def decode_sequence(model, char_dictionary, max_len, src_seq, src_mask, src_len,
                    pad=1, target_seq_gold=None, input_word=None,
                    use_gpu=False,
                    single_sequence=False, eval_time=True, verbose=2,
                    timing=False):

    #eval_time alays True for now
    printing("EVAL TIME is {}", var=eval_time, verbose=verbose, verbose_level=2)
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
    start_decode_sequence = time.time() if timing else None

    for step, char_decode in enumerate(range(2,  max_len)):
        if use_gpu:
            src_seq = src_seq.cuda()
            output_seq = output_seq.cuda()
            src_len = src_len.cuda()
            output_len = output_len.cuda()
        start = time.time() if timing else None
        decoding_states, word_pred, pos_pred, norm_not_norm, attention = model.forward(input_seq=src_seq,
                                                                                       output_seq=output_seq,
                                                                                       input_word_len=src_len,
                                                                                       output_word_len=output_len,
                                                                                       word_embed_input=input_word)
        time_forward, start = get_timing(start)
        # [batch, seq_len, V]

        pred_norm_not_norm = norm_not_norm.argmax(dim=-1) if norm_not_norm is not None else None
        scores = model.generator.forward(x=decoding_states)
        time_generate, start = get_timing(start)
        # each time step predict the most likely
        # len
        # output_len defined based on src_len to remove empty words
        output_len = (src_len[:, :, 0] != 0).unsqueeze(dim=2)*char_decode
        printing("DECODER step {} output len {} ", var=(step, output_len), verbose=verbose, verbose_level=3)
        #output_len[:] = char_decode # before debugging
        # mask
        output_mask = np.ones(src_seq.size(), dtype=np.int64)
        output_mask[:, char_decode:] = 0
        # new seq
        predictions = scores.argmax(dim=-1)

        time_argmax_printing, start = get_timing(start)
        if verbose >= 4:
            # .size() takes some time
            printing("Prediction size {} ", var=(predictions.size()), verbose=verbose, verbose_level=4)
            printing("SCORES {} ", var=[str(scores)], verbose=verbose, verbose_level=5)
            printing("Prediction {} ", var=[predictions], verbose=verbose, verbose_level=5)

            printing("scores: {} scores {} scores sized  {} predicion size {} prediction {} outputseq ", var=(scores,
                     scores.size(), predictions.size(), predictions[:, -1],
                     output_seq.size()),
                     verbose=verbose, verbose_level=5)
        time_printing, start = get_timing(start)

        output_seq = output_seq[:, :scores.size(1), :]
        time_output_seq, start = get_timing(start)
        if pred_norm_not_norm is not None:
            pred_norm_not_norm = pred_norm_not_norm[:, :scores.size(1)]  # followign what's done above
        output_seq[:, :, char_decode - 1] = predictions[:, :, -1]

        if verbose >= 5:
            sequence = [" ".join([char_dictionary.get_instance(output_seq[sent, word_ind, char_i]) for char_i in range(max_len)])
                        + "|sent-{}|".format(sent) for sent in range(output_seq.size(0)) for word_ind in range(output_seq.size(1))]
        else:
            sequence = []

        printing("Decoding step {} decoded target {} ", var=(step, sequence), verbose=verbose, verbose_level=5)
        time_sequence_text, start = get_timing(start)

        if eval_time:
            # at test time : if all prediction in the batch are whether PAD symbol or END symbol : we break
            if ((predictions[:, :, -1] == PAD_ID_CHAR) + (predictions[:, :, -1] == CHAR_END_ID)).all():
                printing("PREDICTION IS ONLY PAD or END SYMBOL SO BREAKING DECODING", verbose=verbose, verbose_level=1)
                break
    # no need to do that in the loop
    pred_word_count, text_decoded, decoded_ls = output_text_(output_seq,
                                                             char_dictionary, single_sequence=single_sequence,
                                                             output_str=output_str, last=char_decode==(max_len-1),
                                                             debug=False)
    time_output_text, start = get_timing(start)
    time_decoding_all_seq, start  = get_timing(start_decode_sequence)
    printing("PREDICTION : array text {} ", var=[text_decoded], verbose=verbose, verbose_level=5)
    src_word_count, src_text, src_all_ls = output_text_(src_seq, char_dictionary, single_sequence=single_sequence,
                                                        output_str=output_str)
    printing("SOURCE  : array text {} ", var=[src_text], verbose=verbose, verbose_level=5)
    src_text_ls.extend(src_text)
    if target_seq_gold is not None:
        target_word_count, target_text, _ = output_text_(target_seq_gold, char_dictionary,
                                                         single_sequence=single_sequence, output_str=output_str)
        target_seq_gold_ls.extend(target_text)
        printing("GOLD : array text {} ", var=[target_text], verbose=verbose, verbose_level=5)
    else:
        target_word_count = None
    if single_sequence:
        if model.decoder.attn_layer is not None:
            attention = attention[0]
        if pred_norm_not_norm is not None:
            pred_norm_not_norm = pred_norm_not_norm[0]
    if timing:
        print("DECODING TIME : {}".format(
            OrderedDict([("time_decoding_all_seq", time_decoding_all_seq),
                         ("time_forward", time_forward),
                         ("time_generate", time_generate),
                         ("time_argmax_printing", time_argmax_printing),
                         ("time_printing", time_printing),
                         ("time_output_seq", time_output_seq),
                         ("time_sequence_text", time_sequence_text),
                         ("time_output_text",time_output_text),
                         ("time_decoding_all_seq",time_decoding_all_seq)
                         ])))

    return (text_decoded, src_text_ls, target_seq_gold_ls, None), \
           {
           "src_word_count": src_word_count,
           "target_word_count": target_word_count,
           "pred_word_count": pred_word_count},\
           (attention, src_all_ls,), \
           (pred_norm_not_norm, output_seq, src_seq, target_seq_gold)


