import torch
import torch.nn as nn
from torch.autograd import Variable
from io_.info_print import printing
from io_.dat.normalized_writer import write_normalization
from io_.from_array_to_text import output_text, output_text_
import numpy as np
from evaluate.normalization_errors import score_norm_not_norm
from evaluate.normalization_errors import score_ls_, correct_pred_counter
from env.project_variables import WRITING_DIR
from io_.dat.constants import CHAR_START_ID
import pdb
import os
from collections import OrderedDict
#from toolbox.beam_related_reshape_ind import get_beam_ind_token_ind
from evaluate.visualize_attention import show_attention
from toolbox.norm_not_norm import get_label_norm
# EPSILON for the test of edit distance 
EPSILON = 0.000001
TEST_SCORING_IN_CODE = False


def _init_metric_report(score_to_compute_ls, mode_norm_score_ls):
    if score_to_compute_ls is not None:
        dic = {score+"-"+norm_mode: 0 for score in score_to_compute_ls for norm_mode in mode_norm_score_ls}
        dic.update({score + "-" + norm_mode + "-" + "n_sents": 0 for score in score_to_compute_ls for norm_mode in
                    mode_norm_score_ls})
        dic.update({score+"-"+norm_mode+"-"+"total_tokens": 0 for score in score_to_compute_ls for norm_mode in mode_norm_score_ls})
        dic.update({score+"-"+norm_mode+"-"+"mean_per_sent": 0 for score in score_to_compute_ls for norm_mode in mode_norm_score_ls})
        dic.update({score + "-" + norm_mode + "-" + "n_word_per_sent": 0 for score in score_to_compute_ls for norm_mode in mode_norm_score_ls})
        dic.update({token + "-norm_not_norm-" + type_ + "-count":0 for token in ["all","need_norm"] for type_ in ["pred", "gold"]})
        dic["all-norm_not_norm-pred_correct-count"] = 0
        dic["need_norm-norm_not_norm-pred_correct-count"] = 0
        return dic
    return None


def _init_metric_report_2():

    formulas = {"recall-normalization": ("NEED_NORM-normalization-pred_correct-count", "NEED_NORM-normalization-gold-count"),
                "tnr-normalization": ("NORMED-normalization-pred_correct-count", "NORMED-normalization-gold-count"),
                "precision-normalization": ("NEED_NORM-normalization-pred_correct-count", "NEED_NORM-normalization-pred-count"),
                "npv-normalization": ("NORMED-normalization-pred_correct-count", "NORMED-normalization-pred-count"),
                "accuracy-normalization": ("all-normalization-pred_correct-count", "all-normalization-gold-count"),
                "accuracy-per_sent-normalization": ("all-normalization-pred_correct_per_sent-count", "all-n_sents"),
                "info-all-per_sent": ("all-normalization-n_word_per_sent-count", "all-n_sents"),
                "info-NORMED-per_sent": ("NORMED-normalization-n_word_per_sent-count", "NORMED-n_sents"),
                "recall-per_sent-normalization": ("NEED_NORM-normalization-pred_correct_per_sent-count", "NEED_NORM-n_sents"),
                "info-NEED_NORM-per_sent": ("NEED_NORM-normalization-n_word_per_sent-count", "NEED_NORM-n_sents"),
                "tnr-per_sent-normalization": ("NORMED-normalization-pred_correct_per_sent-count", "NORMED-n_sents"),
                "aa": ("n_sents","n_sents")
                }


    formulas_2 = {
        "recall-norm_not_norm": ("need_norm-norm_not_norm-pred_correct-count", "need_norm-norm_not_norm-gold-count"),
        "precision-norm_not_norm": ("need_norm-norm_not_norm-pred_correct-count", "need_norm-norm_not_norm-pred-count"),
        "accuracy-norm_not_norm": ("all-norm_not_norm-pred_correct-count", "all-norm_not_norm-gold-count"),
        "IoU-pred-need_norm": (
        "need_norm-norm_not_normXnormalization-pred-count", "normed-norm_not_normUnormalization-pred-count"),
        "IoU-pred-normed": (
        "normed-norm_not_normXnormalization-pred-count", "need_norm-norm_not_normUnormalization-pred-count")
    }

    dic = OrderedDict()
    for a, (val, val2) in formulas.items():
        dic[val] = 0
        dic[val2] = 0
    for a, (val1, val12) in formulas_2.items():
        dic[val1] = 0
        dic[val12] = 0
    dic["all-normalization-pred-count"] = 0


    return dic


def greedy_decode_batch(batchIter, model,char_dictionary, batch_size, pad=1,
                        gold_output=False, score_to_compute_ls=None, stat=None,
                        use_gpu=False,
                        compute_mean_score_per_sent=False,
                        mode_norm_score_ls=None,
                        label_data=None, eval_new=False,
                        write_output=False, write_to="conll", dir_normalized=None, dir_original=None,

                        verbose=0):

        score_dic = _init_metric_report(score_to_compute_ls, mode_norm_score_ls)

        counter_correct = _init_metric_report_2()
        total_count = {"src_word_count": 0,
                       "target_word_count": 0,
                       "pred_word_count": 0}
        if mode_norm_score_ls is None:
            mode_norm_score_ls = ["all"]

        assert len(set(mode_norm_score_ls) & set(["all", "NEED_NORM", "NORMED"])) >0

        with torch.no_grad():
            for step, (batch, _) in enumerate(batchIter):
                # read src sequence
                src_seq = batch.input_seq
                src_len = batch.input_seq_len
                src_mask = batch.input_seq_mask
                target_gold = batch.output_seq if gold_output else None
                target_word_gold = batch.output_word if gold_output else None
                # do something with it : When do you stop decoding ?
                max_len = src_seq.size(-1)
                printing("WARNING : word max_len set to src_seq.size(-1) {} ", var=(max_len), verbose=verbose,
                         verbose_level=3)
                # decoding one batch
                if model.arguments["hyperparameters"]["decoder_arch"].get("char_decoding", True):
                    (text_decoded_ls, src_text_ls, gold_text_seq_ls), counts, _, \
                    (pred_norm, output_seq_n_hot, src_seq, target_seq_gold) = decode_sequence(model=model,
                                      char_dictionary=char_dictionary, single_sequence=False,
                                      target_seq_gold=target_gold, use_gpu=use_gpu,
                                      max_len=max_len, src_seq=src_seq,
                                      src_mask=src_mask, src_len=src_len, pad=pad, verbose=verbose)
                if model.arguments["hyperparameters"]["decoder_arch"].get("word_decoding", False):
                    (text_decoded_ls, src_text_ls, gold_text_seq_ls), counts, _, \
                    (pred_norm, output_seq_n_hot, src_seq, target_seq_gold) = decode_word(model, src_seq, src_len, target_word_gold=target_word_gold)

                if write_output:
                    if dir_normalized is None:
                        dir_normalized = os.path.join(WRITING_DIR, model.model_full_name+"-{}-normalized.conll".format(label_data))
                        dir_original = os.path.join(WRITING_DIR, model.model_full_name+"-{}-original.conll".format(label_data))

                    write_normalization(format=write_to, dir_normalized=dir_normalized, dir_original=dir_original,
                                        text_decoded_ls=text_decoded_ls, src_text_ls=src_text_ls, verbose=verbose)

                total_count["src_word_count"] += counts["src_word_count"]

                total_count["pred_word_count"] += counts["pred_word_count"]
                printing("Source text {} ", var=[(src_text_ls)], verbose=verbose, verbose_level=5)
                printing("Prediction {} ", var=[(text_decoded_ls)], verbose=verbose, verbose_level=5)
                if gold_output:

                    total_count["target_word_count"] += counts["target_word_count"]
                    # we can score
                    printing("Gold {} ", var=[(gold_text_seq_ls)], verbose=verbose, verbose_level=5)
                    # output exact score only
                    # sent mean not yet supported for npv and tnr, precision
                    counter_correct_batch, score_formulas = correct_pred_counter(ls_pred=text_decoded_ls,
                                                                                 ls_gold=gold_text_seq_ls,
                                                                                 output_seq_n_hot=output_seq_n_hot,
                                                                                 src_seq=src_seq,
                                                                                 target_seq_gold=target_seq_gold,
                                                                                 pred_norm_not_norm=pred_norm,
                                                                                 gold_norm_not_norm=batch.output_norm_not_norm,
                                                                                 ls_original=src_text_ls)

                    for key, val in counter_correct_batch.items():
                        try:
                            counter_correct[key] += val
                        except Exception as e:
                            counter_correct[key] += 0
                            print(e)
                            print("EXXCEPTION WHEN updating counter_correct {} and val {} is 0  ".format(key, val))
                    if score_to_compute_ls is not None and not eval_new:
                        for metric in score_to_compute_ls:
                            # TODO : DEPRECIATED : should be remove til # ---- no more need t set the norm/need_norm : all is done by default
                            for mode_norm_score in mode_norm_score_ls:
                                if metric not in ["norm_not_norm-F1", "norm_not_norm-Precision", "norm_not_norm-Recall", "norm_not_norm-accuracy"]:
                                    try:
                                        _score, _n_tokens = score_ls_(text_decoded_ls, gold_text_seq_ls, ls_original=src_text_ls,
                                                                      score=metric, stat=stat,
                                                                      compute_mean_score_per_sent=compute_mean_score_per_sent,
                                                                      normalized_mode=mode_norm_score,
                                                                      verbose=verbose)
                                        if compute_mean_score_per_sent:
                                            score_dic[metric + "-" + mode_norm_score + "-n_sents"] += _score["n_sents"]
                                            score_dic[metric + "-" + mode_norm_score + "-n_word_per_sent"] += _score["n_word_per_sent"]
                                            score_dic[metric + "-" + mode_norm_score+"-mean_per_sent"] += _score["mean_per_sent"]
                                        score_dic[metric + "-" + mode_norm_score] += _score["sum"]
                                        score_dic[metric + "-" + mode_norm_score + "-" + "total_tokens"] += _n_tokens

                                    except Exception as e:
                                        print("Exception {}".format(e))
                                        score_dic[metric + "-" + mode_norm_score] += 0
                                        #score_dic[metric + "-" + mode_norm_score + "-" + "total_tokens"] += 0
                                        if compute_mean_score_per_sent:
                                            score_dic[metric + "-" + mode_norm_score + "-n_sents"] += 0
                                            score_dic[metric + "-" + mode_norm_score + "-n_word_per_sent"] += 0
                                            score_dic[metric + "-" + mode_norm_score+"-mean_per_sent"] += 0

                            if batch.output_norm_not_norm is not None:
                                batch.output_norm_not_norm = batch.output_norm_not_norm[:, :pred_norm.size(1)]  # not that clean : we cut gold norm_not_norm sequence
                                _score, _ = score_norm_not_norm(pred_norm, batch.output_norm_not_norm)
                                # means : aux task is on
                                for token in ["all", "need_norm"]:
                                    for type_ in ["pred", "gold","pred_correct"]:
                                        if token == "all" and type_ == "pred":
                                            continue
                                        score_dic[token+"-norm_not_norm-"+type_+"-count"] += _score[token+"-norm_not_norm-"+type_+"-count"]
                    test_scoring = TEST_SCORING_IN_CODE
                    if test_scoring:
                        assert len(list((set(mode_norm_score_ls)&set(["NEED_NORM", "NORMED","all"])))) == 3, "ERROR : to perform test need all normalization mode "
                            #print("Scoring with mode {}".format(mode_norm_score))
                        for metric in score_to_compute_ls:
                            assert score_dic[metric + "-NEED_NORM-total_tokens"]+score_dic[metric + "-NORMED-total_tokens"] == score_dic[metric + "-all-total_tokens"], \
                                'ERROR all-total_tokens is {}  not equal to NEED NORMED {} +  NORMED {} '.format(score_dic[metric + "-all-total_tokens"], score_dic[metric + "-NEED_NORM-total_tokens"], score_dic[metric + "-NORMED-total_tokens"])
                            assert np.abs(score_dic[metric + "-NEED_NORM"]+score_dic[metric + "-NORMED"] - score_dic[metric + "-all"]) < EPSILON, \
                            "ERROR : correct NEED_NORM {} , NORMED {} and all {} ".format(score_dic[metric + "-NEED_NORM"], score_dic[metric + "-NORMED"], score_dic[metric + "-all"])
                            print("TEST PASSED")
            if gold_output :
                try:
                    assert total_count["src_word_count"] == total_count["target_word_count"], \
                        "ERROR src_word_count {} vs target_word_count {}".format(total_count["src_word_count"], total_count["target_word_count"])
                    assert total_count["src_word_count"] == total_count["pred_word_count"], \
                        "ERROR src_word_count {} vs pred_word_count {}".format(total_count["src_word_count"], total_count["pred_word_count"])
                except Exception as e:
                    print("EXCEPTION {} raised when checking src , pred and gold sequence".format(e))
                printing("Assertion passed : there are as many words in the source side,"
                         "the target side and"
                         "the predicted side : {} ".format(total_count["src_word_count"]), verbose_level=2, verbose=verbose)
            if eval_new:
                return counter_correct, score_formulas
            else:
                return score_dic, None


def decode_sequence_beam(model, max_len, src_seq, src_mask, src_len,char_dictionary,
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
            decoding_states, word_pred, norm_not_norm, attention = model.forward(input_seq=src_seq, output_seq=output_seq[:, :, :, candidate_ind],
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
        print(token_pred_id_cand)
        pdb.set_trace()

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
                            pdb.set_trace()
            pdb.set_trace()
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
                pad=1, target_word_gold=None, use_gpu=False,
                single_sequence=False, verbose=2):

    _, word_pred, norm_not_norm, _ = model.forward(input_seq=src_seq,
                                                   input_word_len=src_len)
    prediction = word_pred.argmax(dim=-1)
    prediction = prediction[:, :src_seq.size(1)]

    pred_norm_not_norm = norm_not_norm.argmax(dim=-1) if norm_not_norm is not None else None
    if pred_norm_not_norm is not None:
        pred_norm_not_norm = pred_norm_not_norm[:, :src_seq.size(1)]  # followign what's done above

    src_word_count, src_text, src_all_ls = output_text_(src_seq, model.char_dictionary,
                                                        single_sequence=single_sequence,
                                                        output_str=True)
    words_count_pred, text_decoded, _ = output_text_(prediction, word_decode=True, word_dic=model.word_nom_dictionary,
                                                     single_sequence=single_sequence, char_decode=False,
                                                     output_str=True)
    words_count_gold, target_word_gold_text, _ = output_text_(target_word_gold, word_decode=True,
                                                              word_dic=model.word_nom_dictionary,
                                                              single_sequence=single_sequence, char_decode=False,
                                                              output_str=True)
    if single_sequence:
        if pred_norm_not_norm is not None:
            pred_norm_not_norm = pred_norm_not_norm[0]

    return (text_decoded, src_text, target_word_gold_text), {"src_word_count": src_word_count,
                                                            "target_word_count": words_count_gold,
                                                            "pred_word_count": words_count_pred}, \
            (None, None,), \
            (pred_norm_not_norm, None, src_seq, target_word_gold)


def decode_sequence(model, char_dictionary, max_len, src_seq, src_mask, src_len,
                    pad=1, target_seq_gold=None,
                    use_gpu=False,
                    single_sequence=False, verbose=2):

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
    for step, char_decode in enumerate(range(2,  max_len)):
        if use_gpu:
            src_seq = src_seq.cuda()
            output_seq = output_seq.cuda()
            src_len = src_len.cuda()
            output_len = output_len.cuda()
        pdb.set_trace()
        decoding_states, word_pred, norm_not_norm, attention = model.forward(input_seq=src_seq,output_seq=output_seq,
                                                                             input_word_len=src_len,
                                                                             output_word_len=output_len)
        # [batch, seq_len, V]
        pred_norm_not_norm = norm_not_norm.argmax(dim=-1) if norm_not_norm is not None else None
        scores = model.generator.forward(x=decoding_states)
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

        printing("Prediction size {} ", var=(predictions.size()), verbose=verbose, verbose_level=4)
        printing("Prediction {} ", var=(predictions), verbose=verbose, verbose_level=5)

        printing("scores: {} scores {} scores sized  {} predicion size {} prediction {} outputseq ", var=(scores,
                 scores.size(), predictions.size(), predictions[:, -1],
                 output_seq.size()),
                 verbose=verbose, verbose_level=5)

        output_seq = output_seq[:, :scores.size(1), :]

        if pred_norm_not_norm is not None:
            pred_norm_not_norm = pred_norm_not_norm[:, :scores.size(1)]  # followign what's done above
        output_seq[:, :, char_decode - 1] = predictions[:, :, -1]

        if verbose >= 5:
            sequence = [" ".join([char_dictionary.get_instance(output_seq[sent, word_ind, char_i]) for char_i in range(max_len)])
                        + "|sent-{}|".format(sent) for sent in range(output_seq.size(0)) for word_ind in range(output_seq.size(1))]
        else:
            sequence = []

        printing("Decoding step {} decoded target {} ", var=(step, sequence), verbose=verbose, verbose_level=5)

        pred_word_count, text_decoded, decoded_ls = output_text_(output_seq,#predictions,
                                                                 char_dictionary, single_sequence=single_sequence,
                                                                 output_str=output_str, last=char_decode==(max_len-1),
                                                                 debug=False)
        printing("PREDICTION : array text {} ", var=[text_decoded],
                 verbose=verbose,
                 verbose_level=5)

    src_word_count, src_text, src_all_ls = output_text_(src_seq, char_dictionary, single_sequence=single_sequence,
                                                        output_str=output_str)
    src_text_ls.extend(src_text)
    if target_seq_gold is not None:
        target_word_count, target_text, _ = output_text_(target_seq_gold, char_dictionary,
                                                         single_sequence=single_sequence, output_str=output_str)
        target_seq_gold_ls.extend(target_text)
    else:
        target_word_count = None
    if single_sequence:
        if model.decoder.attn_layer is not None:
            attention = attention[0]
        if pred_norm_not_norm is not None:
            pred_norm_not_norm = pred_norm_not_norm[0]

    return (text_decoded, src_text_ls, target_seq_gold_ls), \
           {
           "src_word_count": src_word_count,
           "target_word_count": target_word_count,
           "pred_word_count": pred_word_count
           },\
           (attention, src_all_ls,), \
           (pred_norm_not_norm, output_seq, src_seq, target_seq_gold)


def decode_seq_str(seq_string, model, char_dictionary, pad=1,
                   dir_attention=None, save_attention=False,
                   show_att=False, beam_decode=False,beam_size=None,
                   max_len=20, verbose=2, sent_mode=False):
    assert sent_mode
    sent = seq_string.copy()
    # we add empty words at the end otherwie poblem !! # TODO : understand why ? is it because we need word padded at the end of the sentence ?
    sent.append("")
    with torch.no_grad():
        sent_character = []
        sent_words_mask = []
        sent_words_lens = []
        print("sent",sent)
        for seq_string in sent:
            if len(seq_string) > 0:
                _seq_string = ["_START"]
                printing("WARNING : we added _START symbol and _END_CHAR ! ", verbose=verbose, verbose_level=2)
                _seq_string.extend(list(seq_string))
                seq_string = _seq_string + ["_END_CHAR"] #["_END_CHAR"]#["_PAD_CHAR"]
            if len(seq_string) > max_len:
                # cutting to respect dim requirements
                seq_string = seq_string[:max_len-1]+["_PAD_CHAR"]
            if len(seq_string)>0:
                printing("INPUT SEQ is {} ", var=[seq_string], verbose=verbose, verbose_level=2)
            sequence_characters = [char_dictionary.get_index(letter) for letter in seq_string]+[pad for _ in range(max_len-len(seq_string))]
            sent_character.append(sequence_characters)
            masks = [1 for _ in seq_string]+[0 for _ in range(max_len-len(seq_string))]
            sent_words_mask.append(masks)
            words_lens = min(max_len, len(seq_string))
            sent_words_lens.append(words_lens)
            # we have to create batch_size == 2 because of bug

        batch = Variable(torch.from_numpy(np.array([sent_character, sent_character])),
                                       requires_grad=False)
        batch_masks = Variable(torch.from_numpy(np.array([sent_words_mask, sent_words_mask])), requires_grad=False)
        batch_lens = Variable(torch.from_numpy(np.array([sent_words_lens, sent_words_lens])), requires_grad=False)
        batch_lens = batch_lens.unsqueeze(dim=2)
        if beam_decode:
            decode_sequence_beam(model=model,char_dictionary=char_dictionary,
                                  max_len=max_len, src_seq=batch, src_len=batch_lens,beam_size=beam_size,
                                  src_mask=batch_masks, pad=pad, verbose=verbose)
        else:
            (text_decoded, src_text, target), _, (attention, src_seq), (pred_norm,_, _, _)  \
                = decode_sequence(model=model, char_dictionary=char_dictionary,
                                  max_len=max_len, src_seq=batch, src_len=batch_lens,
                                  src_mask=batch_masks, single_sequence=True, pad=pad, verbose=verbose)
        if attention is not None:
            print("Attention", attention, src_seq, text_decoded)
            for pred_word, src_word, attention_word in zip(text_decoded, src_seq, attention):

                show_attention(list(pred_word), src_word[:attention_word.size(1)],
                               attention_word.transpose(1, 0), save=save_attention, dir_save=dir_attention,show=show_att,
                               model_full_name=model.model_full_name)
            #show_attention("[lekfezlfkh efj ", ["se", "mjfsemkfj"], torch.tensor([[0, .4], [1, 0.6]]))

        if pred_norm is not None:
            norm_not_norm_seq = [(get_label_norm(norm), word) for norm, word in zip(pred_norm, src_text)]
            printing("NORMALIZING : {} ", var=[norm_not_norm_seq], verbose_level=0, verbose=0)
        printing("DECODED text is : {} original is {}",var=(text_decoded, src_text), verbose_level=0, verbose=0)


def decode_interacively(model, char_dictionary,  max_len, pad=1, sent_mode=False, save_attention=False,
                        show_attention=False, beam_decode=False,beam_size=None,
                        dir_attention=None, verbose=0):
    if char_dictionary is None:
        printing("INFO : dictionary is None so setting char_dictionary to model.char_dictionary",
                 verbose=verbose, verbose_level=0)
        char_dictionary = model.char_dictionary
    sentence = []
    while True:
        seq_string = input("Please type what you want to normalize word by word and "
                           "finishes by 'stop' ? to end type : 'END'    ")
        if seq_string == "":
            continue
        if seq_string == "stop":
            if not sent_mode:

                break
            else:
                decode_seq_str(seq_string=sentence, model=model, char_dictionary=char_dictionary, pad=pad, max_len= max_len,
                               show_att=show_attention, beam_decode=beam_decode,beam_size=beam_size,
                               verbose=verbose, sent_mode=True, dir_attention=dir_attention, save_attention=save_attention)
                sentence = []
        elif seq_string == "END":
            printing("ENDING INTERACTION", verbose=verbose, verbose_level=0)
            break
        else:
            sentence.append(seq_string)
            if not sent_mode:
                decode_seq_str(seq_string, model, char_dictionary, pad, max_len, verbose)