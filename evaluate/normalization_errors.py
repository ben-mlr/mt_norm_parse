from scipy.stats import hmean
import numpy as np
from nltk import edit_distance
from io_.info_print import printing
import pdb
from env.project_variables import SUPPORTED_STAT
from collections import OrderedDict
from io_.dat.constants import PAD_ID_CHAR, ROOT_CHAR, END, ROOT_POS, END_POS
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

smoothing = SmoothingFunction()


def exact_match(pred, gold):
    if pred == gold:
        return 1
    else:
        return 0


def edit_inverse(pred, gold):
    edit = edit_distance(pred, gold)
    try:
        return 1-edit/max(len(pred), len(gold))
    except Exception as e:
        print("ERROR {} ".format(e))
        print("pred {} gold {}".format(pred, gold))


SCORING_FUNC_AVAILABLE = ["exact_match", "edit_inverse", "BLUE"]


def score_ls(ls_pred, ls_gold, scoring_func, stat="mean", verbose=0):
    # give score sum of based on METRIC_DIC evaluator of list of words gold and pred
    assert scoring_func in SCORING_FUNC_AVAILABLE
    assert stat in SUPPORTED_STAT ,"ERROR : metric should be in {} ".format(str(SUPPORTED_STAT))
    assert len(ls_gold) == len(ls_pred), "ERROR ls_gold is len {} while ls_pred is len {} ".format(len(ls_gold), len(ls_pred))

    scores = []
    for gold, pred in zip(ls_gold, ls_pred):
        eval_func = eval(scoring_func)
        scores.append(eval_func(pred, gold))

    if stat == "sum":
        score = np.sum(scores)

    return score, len(scores)


def correct_pred_counter(ls_pred, ls_gold, ls_original, pred_norm_not_norm=None, gold_norm_not_norm=None,
                         output_seq_n_hot=None, src_seq=None, target_seq_gold=None, task="normalize",
                         scoring_func="exact_match",
                         in_vocab_ls=None, verbose=0):
    # only exact score here !!
    assert task in ["normalize", "pos", "norm_not_norm"]
    print("EVALUATING correct_pred_counter task ", task)
    assert scoring_func in SCORING_FUNC_AVAILABLE
    dic = OrderedDict()
    normalized_mode_ls = []
    if task == "normalize" or task == "pos":

        assert len(ls_gold) == len(ls_pred), "ERROR ls_gold is len {} vs {} : {} while ls_pred is {} ".format(len(ls_gold),
                                                                                                              len(ls_pred),
                                                                                                              ls_gold,
                                                                                                              ls_pred)
        assert len(ls_gold) == len(ls_original), "ERROR ls_gold is len {} vs {} : {} while src is {} ".format(len(ls_gold),
                                                                                                              len(ls_original),
                                                                                                              ls_gold,
                                                                                                              ls_original)
        normalization_ls = [[src_token == gold_token for src_token, gold_token in zip(batch_src, batch_gold)] for batch_src, batch_gold in zip(ls_original, ls_gold)]
        # if False : sequence predictor predicts NEED_NORM otherwise NORMED
        normalization_prediction_by_seq = [[src_token == pred_token for src_token, pred_token in zip(batch_src, batch_pred)] for batch_src, batch_pred in zip(ls_original, ls_pred)]
        #normalization_prediction_by_seq_norm = len(normalization_prediction_by_seq[normalization_prediction_by_seq == True])
        #normalization_prediction_by_seq_need_norm = len(normalization_prediction_by_seq[normalization_prediction_by_seq == False])
        #assert len(normalization_prediction_by_seq) == len(normalization_ls)
        normalized_mode_ls = ["all", "NEED_NORM", "NORMED"] if task == "normalize" else ["all"]
        if in_vocab_ls is not None:
            normalized_mode_ls.extend(["InV", "OOV"])
        for normalized_mode in normalized_mode_ls:

            scores = []
            sent_score_ls = []
            try:
                assert ls_original is not None, "ERROR : need to provide original sequence to compute NORMED/NEED_NORM analysis"
                assert len(ls_original) == len(ls_gold), "ERROR ls_original sizes provided does not fit pred and gold"
            except Exception as e:
                print(e)
                print("ERROR len(ls_original) != len(ls_gold) ", len(ls_original) , len(ls_gold))
            if normalized_mode in ["NEED_NORM", "NORMED"]:
                norm_mode = 1 if normalized_mode == "NORMED" else 0
                _ls_gold = [[token for token, normed in zip(batch, batch_norm) if normed == norm_mode] for batch, batch_norm in zip(ls_gold, normalization_ls)]
                _ls_pred = [[token for token, normed in zip(batch, batch_norm) if normed == norm_mode] for batch, batch_norm in zip(ls_pred, normalization_ls)]
            elif normalized_mode == "all":
                _ls_gold = ls_gold
                _ls_pred = ls_pred
            elif normalized_mode in ["InV", "OOV"]:
                if normalized_mode == "InV":
                    gold_pred = [[(token, normalization) for token, normalization in zip(batch, batch_pred) if token in in_vocab_ls] for batch, batch_pred in zip(ls_gold, ls_pred)]
                else:
                    gold_pred = [[(token, normalization) for token, normalization in zip(batch, batch_pred) if token not in in_vocab_ls] for batch, batch_pred in zip(ls_gold, ls_pred)]
                _ls_gold = [[tup[0] for tup in batch] for batch in gold_pred]
                _ls_pred = [[tup[1] for tup in batch] for batch in gold_pred]

            else:
                print("ERROR normalized_mode {} not supported".format(normalized_mode))
                raise(Exception)
            for ind, (gold_sent, pred_sent) in enumerate(zip(_ls_gold, _ls_pred)):
                try:
                    assert len(gold_sent) == len(pred_sent), "len : pred {}, gold {} - pred {} gold {} " \
                                                             "(normalized_mode is {})".format(len(pred_sent), len(gold_sent), pred_sent, gold_sent, normalized_mode)
                except Exception as e:
                    print("Assertion failed")
                    print(e)
                    pdb.set_trace()
                sent_score = []
                for word_gold, word_pred in zip(gold_sent, pred_sent):

                    if scoring_func == "BLUE":
                        word_pred = word_pred.split()
                        word_gold = word_gold.split()
                        score_word = sentence_bleu(references=[word_gold], hypothesis=word_pred, smoothing_function=smoothing.method3)
                        sent_score.append(score_word)
                        scores.append(score_word)
                        printing("GOLD {} PRED {} BLEU {} ".format(word_gold, word_pred, score_word), verbose_level=2, verbose=verbose)
                    else:
                        eval_func = eval(scoring_func)
                        if word_gold not in [ROOT_CHAR, END, ROOT_POS, END_POS]:
                            score_word = eval_func(word_pred, word_gold)
                            sent_score.append(score_word)
                            scores.append(score_word)
                            printing("GOLD {} PRED {} exact_match {} on task {} ".format(word_gold, word_pred, score_word, task),verbose_level=2, verbose=verbose)
                        else:
                            score_word = "not given cause special char"

                sent_score_ls.append(sent_score)

            score = np.sum(scores)

            n_mode_words_per_sent = np.sum([len(scores_sent) for scores_sent in sent_score_ls])
            n_sents = len([a for a in sent_score_ls if len(a) > 0])
            normalized_sent_error_out_of_overall_sent_len = [np.sum(scores_sent)/len(scores_sent) if len(scores_sent) else 0  for scores_sent in sent_score_ls]
            mean_score_per_sent = np.sum(normalized_sent_error_out_of_overall_sent_len)

            if normalized_mode == "all":
                count_pred_number = len(scores)
            else:
                cond = False if normalized_mode == "NEED_NORM" else True
                normalization_ls_flat = np.array([a for ls in normalization_prediction_by_seq for a in ls])
                count_pred_number = len(normalization_ls_flat[normalization_ls_flat == cond])
            return_dic = {
                          normalized_mode+"-"+task+"-pred_correct-count": score,
                          normalized_mode+"-"+task+"-n_word_per_sent-count": n_mode_words_per_sent,
                          normalized_mode+"-"+task+"-pred_correct_per_sent-count": mean_score_per_sent,
                          normalized_mode+"-"+task+"-gold-count": len(scores),
                          normalized_mode+"-"+task+"-n_sents": n_sents
                         }
            if task == "normalize" and normalized_mode not in ["InV", "OOV"]:
                return_dic[normalized_mode + "-" + task + "-pred-count"] = count_pred_number
            dic.update(return_dic)
            # DEPRECIATED
            dic["n_sents"] = len(sent_score_ls)
            pdb.set_trace()

    formulas_bin = None
    if task in ["normalize", "norm_not_norm"]:
        if task == "norm_not_norm":
            assert pred_norm_not_norm is not None and gold_norm_not_norm is not None," ERROR pred_norm_not_norm and gold_" \
                                                                                     " norm_not_norm are empty while they should not {} - {}".format(pred_norm_not_norm, gold_norm_not_norm)
        if pred_norm_not_norm is not None and gold_norm_not_norm is not None:
            score_binary, formulas_bin = score_norm_not_norm(pred_norm_not_norm, gold_norm_not_norm[:, :pred_norm_not_norm.size(1)], output_seq_n_hot, src_seq, target_seq_gold)
            # testing consistency in counting
            try:
                assert score_binary["all-norm_not_norm-gold-count"] == dic["all-normalize-gold-count"], \
                    "ERROR : inconsistency between score binary gold count on all and on sequence prediction"
                assert score_binary["need_norm-norm_not_norm-gold-count"] == dic["NEED_NORM-normalize-gold-count"], \
                    "ERROR : inconsistency between score binary gold count on NEED_NORM and on sequence prediction"
            except Exception as e:
                pdb.set_trace()
                print("Assertion failed : CONSISTENCY between two tasks in terms of tokens ", e)
            dic.update(score_binary)


    # output raw performance counting and formulas associated
    # to a metric name:(numerator/denominator) for higher level score just as to sum

    # SHOULD MAKE DISTINCT CASES BETWEEN EXACT MATCH (as here) , EDIT, and BLUE (BLUE SCORE FORMULA ONly at the sentence ,
    # word lebel()
    formulas = dict()
    if task != "norm_not_norm":
        _formulas = {
                "accuracy-"+task+"": ("all-"+task+"-pred_correct-count", "all-"+task+"-gold-count"),
                "accuracy-per_sent-"+task+"": ("all-"+task+"-pred_correct_per_sent-count", "all-"+task+"-n_sents"),
                "info-all-per_sent": ("all-"+task+"-n_word_per_sent-count", "all-"+task+"-n_sents"),
                "info-all_tokens-"+task+"": ("all-"+task+"-gold-count"),
                "info-"+task+"-n_sents": ("all-"+task+"-n_sents")
                }
        formulas.update(_formulas)

    if task == "normalize":
        formulas_fine = {"recall-"+task+"": ("NEED_NORM-"+task+"-pred_correct-count", "NEED_NORM-"+task+"-gold-count"),
                         "tnr-"+task+"": ("NORMED-"+task+"-pred_correct-count", "NORMED-"+task+"-gold-count"),
                         "precision-"+task+"": ("NEED_NORM-"+task+"-pred_correct-count", "NEED_NORM-"+task+"-pred-count"),
                         "npv-"+task+"": ("NORMED-"+task+"-pred_correct-count", "NORMED-"+task+"-pred-count"),
                         "recall-per_sent-"+task+"": ("NEED_NORM-"+task+"-pred_correct_per_sent-count", "NEED_NORM-"+task+"-n_sents"),
                         "tnr-per_sent-"+task+"": ("NORMED-"+task+"-pred_correct_per_sent-count", "NORMED-"+task+"-n_sents"),
                         "info-NEED_NORM_tokens-"+task+"": ("NEED_NORM-"+task+"-gold-count"),
                         ""
                         "info-NORMED_tokens-"+task+"": ("NORMED-"+task+"-gold-count"),
                         "info-NORMED-per_sent": (
                         "NORMED-" + task + "-n_word_per_sent-count", "NORMED-" + task + "-n_sents"),
                         "info-NEED_NORM-per_sent": ("NEED_NORM-" + task + "-n_word_per_sent-count", "NEED_NORM-" + task + "-n_sents"),

                         }
        if len(set(["InV","OOV"])&set(normalized_mode_ls))>0:
            for vocab_filter in ["InV","OOV"]:
                formulas.update({vocab_filter+"-accuracy-"+task+"": (vocab_filter+"-"+task+"-pred_correct-count", vocab_filter+"-"+task+"-gold-count"),
                                 vocab_filter+"-accuracy_per_sent-"+task+"": (vocab_filter+"-"+task+"-pred_correct_per_sent-count", vocab_filter+"-"+task+"-n_sents")
                                })
        formulas.update(formulas_fine)

    if formulas_bin is not None:
        formulas.update(formulas_bin)

    return dic, formulas


def get_same_word_batch(output_seq_n_hot, src_seq):
    pred_normed_need_norm_by_seq_pred = torch.empty(output_seq_n_hot.size(0), output_seq_n_hot.size(1), dtype=torch.int)
    for sent in range(output_seq_n_hot.size(0)):
        for word in range(output_seq_n_hot.size(1)):
            pred_normed_need_norm_by_seq_pred[sent, word] = \
                int(torch.equal(output_seq_n_hot[sent, word, :][output_seq_n_hot[sent, word, :] != PAD_ID_CHAR], src_seq[sent, word, :][src_seq[sent, word, :] != PAD_ID_CHAR]))
    return pred_normed_need_norm_by_seq_pred


def score_norm_not_norm(norm_not_norm_pred, norm_not_norm_gold, output_seq_n_hot=None, src_seq=None, target_seq_gold=None):
    # remove padding
    predicted_not_pad = norm_not_norm_pred[norm_not_norm_gold != 2]
    gold_not_pad = norm_not_norm_gold[norm_not_norm_gold != 2]
    # if we provide output_seq_n_hot and src_seq we compute score on agreement of both tasks
    if output_seq_n_hot is not None:
        # TODO : confirm why you need this , why more words in src than in prediction
        src_seq = src_seq[:, :output_seq_n_hot.size(1), :]
        need_norm_norm = get_same_word_batch(output_seq_n_hot, src_seq)
        # we trust padding from norm_not_norm_gold to compute metric on sequence prediction
        predicted_not_pad_seq = need_norm_norm[norm_not_norm_gold != 2]
        # sequence prediction based norm_not_norm
        predicted_not_pad_seq_need_norm = np.argwhere(predicted_not_pad_seq == 0)[0,:]#.squeeze()
        predicted_not_pad_seq_normed = np.argwhere(predicted_not_pad_seq == 1)#[0,:]#.squeeze()
        # binary predition based norm_not_norm # TODO : confirm the inversion
        try:
            predicted_not_pad_need_norm = np.argwhere(predicted_not_pad == 1)[0,:]#.squeeze()
        except:
            print("WARNING : no need_norm predicted : == 1 [0,:] failed on predicted_not_pad  {} ".format(predicted_not_pad))
            predicted_not_pad_need_norm = torch.tensor([])
        try:
            predicted_not_pad_normed = np.argwhere(predicted_not_pad == 0)[0, :]#.squeeze()
        except:
            print("WARNING : no normed predicted : ==0 [0,:] failed on predicted_not_pad_normed {} ".format(predicted_not_pad))
            predicted_not_pad_normed = torch.tensor([])
        try:
            need_norm_norm_not_normUnormalization_pred_count = len(set(predicted_not_pad_need_norm)) + len(
                set(predicted_not_pad_seq_need_norm)) - len(
                list(set(predicted_not_pad_need_norm.tolist()) & set(predicted_not_pad_seq_need_norm.tolist())))
            normed_norm_not_normUnormalization_pred_count = len(set(predicted_not_pad_seq_normed))+len(set(predicted_not_pad_normed))-len(list(set(predicted_not_pad_seq_normed.tolist()) & set(predicted_not_pad_normed.tolist())))
            need_norm_norm_not_normXnormalization_pred_count = len(
                list(set(predicted_not_pad_need_norm.tolist()) & set(predicted_not_pad_seq_need_norm.tolist())))
            normed_norm_not_normXnormalization_pred_count = len(
                list(set(predicted_not_pad_seq_normed.tolist()) & set(predicted_not_pad_normed.tolist())))
        except Exception as e:
            # TODO MULTITASK : handle case were norm_not_norm by itself
            print("ERROR ", Exception(e))
            need_norm_norm_not_normUnormalization_pred_count = None
            normed_norm_not_normUnormalization_pred_count = None
            need_norm_norm_not_normXnormalization_pred_count = None
            normed_norm_not_normXnormalization_pred_count = None

    else:
        need_norm_norm_not_normUnormalization_pred_count = None
        normed_norm_not_normUnormalization_pred_count = None
        need_norm_norm_not_normXnormalization_pred_count = None
        normed_norm_not_normXnormalization_pred_count = None

    pred_correct_need_norm_prediction_count = np.sum(np.array(gold_not_pad == predicted_not_pad)[np.array(gold_not_pad) == 0])
    pred_correct_prediction_count = np.sum(np.array(gold_not_pad == predicted_not_pad))
    total_word = len(gold_not_pad)
    assert len(gold_not_pad) == len(predicted_not_pad)

    gold_need_norm_count = len(np.array(gold_not_pad[gold_not_pad == 0]))
    pred_need_norm_count = len(np.array(predicted_not_pad[predicted_not_pad == 0]))

    # per sent not supported for auxillary
    formulas = {
        "recall-norm_not_norm": ("need_norm-norm_not_norm-pred_correct-count", "need_norm-norm_not_norm-gold-count"),
        "precision-norm_not_norm": ("need_norm-norm_not_norm-pred_correct-count", "need_norm-norm_not_norm-pred-count"),
        "accuracy-norm_not_norm": ("all-norm_not_norm-pred_correct-count", "all-norm_not_norm-gold-count"),
        "IoU-pred-need_norm": ("need_norm-norm_not_normXnormalization-pred-count", "normed-norm_not_normUnormalization-pred-count"),
        "IoU-pred-normed": ("normed-norm_not_normXnormalization-pred-count", "need_norm-norm_not_normUnormalization-pred-count")
    }

    return {
           "need_norm-norm_not_normUnormalization-pred-count": need_norm_norm_not_normUnormalization_pred_count,
           "normed-norm_not_normUnormalization-pred-count": normed_norm_not_normUnormalization_pred_count,
           "need_norm-norm_not_normXnormalization-pred-count": need_norm_norm_not_normXnormalization_pred_count,
           "normed-norm_not_normXnormalization-pred-count": normed_norm_not_normXnormalization_pred_count,
           "all-norm_not_norm-pred_correct-count": pred_correct_prediction_count,
           "need_norm-norm_not_norm-pred_correct-count": pred_correct_need_norm_prediction_count,
           "need_norm-norm_not_norm-gold-count": gold_need_norm_count,
           "need_norm-norm_not_norm-pred-count": pred_need_norm_count,
           "all-norm_not_norm-gold-count": total_word
            }, formulas


def score_ls_(ls_pred, ls_gold, score_func, ls_original=None, stat="mean", normalized_mode="all",
              compute_mean_score_per_sent=False, verbose=0):

    assert stat in SUPPORTED_STAT, "ERROR : metric should be in {} ".format(str(SUPPORTED_STAT))
    assert len(ls_gold) == len(ls_pred), "ERROR ls_gold is len {} vs {} : {} while ls_pred is {} ".format(len(ls_gold), len(ls_pred), ls_gold, ls_pred)
    assert score_func in SCORING_FUNC_AVAILABLE
    scores = []
    sent_score_ls = []
    assert normalized_mode in ["NEED_NORM", "NORMED", "all"], \
        'ERROR :normalized_mode should be in  ["NEED_NORM", "NORMED","all"]'

    if normalized_mode in ["NEED_NORM", "NORMED"] :
        assert ls_original is not None, "ERROR : need to provide original sequence to compute NORMED/NEED_NORM analysis"
        assert len(ls_original) == len(ls_gold), "ERROR ls_original sizes provided does not fit pred and gold"
        normalization_ls = [[src_token == gold_token for src_token, gold_token in zip(batch_src, batch_gold)] for batch_src, batch_gold in zip(ls_original, ls_gold)]
        norm_mode = 1 if normalized_mode == "NORMED" else 0
        ls_gold = [[token for token, normed in zip(batch, batch_norm) if normed == norm_mode] for batch, batch_norm in zip(ls_gold, normalization_ls)]
        ls_pred = [[token for token, normed in zip(batch, batch_norm) if normed == norm_mode] for batch, batch_norm in zip(ls_pred, normalization_ls)]

    for ind, (gold_sent, pred_sent) in enumerate(zip(ls_gold, ls_pred)):
        assert len(gold_sent) == len(pred_sent), "len : pred {}, gold {} - pred {} gold {} (normalized_mode is {}) ".format(len(pred_sent), len(gold_sent), pred_sent, gold_sent, normalized_mode)
        sent_score = []
        for word_gold, word_pred in zip(gold_sent, pred_sent):
            eval_func = eval(score_func)
            score_word = eval_func(word_pred, word_gold)
            sent_score.append(score_word)
            scores.append(score_word)
            printing("{} score ,  predicted word {} sentence predicted {} ".format(eval_func(word_pred, word_gold),
                                                                                   word_pred, word_gold), verbose=verbose, verbose_level=5)
        sent_score_ls.append(sent_score)

    if stat == "sum":
        score = np.sum(scores)
    if compute_mean_score_per_sent :
        #get_sent_lengths = [len(sent) for sent in normalization_ls]
        normalized_sent_error_out_of_overall_sent_len = [np.sum(scores_sent)/len(scores_sent) for scores_sent in sent_score_ls]#, get_sent_lengths)]
        n_mode_words_per_sent = np.mean([len(scores_sent) for scores_sent in sent_score_ls])
        n_sents = len(sent_score_ls)
        mean_score_per_sent = np.sum(normalized_sent_error_out_of_overall_sent_len)
    else:
        mean_score_per_sent = None
        n_mode_words_per_sent = None
        n_sents = None

    return {"sum": score, "mean_per_sent": mean_score_per_sent,
            "all-normalize-gold-count": len(scores), "all-normalize-score": score,
            "n_word_per_sent": n_mode_words_per_sent, "n_sents": n_sents}, len(scores)


def score_auxiliary(score_label, score_dic):
    if score_label.endswith("Precision"):
        score_name = "norm_not_norm-Precision"
        precision = score_dic["need_norm-norm_not_norm-pred_correct-count"] / score_dic["need_norm-norm_not_norm-pred-count"] if score_dic["need_norm-norm_not_norm-pred-count"] > 0 else None
        n_tokens_score =  score_dic["need_norm-norm_not_norm-pred-count"]
        score_value = precision
    elif score_label.endswith("Recall"):
        score_name = "norm_not_norm-Recall"
        recall = score_dic["need_norm-norm_not_norm-pred_correct-count"] / score_dic["need_norm-norm_not_norm-gold-count"] if score_dic["need_norm-norm_not_norm-gold-count"] > 0 else None
        score_value = recall
        n_tokens_score = score_dic["need_norm-norm_not_norm-gold-count"]
    elif score_label.endswith("accuracy"):
        score_name = "norm_not_norm-accuracy"
        score_value = score_dic["all-norm_not_norm-pred_correct-count"] / score_dic["all-norm_not_norm-gold-count"] if  score_dic["all-norm_not_norm-gold-count"] else None
        n_tokens_score = score_dic["all-norm_not_norm-gold-count"]
    elif score_label.endswith("F1"):
        score_name = "norm_not_norm-F1"
        n_tokens_score = score_dic["all-norm_not_norm-gold-count"]
        recall = score_dic["need_norm-norm_not_norm-pred_correct-count"]/score_dic["need_norm-norm_not_norm-gold-count"] if score_dic["need_norm-norm_not_norm-gold-count"] > 0 else None
        precision = score_dic["need_norm-norm_not_norm-pred_correct-count"] / score_dic["need_norm-norm_not_norm-pred-count"] if score_dic["need_norm-norm_not_norm-pred-count"] > 0 else None
        score_value = hmean([precision, recall]) if precision is not None and precision>0 and recall is not None and recall > 0 else None
    else:
        return None, None, None
    return score_name, score_value, n_tokens_score


#print(score_ls(["aad"], ["abcccc"], score="edit"))

