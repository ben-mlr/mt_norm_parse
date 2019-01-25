from scipy.stats import hmean
import numpy as np
from nltk import edit_distance
from io_.info_print import printing
import pdb
from env.project_variables import SUPPORTED_STAT


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


METRIC_DIC = {"exact": exact_match, "edit": edit_inverse}


def score_ls(ls_pred, ls_gold, score, stat="mean", verbose=0):
    assert stat in SUPPORTED_STAT , "ERROR : metric should be in {} ".format(str(SUPPORTED_STAT))
    assert len(ls_gold) == len(ls_pred), "ERROR ls_gold is len {} while ls_pred is len {} ".format(len(ls_gold), len(ls_pred))

    scores = []
    for gold, pred in zip(ls_gold, ls_pred):
        eval_func = METRIC_DIC[score]
        scores.append(eval_func(pred, gold))

    if stat == "sum":
        score = np.sum(scores)

    return score, len(scores)

from collections import OrderedDict
def score_ls_2(ls_pred, ls_gold, ls_original=None,
               verbose=0):
    stat = "sum"
    dic = OrderedDict()

    assert len(ls_gold) == len(ls_pred), "ERROR ls_gold is len {} vs {} : {} while ls_pred is {} ".format(len(ls_gold), len(ls_pred), ls_gold, ls_pred)
    normalization_ls = [[src_token == gold_token for src_token, gold_token in zip(batch_src, batch_gold)] for
                        batch_src, batch_gold in zip(ls_original, ls_gold)]
    for normalized_mode in ["all","NEED_NORM", "NORMED"]:
        scores = []
        sent_score_ls = []
        assert ls_original is not None, "ERROR : need to provide original sequence to compute NORMED/NEED_NORM analysis"
        assert len(ls_original) == len(ls_gold), "ERROR ls_original sizes provided does not fit pred and gold"
        if normalized_mode in ["NEED_NORM", "NORMED"]:
            norm_mode = 1 if normalized_mode == "NORMED" else 0
            _ls_gold = [[token for token, normed in zip(batch, batch_norm) if normed == norm_mode] for batch, batch_norm in zip(ls_gold, normalization_ls)]
            _ls_pred = [[token for token, normed in zip(batch, batch_norm) if normed == norm_mode] for batch, batch_norm in zip(ls_pred, normalization_ls)]
        else:
            _ls_gold = ls_gold
            _ls_pred = ls_pred
        for ind, (gold_sent, pred_sent) in enumerate(zip(_ls_gold, _ls_pred)):
            assert len(gold_sent) == len(pred_sent), "len : pred {}, gold {} - pred {} gold {} (normalized_mode is {})".format(len(pred_sent), len(gold_sent), pred_sent, gold_sent, normalized_mode)
            sent_score = []
            for word_gold, word_pred in zip(gold_sent, pred_sent):
                eval_func = METRIC_DIC["exact"]
                score_word = eval_func(word_pred, word_gold)
                sent_score.append(score_word)
                scores.append(score_word)
                printing("{} score ,  predicted word {} sentence predicted {} ".format(eval_func(word_pred, word_gold), word_pred, word_gold),
                         verbose=verbose, verbose_level=6)
            sent_score_ls.append(sent_score)
        score = np.sum(scores)
        n_mode_words_per_sent = np.mean([len(scores_sent) for scores_sent in sent_score_ls])
        n_sents = len(sent_score_ls)
        normalized_sent_error_out_of_overall_sent_len = [np.sum(scores_sent)/len(scores_sent) for scores_sent in sent_score_ls]#, get_sent_lengths)]
        mean_score_per_sent = np.sum(normalized_sent_error_out_of_overall_sent_len)

        return_dic = {normalized_mode+"-normalization-pred_correct-count": score,
                      normalized_mode+"-normalization-n_word_per_sent-count": n_mode_words_per_sent,
                      normalized_mode+"-normalization-pred_correct_per_sent-count": mean_score_per_sent,
                      normalized_mode+"-normalization-gold-count": len(scores)}
        dic.update(return_dic)
    dic["n_sents"] = n_sents
    return dic


def score_norm_not_norm(norm_not_norm_pred, norm_not_norm_gold):
    predicted_not_pad = norm_not_norm_pred[norm_not_norm_gold != 2]
    gold_not_pad = norm_not_norm_gold[norm_not_norm_gold != 2]
    total_word = len(gold_not_pad)
    assert len(gold_not_pad) == len(predicted_not_pad)

    pred_correct_need_norm_prediction_count = np.sum(np.array(gold_not_pad == predicted_not_pad)[np.array(gold_not_pad)==0])
    pred_correct_prediction_count = np.sum(np.array(gold_not_pad == predicted_not_pad))
    #print(gold_not_pad[gold_not_pad == 0])#,dtype=int))
    gold_need_norm_count = len(np.array(gold_not_pad[gold_not_pad == 0]))
    pred_need_norm_count = len(np.array(predicted_not_pad[predicted_not_pad == 0]))
    pdb.set_trace()
    return {
            "all-norm_not_norm-pred_correct-count": pred_correct_prediction_count,
            "need_norm-norm_not_norm-pred_correct-count": pred_correct_need_norm_prediction_count,
            "need_norm-norm_not_norm-gold-count": gold_need_norm_count,
            "need_norm-norm_not_norm-pred-count": pred_need_norm_count,
            "all-norm_not_norm-gold-count": total_word
            }


def score_ls_(ls_pred, ls_gold, score, ls_original=None, stat="mean", normalized_mode="all",
              compute_mean_score_per_sent=False, verbose=0):

    assert stat in SUPPORTED_STAT, "ERROR : metric should be in {} ".format(str(SUPPORTED_STAT))
    assert len(ls_gold) == len(ls_pred), "ERROR ls_gold is len {} vs {} : {} while ls_pred is {} ".format(len(ls_gold), len(ls_pred), ls_gold, ls_pred)

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
        assert len(gold_sent) == len(pred_sent), "len : pred {}, gold {} - pred {} gold {} (normalized_mode is {})".format(len(pred_sent), len(gold_sent), pred_sent, gold_sent, normalized_mode)
        #assert len(gold_sent) == sum(normalization_ls[ind]), "len gold and original sent word are not the same len {} and {} pred {} ".format(gold_sent, ls_original, pred_sent)
        sent_score = []
        for word_gold, word_pred in zip(gold_sent, pred_sent):
            eval_func = METRIC_DIC[score]
            score_word = eval_func(word_pred, word_gold)
            sent_score.append(score_word)
            scores.append(score_word)
            printing("{} score ,  predicted word {} sentence predicted {} ".format(eval_func(word_pred, word_gold),
                                                                                   word_pred, word_gold), verbose=verbose, verbose_level=6)
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
            "all-normalization-gold-count": len(scores), "all-normalization-score": score,
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
        score_value = score_dic["all-norm_not_norm-pred_correct-count"] / score_dic["all-norm_not_norm-gold-count"]
        n_tokens_score = score_dic["all-norm_not_norm-gold-count"]
    elif score_label.endswith("F1"):
        score_name = "norm_not_norm-F1"
        pdb.set_trace()
        n_tokens_score = score_dic["all-norm_not_norm-gold-count"]
        recall = score_dic["need_norm-norm_not_norm-pred_correct-count"]/score_dic["need_norm-norm_not_norm-gold-count"] if score_dic["need_norm-norm_not_norm-gold-count"] > 0 else None
        precision = score_dic["need_norm-norm_not_norm-pred_correct-count"] / score_dic["need_norm-norm_not_norm-pred-count"] if score_dic["need_norm-norm_not_norm-pred-count"] > 0 else None
        score_value = hmean([precision, recall]) if precision is not None and precision>0 and recall is not None and recall > 0 else None
    else:
        return None, None, None
    return score_name, score_value, n_tokens_score


#print(score_ls(["aad"], ["abcccc"], score="edit"))

