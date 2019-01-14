
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


def score_ls_(ls_pred, ls_gold, score, ls_original=None, stat="mean", normalized_mode="all",
              compute_mean_score_per_sent=False, verbose=0):

    assert stat in SUPPORTED_STAT, "ERROR : metric should be in {} ".format(str(SUPPORTED_STAT))
    pdb.set_trace()

    assert len(ls_gold) == len(ls_pred), "ERROR ls_gold is len {} while ls_pred is len {} ".format(ls_gold, ls_pred)

    scores = []
    sent_score_ls = []
    assert normalized_mode in ["NEED_NORM", "NORMED", "all"], \
        'ERROR :normalized_mode should be in  ["NEED_NORM", "NORMED","all"]'

    if normalized_mode in ["NEED_NORM", "NORMED"]:
        assert ls_original is not None, "ERROR : need to provide original sequence to compute NORMED/NEED_NORM analysis"
        assert len(ls_original) == len(ls_gold), "ERROR ls_original sizes provided does not fit pred and gold"
        normalization_ls = [[src_token == gold_token for src_token, gold_token in zip(batch_src, batch_gold)] for batch_src, batch_gold in zip(ls_original, ls_gold)]
        norm_mode = 1 if normalized_mode == "NORMED" else 0
        ls_gold = [[token for token, normed in zip(batch, batch_norm) if normed == norm_mode] for batch, batch_norm in zip(ls_gold, normalization_ls)]
        ls_pred = [[token for token, normed in zip(batch, batch_norm) if normed == norm_mode] for batch, batch_norm in zip(ls_pred, normalization_ls)]
    for ind, (gold_sent, pred_sent) in enumerate(zip(ls_gold, ls_pred)):
        assert len(gold_sent) == len(pred_sent), "len : pred {}, gold {} - pred {} gold {} ".format(len(pred_sent), len(gold_sent), pred_sent, gold_sent)
        #assert len(gold_sent) == sum(normalization_ls[ind]), "len gold and original sent word are not the same len {} and {} pred {} ".format(gold_sent, ls_original, pred_sent)
        sent_score = []
        for word_gold, word_pred in zip(gold_sent, pred_sent):
            eval_func = METRIC_DIC[score]
            score_word = eval_func(word_pred, word_gold)
            sent_score.append(score_word)
            scores.append(score_word)
            printing("{} score ,  predicted word {} sentence predicted {} ".format(eval_func(word_pred, word_gold),
                                                                                   word_pred, word_gold),
                     verbose=verbose, verbose_level=6)
        sent_score_ls.append(sent_score)

    # TODO output sentence level score in some way

    if stat == "sum":
        score = np.sum(scores)
    if compute_mean_score_per_sent :
        #get_sent_lengths = [len(sent) for sent in normalization_ls]
        normalized_sent_error_out_of_overall_sent_len = [np.sum(scores_sent)/len(scores_sent) for scores_sent in sent_score_ls]#, get_sent_lengths)]
        n_mode_words_per_sent = np.mean([len(scores_sent) for scores_sent in sent_score_ls])
        mean_score_per_sent = np.mean(normalized_sent_error_out_of_overall_sent_len)
    else:
        mean_score_per_sent = None
        n_mode_words_per_sent = None

    return {"sum": score, "mean_per_sent": mean_score_per_sent,
            "n_word_per_sent": n_mode_words_per_sent}, len(scores)


#print(score_ls(["aad"], ["abcccc"], score="edit"))

