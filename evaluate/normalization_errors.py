
import numpy as np
from nltk import edit_distance
from io_.info_print import printing
SUPPORTED_STAT = ["sum"]


def exact_match(pred, gold):
    if pred == gold :
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


def score_ls_(ls_pred, ls_gold, score, stat="mean", normalization_ls=None, normalized_mode=None, verbose=0):
    assert stat in SUPPORTED_STAT, "ERROR : metric should be in {} ".format(str(SUPPORTED_STAT))
    assert len(ls_gold) == len(ls_pred), "ERROR ls_gold is len {} while ls_pred is len {} ".format(len(ls_gold),
                                                                                                   len(ls_pred))
    scores = []
    sent_score_ls = []
    if normalization_ls is not None:
        assert normalized_mode in ["NEED_NORM", "NORMED", "all"],'ERROR :normalized_mode should be in  ["NEED_NORM", "NORMED"]'
        norm_mode = 1 if normalized_mode == "NEED_NORM" else 0
        print("Filtering gold and pred ")
        ls_gold = [[token for token, normed in zip(batch, batch_norm) if normed == norm_mode] for batch, batch_norm in zip(ls_gold, normalization_ls)]
        ls_pred = [[token for token, normed in zip(batch, batch_norm) if normed == norm_mode] for batch, batch_norm in zip(ls_pred, normalization_ls)]
    for gold, pred in zip(ls_gold, ls_pred):
        assert len(gold) == len(pred), "len : pred {}, gold {} - pred {} gold {} ".format(len(pred), len(gold), pred, gold)
        sent_score = []
        for sent_gold, sent_pred in zip(gold, pred):
            eval_func = METRIC_DIC[score]
            sent_score.append(eval_func(sent_pred, sent_gold))
            scores.append(eval_func(sent_pred, sent_gold))
            printing("{} score ,  predicted word {} sentence predicted {} ".format(eval_func(sent_pred, sent_gold),sent_pred, sent_gold),
                     verbose=verbose, verbose_level=6)
        sent_score_ls.append(scores)

    # TODO output sentence level score in some way

    if stat == "sum":
        score = np.sum(scores)

    return score, len(scores)

#print(score_ls(["aad"], ["abcccc"], score="edit"))

