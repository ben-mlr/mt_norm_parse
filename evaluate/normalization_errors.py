
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
    return 1-edit/max(len(pred), len(gold))

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


def score_ls_(ls_pred, ls_gold, score, stat="mean", verbose=0):
    assert stat in SUPPORTED_STAT, "ERROR : metric should be in {} ".format(str(SUPPORTED_STAT))
    assert len(ls_gold) == len(ls_pred), "ERROR ls_gold is len {} while ls_pred is len {} ".format(len(ls_gold),
                                                                                                   len(ls_pred))

    scores = []
    sent_score_ls = []
    for gold, pred in zip(ls_gold, ls_pred):
        assert len(gold)==len(pred), "pred {} gold {} ".format(pred, gold)
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

