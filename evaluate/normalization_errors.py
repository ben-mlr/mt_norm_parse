
import numpy as np
from nltk import edit_distance

def exact_match(pred, gold):
    if pred == gold :
        return 1
    else:
        return 0

def edit_inverse(pred, gold):
    edit = edit_distance(pred, gold)
    return 1-edit/max(len(pred), len(gold))

METRIC_DIC = {"exact": exact_match, "edit": edit_inverse}


def score_ls(ls_pred, ls_gold, score, metric="mean"):

    assert len(ls_gold) == len(ls_pred), "ERROR ls_gold is len {} while ls_pred is len {} ".format(len(ls_gold), len(ls_pred))
    scores = []

    for gold, pred in zip(ls_gold, ls_pred):
        eval_func = METRIC_DIC[score]
        scores.append(eval_func(pred, gold))

    if metric == "mean":
        score = np.mean(scores)

    return score, len(ls_gold)


#print(score_ls(["aad"], ["abcccc"], score="edit"))

