
import numpy as np
from nltk import edit_distance
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


def score_ls(ls_pred, ls_gold, score, stat="mean"):
    assert stat in SUPPORTED_STAT , "ERROR : metric should be in {} ".format(str(SUPPORTED_STAT))
    assert len(ls_gold) == len(ls_pred), "ERROR ls_gold is len {} while ls_pred is len {} ".format(len(ls_gold), len(ls_pred))
    scores = []
    for gold, pred in zip(ls_gold, ls_pred):
        eval_func = METRIC_DIC[score]
        scores.append(eval_func(pred, gold))

    if stat == "sum":
        score = np.sum(scores)

    return score, len(scores)


#print(score_ls(["aad"], ["abcccc"], score="edit"))

