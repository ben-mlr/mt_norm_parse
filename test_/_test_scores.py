from evaluate.normalization_errors import score_ls, score_ls_
import pdb
pdb.set_trace = lambda :1
import numpy as np
from evaluate.normalization_errors import score_norm_not_norm
# depreciated
def _0test_exact_match(func_name="_0test_exact_match"):
    ls_gold = ["abcdefg", "aaaaaa", "", "_"]
    ls_pred = ["abcdefg", "aaaaaa", "", "_"]
    score_1, _ = score_ls(ls_pred, ls_gold, score="exact", stat="sum")
    assert score_1 == len(ls_pred), "Score is {} not 1".format(score_1, len(ls_pred))
    ls_pred = ["abcdefg", "aaaaaab", "0", "_"]
    score_1, _ = score_ls(ls_pred, ls_gold,score="exact", stat="sum")
    assert score_1 == 0.5*len(ls_pred), "Score is {} not {} ".format(score_1, 0.5*len(ls_pred))
    ls_pred = ["abcdefg4", "aaaaaab", "0", "_a"]
    score_1, _ = score_ls(ls_pred, ls_gold, score="exact", stat="sum")
    assert score_1 == 0, "Score is {} not 0".format(score_1)
    print("Test {} function passed".format(func_name))


# depreciated
def _0test_edit_inverse(func_name="_0test_edit_inverse"):
    ls_gold = ["abcdefg", "aaaaaa", "0", "_"]
    ls_pred = ["abcdefg", "aaaaaa", "0", "_"]
    score_1, _ = score_ls(ls_pred, ls_gold, score="exact", stat="sum")
    assert score_1 == len(ls_pred)
    print("Test {} function passed".format(func_name))


def _1test_exact_match(func_name="_0test_exact_match"):
    ls_gold = [["abcdefg", "aaaaaa", "", "_"], ["abcdefg", "aaaaaa", "", "_"]]
    ls_pred = [["abcdefg", "aaaaaa", "", "_"], ["abcdefg", "aaaaaa", "", "_"]]
    score_1, n_words = score_ls_(ls_pred, ls_gold, score="exact", stat="sum")
    score_1 = score_1["sum"]
    assert score_1/n_words == 1, " ERROR score_1 {} vs 1".format(score_1)
    ls_pred = [["abcdefg", "aaaaaab", "0", "_"], ["abcdefg", "aaaaaab", "0", "_"]]
    score_1, n_words = score_ls_(ls_pred, ls_gold, score="exact", stat="sum")
    score_1 = score_1["sum"]
    assert score_1/n_words == 0.5, " ERROR score_1 {} vs 0.5 ".format(score_1)
    ls_pred = [["abcdefg4", "aaaaaab", "0", "_a"],["abcdefg4", "aaaaaab", "0", "_a"]]
    score_1, n_words = score_ls_(ls_pred, ls_gold, score="exact", stat="sum")
    score_1 = score_1["sum"]
    assert score_1 == 0
    print("Test {} function passed".format(func_name))


def _1test_exact_match_NORMED_NEED_NORM(func_name="_1test_exact_match_NORMED_NEED_NORM", comments="Evaluating scoring on details NEED_NORM/NORMED"):

    normalized_mode = "NORMED"
    ls_gold = [["abcdefg", "aaaaaa", "", "_"], ["abcdefg", "aaaaaa", "", "_"]]
    ls_pred = [["abcdefg", "aaaaaa", "", "_"], ["abcdefg", "aaaaaa", "", "_"]]
    # testing if original is same as gold
    ls_original = [["abcdefg", "aaaaaa", "", "_"], ["abcdefg", "aaaaaa", "", "_"]]
    score_1, n_words = score_ls_(ls_pred, ls_gold, score="exact", stat="sum", normalized_mode=normalized_mode,
                                 ls_original=ls_original,compute_mean_score_per_sent=True)
    score_1 = score_1["sum"]
    assert score_1/n_words == 1, " ERROR score_1 {} vs 1".format(score_1)
    ls_pred = [["abcdefg", "aaaaaab", "0", "_"], ["abcdefg", "aaaaaab", "0", "_"]]
    score_1, n_words = score_ls_(ls_pred, ls_gold, score="exact", stat="sum", normalized_mode=normalized_mode,
                                 ls_original=ls_original,compute_mean_score_per_sent=True)
    score_1 = score_1["sum"]
    assert score_1/n_words == 0.5, " ERROR score_1 {} vs 0.5 ".format(score_1)
    ls_pred = [["abcdefg4", "aaaaaab", "0", "_a"],["abcdefg4", "aaaaaab", "0", "_a"]]
    score_1, n_words = score_ls_(ls_pred, ls_gold, score="exact", stat="sum", normalized_mode=normalized_mode,
                                 ls_original=ls_original,compute_mean_score_per_sent=True)
    score_1 = score_1["sum"]
    assert score_1 == 0
    # testing if original is different completely of gold
    ls_original = [["abcdefgaaa", "aaaaaabb", "!", ":_"], ["abcdefgaaa", "aaaaaabb", "!", ":_"]]
    score_1, n_words = score_ls_(ls_pred, ls_gold, score="exact", stat="sum", normalized_mode=normalized_mode,
                                 ls_original=ls_original,compute_mean_score_per_sent=True)
    assert n_words == 0, " ERROR n_words should be 0 and is {} ".format(n_words)
    ls_pred = [["abcdefg", "aaaaaab", "0", "_"], ["abcdefg", "aaaaaab", "0", "_"]]
    score_1, n_words = score_ls_(ls_pred, ls_gold, score="exact", stat="sum", normalized_mode=normalized_mode,
                                 compute_mean_score_per_sent=True,
                                 ls_original=ls_original)
    assert n_words == 0, " ERROR n_words should be 0 and is {} ".format(n_words)
    ls_pred = [["abcdefg4", "aaaaaab", "0", "_a"], ["abcdefg4", "aaaaaab", "0", "_a"]]
    score_1, n_words = score_ls_(ls_pred, ls_gold, score="exact", stat="sum", normalized_mode=normalized_mode,
                                 compute_mean_score_per_sent=True,
                                 ls_original=ls_original)

    # testing if 2 words NORMED only 1 NORMED words that have exact match
    ls_pred = [["abcdefg", "aaaaaa0", "", "_"], ["abcdefg", "aaaaaa0", "", "_"]]
    ls_original = [["abcdefg", "aaaaaa", "!", ":_"], ["abcdefg", "aaaaaa", "!", ":_"]]
    ls_gold = [["abcdefg", "aaaaaa", "", "_"], ["abcdefg", "aaaaaa", "", "_"]]
    score_1, n_words = score_ls_(ls_pred, ls_gold, score="exact", stat="sum", normalized_mode=normalized_mode,
                                 ls_original=ls_original,
                                 compute_mean_score_per_sent=True)
    score_1 = score_1["sum"]
    assert score_1/n_words == 0.5, " ERROR score {} vs 0.5 ".format(score_1)

    # testing NEED NORMED
    ls_pred = [["abcdefg", "aaaaaa0", "", "_"], ["abcdefg", "aaaaaa0", "", "_"]]
    score_1, n_words = score_ls_(ls_pred, ls_gold, score="exact", stat="sum", normalized_mode="NEED_NORM",
                                 compute_mean_score_per_sent=True,
                                 ls_original=ls_original)
    score_1 = score_1["sum"]
    assert score_1 / n_words == 1, " ERROR score {} vs 1 ".format(score_1)
    ls_pred = [["abcdefg", "aaaaaa0", "0", "_"], ["abcdefg", "aaaaaa0", "0", "_"]]
    score_1, n_words = score_ls_(ls_pred, ls_gold, score="exact", stat="sum", normalized_mode="NEED_NORM",
                                 compute_mean_score_per_sent=True,
                                 ls_original=ls_original)
    score_1 = score_1["sum"]
    assert score_1 / n_words == 0.5, " ERROR score {} vs 1 ".format(score_1)
    print("Test {} function passed, {} ".format(func_name, comments))


def _1test_edit_inverse(func_name="_0test_edit_inverse"):
    ls_gold = [["abcdefg", "aaaaaa", "0", "_"], ["abcdefg", "aaaaaa", "0", "_"]]
    ls_pred = [["abcdefg", "aaaaaa", "0", "_"], ["abcdefg", "aaaaaa", "0", "_"]]
    score_1, n_words = score_ls_(ls_pred, ls_gold, score="edit", stat="sum")
    score_1 = score_1["sum"]
    assert score_1/n_words == 1, "ERROR score_1 {} ".format(score_1)
    print("Test {} function passed".format(func_name))


def _test_score_norm_not_norm():
    norm_not_norm_gold = np.array([[0, 1, 1, 1, 2],[0, 1, 0, 2, 2]])
    norm_not_norm_pred = np.array([[0, 1, 0, 0, 0],[0, 1, 1, 1, 2]])
    out = score_norm_not_norm(norm_not_norm_pred, norm_not_norm_gold)
    assert out["all-norm_not_norm-pred_correct-count"] == 2+2
    assert out["need_norm-norm_not_norm-pred_correct-count"] == 1+1, \
        " need_norm-norm_not_norm-pred_correct-count is {} while should be 1+1".format(out["need_norm-norm_not_norm-pred_correct-count"])
    assert out["need_norm-norm_not_norm-gold-count"] == 1+2,\
        " need_norm-norm_not_norm-gold-count is {} while it should be 1+2".format(out["need_norm-norm_not_norm-gold-count"])
    assert out["need_norm-norm_not_norm-pred-count"] == 3+1
    assert out["all-norm_not_norm-gold-count"] == 4+3
    print("_test_score_norm_not_norm all test passed for counting score")




if __name__=="__main__":
    _0test_exact_match()
    _0test_edit_inverse()
    print("all tests score_ls  0 passed ")
    _1test_exact_match()
    _1test_edit_inverse()
    _1test_exact_match_NORMED_NEED_NORM()
    print("all tests score_ls_ all passed ")
    _test_score_norm_not_norm()

