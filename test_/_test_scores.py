from evaluate.normalization_errors import score_ls, score_ls_
import pdb

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


from evaluate.normalization_errors import correct_pred_counter, score_norm_not_norm


def _test_correct_pred_counter():
    norm_not_norm_gold = np.array([['Watchinqq', 'ayee', 'Moviiie'], ['new', 'pix', 'comming', 'tomoroe'], ['lovin', 'my', 'bg', ':)'], ['C', 'nt', 'fuckn', 'Slp']])
    norm_not_norm_pred = np.array([['Watchinqq', 'ayee', 'Moviiie0'], ['new0', 'pix0', 'comming0', 'tomoroe0'], ['lovin0', 'my0', 'bg0', ':)'], ['0C', '0nt', '0fuckn', '0Slp']])
    norm_not_norm_ori = np.array([['watching', 'ayee', 'movie'], ['new', 'pictures', 'coming', 'tomorrow'], ['loving', 'my', 'background', ':)'], ['can', 'not', 'fucking', 'sleep']])

    ret, _ = correct_pred_counter(ls_pred=norm_not_norm_pred, ls_gold=norm_not_norm_gold, ls_original=norm_not_norm_ori)
    print(ret.keys())
    # testing counting
    assert ret['all-normalization-gold-count'] == 15
    assert ret['all-normalization-pred-count'] == 15
    assert ret['all-normalization-pred_correct-count'] == 3

    assert ret['NEED_NORM-normalization-pred_correct-count'] == 1
    assert ret['NORMED-normalization-pred_correct-count'] == 2
    assert ret['n_sents'] == 4
    assert ret['NEED_NORM-normalization-gold-count'] == 11, "NEED_NORM-normalization-gold-count"
    assert ret['NEED_NORM-normalization-pred-count'] == 13, "NEED_NORM-normalization-pred-count"
    assert ret['NORMED-normalization-gold-count'] == 4, "NORMED-normalization-gold-count"
    assert ret['NORMED-normalization-pred-count'] == 2, "NORMED-normalization-pred-count"

    assert ret['NEED_NORM-normalization-n_word_per_sent-count'] == 11
    assert ret['NORMED-normalization-n_word_per_sent-count'], ret['NORMED-normalization-n_word_per_sent-count'] == 3
    assert ret['all-normalization-n_word_per_sent-count'] == 15, str(ret['all-normalization-n_word_per_sent-count'])+"all-normalization-n_word_per_sent-count"
    assert ret['all-normalization-pred_correct_per_sent-count'] == 2 / 3 + 1 / 4
    assert ret['NEED_NORM-normalization-pred_correct_per_sent-count'] == 1/2,  "NEED_NORM-normalization-pred_correct_per_sent-count"
    assert ret['NORMED-normalization-pred_correct_per_sent-count'] == 1/1+1/2,  "NORMED-normalization-pred_correct_per_sent-count"

    assert ret["NEED_NORM-n_sents"] == 4
    assert ret["all-n_sents"] == 4
    assert ret["NORMED-n_sents"] == 3
    print("Test counter passed ")


def _test_correct_pred_counter_formulas():

    norm_not_norm_gold = np.array([['Watchinqq', 'ayee', 'Moviiie'], ['new', 'pix', 'comming', 'tomoroe'], ['lovin', 'my', 'bg', ':)'],['C', 'nt', 'fuckn', 'Slp']])
    norm_not_norm_pred = np.array([['Watchinqq', 'ayee', 'Moviiie0'], ['new0', 'pix0', 'comming0', 'tomoroe0'], ['lovin0', 'my0', 'bg0', ':)'],['0C', '0nt', '0fuckn', '0Slp']])
    norm_not_norm_ori = np.array([['watching', 'ayee', 'movie'], ['new', 'pictures', 'coming', 'tomorrow'], ['loving', 'my', 'background', ':)'],['can', 'not', 'fucking', 'sleep']])

    ret, formulas = correct_pred_counter(ls_pred=norm_not_norm_pred, ls_gold=norm_not_norm_gold, ls_original=norm_not_norm_ori)
    dic= {}
    for score, val in formulas.items():
        if isinstance(val, tuple) and len(val)>0:
            dic[score] = ret[val[0]]/ret[val[1]]

    # NB : n_sents takes into account if no NEED_NORM and no NORMED
    assert dic['recall-normalization'] == 1/11
    assert abs(dic['precision-normalization'] - 1/13) < 0.00001, dic['precision-normalization']
    assert dic['npv-normalization'] == 2/2
    assert dic['tnr-normalization'] == 2/4
    assert dic['accuracy-normalization'] == 3/15
    assert dic['recall-per_sent-normalization'] == (1/2+0/3+0/2+0/4)/4, dic['recall-per_sent-normalization']
    assert dic['tnr-per_sent-normalization'] == (1/1+0/1+1/2)/3
    assert dic['accuracy-per_sent-normalization'] == (2/3+0/4+1/4+0/4)/4
    print("Test formula passed ")

def _test_correct_norm_no_norm_counter():

    gold_seq = np.array([[0, 1, 0, 2],
                         [1, 0, 0, 0],
                         [0, 1, 0, 1],
                         [0, 0, 0, 0]])
    pred = np.array([[0, 1, 1, 1],
                     [1, 1, 0, 1],
                     [0, 1, 0, 1],
                     [1, 1, 1, 0]])
    #tp =
    #tn =
    #pp =
    #np =

    ret, formulas = score_norm_not_norm(norm_not_norm_pred=pred, norm_not_norm_gold=gold_seq)
    print(ret)

if __name__=="__main__":

    pdb.set_trace = lambda: 1

    if False:
        _0test_exact_match()
        _0test_edit_inverse()
        print("all tests score_ls  0 passed ")
        _1test_exact_match()
        _1test_edit_inverse()
        _1test_exact_match_NORMED_NEED_NORM()
        print("all tests score_ls_ all passed ")
        _test_score_norm_not_norm()

    _test_correct_pred_counter()
    _test_correct_pred_counter_formulas()
    #_test_correct_norm_no_norm_counter()







