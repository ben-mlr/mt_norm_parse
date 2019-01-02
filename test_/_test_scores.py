from evaluate.normalization_errors import score_ls, score_ls_


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
    assert score_1/n_words == 1, " ERROR score_1 {} vs 1".format(score_1)
    ls_pred = [["abcdefg", "aaaaaab", "0", "_"], ["abcdefg", "aaaaaab", "0", "_"]]
    score_1, n_words = score_ls_(ls_pred, ls_gold, score="exact", stat="sum")
    assert score_1/n_words == 0.5, " ERROR score_1 {} vs 0.5 ".format(score_1)
    ls_pred = [["abcdefg4", "aaaaaab", "0", "_a"],["abcdefg4", "aaaaaab", "0", "_a"]]
    score_1, n_words = score_ls(ls_pred, ls_gold, score="exact", stat="sum")
    assert score_1 == 0
    print("Test {} function passed".format(func_name))


def _1test_edit_inverse(func_name="_0test_edit_inverse"):
    ls_gold = [["abcdefg", "aaaaaa", "0", "_"], ["abcdefg", "aaaaaa", "0", "_"]]
    ls_pred = [["abcdefg", "aaaaaa", "0", "_"], ["abcdefg", "aaaaaa", "0", "_"]]
    score_1, n_words = score_ls_(ls_pred, ls_gold, score="edit", stat="sum")
    assert score_1/n_words == 1, "ERROR score_1 {} ".format(score_1)
    print("Test {} function passed".format(func_name))


if __name__=="__main__":
    _0test_exact_match()
    _0test_edit_inverse()
    print("all tests score_ls passed ")
    _1test_exact_match()
    _1test_edit_inverse()
    print("all tests score_ls_ passed ")