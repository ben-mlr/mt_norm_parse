from evaluate.normalization_errors import score_ls


def _0test_exact_match(func_name="_0test_exact_match"):
    ls_gold = ["abcdefg", "aaaaaa", "", "_"]
    ls_pred = ["abcdefg", "aaaaaa", "", "_"]
    score_1, _ = score_ls(ls_pred, ls_gold, "exact", metric="mean")
    assert score_1 == 1
    ls_pred = ["abcdefg", "aaaaaab", "0", "_"]
    score_1, _ = score_ls(ls_pred, ls_gold, "exact", metric="mean")
    assert score_1 == 0.5
    ls_pred = ["abcdefg4", "aaaaaab", "0", "_a"]
    score_1, _ = score_ls(ls_pred, ls_gold, "exact", metric="mean")
    assert score_1 == 0
    print("Test {} function passed".format(func_name))


def _0test_edit_inverse(func_name="_0test_edit_inverse"):
    ls_gold = ["abcdefg", "aaaaaa", "0", "_"]
    ls_pred = ["abcdefg", "aaaaaa", "0", "_"]
    score_1, _ = score_ls(ls_pred, ls_gold, "edit", metric="mean")
    assert score_1 == 1
    print("Test {} function passed".format(func_name))


if __name__=="__main__":
    _0test_exact_match()
    _0test_edit_inverse()
    print("all test passed")