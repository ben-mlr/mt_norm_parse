from env.importing import pdb
from io_.dat.constants import SPECIAL_TOKEN_LS

def word_level_scoring(metric, gold, topk_pred, topk):
    """
    compare a gold string and a list of candidate
    return a score based on it
    only exact_match supported so far
    (originally designed for bert eval)
    :param metric:
    :param gold:
    :param topk_pred:
    :param topk:
    :return:
    """
    assert metric in ["exact_match"], "metric is {} ".format(metric)
    if topk > 1:
        assert metric == "exact_match", "ERROR : only exact_match allows for looking into topk prediction "
    assert len(topk_pred) == topk, "ERROR : inconsinstent provided topk and what I got "
    if metric == "exact_match":
        for pred in topk_pred:
            if gold == pred:
                return 1
        return 0


def word_level_filter(gold, topk_pred, topk, src, sample="all", word_reference_dic_ls=None):
    """
    compare a gold string and a list of candidate
    return a score based on it
    only exact_match supported so far
    (originally designed for bert eval)
    :param metric:
    :param gold:
    :param topk_pred:
    :param topk:
    :return:
    """
    #assert sample in ["all", "NORMED", "NEED_NORM"]
    assert len(topk_pred) == topk, "ERROR : inconsinstent provided topk and what I got "

    if gold in SPECIAL_TOKEN_LS:
        return 0
    if sample == "all":
        return 1
    elif sample == "NORMED":
        return src == gold
    elif sample == "NEED_NORM":
        return src != gold
    elif sample == "InV":
        assert word_reference_dic_ls is not None, "No word_reference_dic_ls provided"
        assert word_reference_dic_ls.get("InV", None) is not None, "No word_reference_dic_ls['InV'] provided"
        return src in word_reference_dic_ls["InV"] or src.lower() in word_reference_dic_ls["InV"]
    elif sample == "OOV":
        assert word_reference_dic_ls is not None, "No word_reference_dic_ls provided"
        assert word_reference_dic_ls.get("InV", None) is not None, "No word_reference_dic_ls['InV'] provided"
        return src not in word_reference_dic_ls["InV"] and src.lower() not in word_reference_dic_ls["InV"]
