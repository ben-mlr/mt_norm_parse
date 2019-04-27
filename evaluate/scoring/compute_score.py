

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
