



def agg_func_batch_score(overall_ls_sent_score, agg_func):

    sum_ = sum([score for score_ls in overall_ls_sent_score for score in score_ls])
    n_tokens = sum([1 for score_ls in overall_ls_sent_score for _ in score_ls])
    n_sents = len(overall_ls_sent_score)

    if agg_func == "sum":
        return sum_
    elif agg_func == "n_tokens":
        return n_tokens
    elif agg_func == "n_sents":
        return n_sents
    elif agg_func == "mean":
        return sum_/n_tokens
    elif agg_func == "sum_mean_per_sent":
        sum_per_sent = [sum(score_ls) for score_ls in overall_ls_sent_score]
        token_per_sent = [len(score_ls) for score_ls in overall_ls_sent_score]
        sum_mean_per_sent_score = sum([sum_/token_len for sum_, token_len in zip(sum_per_sent, token_per_sent)])
        return sum_mean_per_sent_score
    else:
        raise(Exception("agg_func: {} not supported".format(agg_func)))

