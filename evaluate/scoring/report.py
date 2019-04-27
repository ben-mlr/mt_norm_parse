from evaluate.scoring.compute_score import word_level_scoring
from evaluate.scoring.aggregate_score import agg_func_batch_score


def overall_word_level_metric_measure(gold_sent_ls,
                                      pred_sent_ls_topk, topk,
                                      metric="exact_match",
                                      agg_func_ls=["sum"]):
    """
    'metric' based on a word level comparison of (pred,gold) : e.g : exact_match , edit
    'agg_func' based on a aggregation func to get the overall batch score : e.g : sum
    :param metric:
    :param agg_func:
    :return batch : score, number of token measured
    """
    assert isinstance(agg_func_ls, list)
    assert len(pred_sent_ls_topk) == topk, "ERROR topk not consistent with prediction list " \
        .format(len(pred_sent_ls_topk), topk)
    overall_score_ls_sent = []
    skipping_sent = 0
    for gold_ind_sent, gold_sent in enumerate(gold_sent_ls):
        # TODO test for all topk
        try:
            assert len(gold_sent) == len(pred_sent_ls_topk[0][gold_ind_sent])
        except Exception as e:
            print(e)
            skipping_sent += len(gold_sent_ls)
            overall_score_ls_sent = [[0]]
            pdb.set_trace()
            break
        score_sent = []
        for ind_word in range(len(gold_sent)):
            gold_token = gold_sent[ind_word]
            topk_word_pred = [pred_sent_ls_topk[top][gold_ind_sent][ind_word] for top in range(topk)]
            score_sent.append(word_level_scoring(metric=metric, gold=gold_token, topk_pred=topk_word_pred, topk=topk))
        overall_score_ls_sent.append(score_sent)

    result = []
    for agg_func in agg_func_ls:
        result.append({"score": agg_func_batch_score(overall_ls_sent_score=overall_score_ls_sent, agg_func=agg_func),
                       "agg_func": agg_func,
                       "metric": "exact_match",
                       "n_tokens": agg_func_batch_score(overall_ls_sent_score=overall_score_ls_sent,
                                                        agg_func="n_tokens"),
                       "n_sents": agg_func_batch_score(overall_ls_sent_score=overall_score_ls_sent,
                                                       agg_func="n_sents")})
    return result, skipping_sent
