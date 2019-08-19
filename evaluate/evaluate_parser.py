"""
evaluating parsing : heads, types and POS assuming gold tokens
"""
from env.importing import re, OrderedDict, os


CONLL_FIELD2INDEX = OrderedDict([("UPOS", 3), ("XPOS", 4), ("UAS", 6), ("LAS", 8)])


def eval_parsing(predicted_path, gold_path, list_score=None):
    """
    evaluating normalization with exact_match only on aligned conll like files gold and prediction
    TODO : - add flexible scoring metric
           - generally : integrate with scoring/ module toolkipt
    :param src_path:
    :param target_path:
    :return:
    """
    print("INFO : scoring predicted tree {} based on gold {} ".format(predicted_path, gold_path))
    if list_score is None:
        list_score = CONLL_FIELD2INDEX.keys()
        print("INFO : no scores passed in list_score so scoring all {} conllu fields".format(list_score ))

    predicted = open(predicted_path, "r")
    gold = open(gold_path, "r")

    scoring_dict = OrderedDict([(score,0) for score in list_score])

    exact_match = 0
    exact_match_flex = 0
    match_need_norm = 0
    match_need_norm_flex = 0
    match_normed_flex = 0
    match_normed = 0
    n_normed = 0
    n_need_norm = 0
    n_tokens = 0

    n_pred_need_norm = 0
    n_pred_normed = 0

    empty_line = 0

    negative_class = 0
    positive_class = 0

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    while True:
        pred_line = predicted.readline()
        gold_line = gold.readline()
        if len(pred_line.strip()) == 0:
            empty_line += 1
            assert len(gold_line.strip()) == 0, "alignement broken on src:{} target:{}".format(src_line, target_line)
            if empty_line > 2:
                break
            else:
                continue
        elif pred_line.startswith("#"):
            empty_line = 0
            assert gold_line.startswith("#"), "alignement broken on src:{} target:{}".format(src_line, target_line)
        else:
            empty_line = 0
            # should 1 _ .. lines
            pred_line = pred_line.strip().split("\t")
            gold_line = gold_line.strip().split("\t")
            if "-" in gold_line[0]:
                continue
            pred_line_form = pred_line[1]
            gold_original_form = gold_line[1]
            gold_original_form = gold_original_form .replace(" ", "")
            # - x_token = 0

            assert pred_line_form == gold_original_form or pred_line_form == gold_original_form.lower(), \
                "alignement broken on pred:{} gold:{}".format(pred_line, gold_line)

            for score in list_score:
                try:
                    pred_norm = pred_line[CONLL_FIELD2INDEX[score]]
                except Exception as e:
                    print("ERROR : field {} score was not find in predicted file line {} ".format(score, pred_line))
                    raise(e)
                try:
                    gold_norm = gold_line[CONLL_FIELD2INDEX[score]]
                except Exception as e:
                    print("ERROR : field {} score was not find in gold file".format(score))
                    raise(e)
                assert pred_norm != "_" and gold_norm != "_", "ERROR : {} or {} found empty".format(pred_norm, gold_norm)
                if score != "LAS":
                    scoring_dict[score] += pred_norm == gold_norm
                else:
                    scoring_dict[score] += (pred_norm == gold_norm)*(gold_line[CONLL_FIELD2INDEX["UAS"]] == pred_line[CONLL_FIELD2INDEX["UAS"]])
            n_tokens += 1

    for score in scoring_dict:
        scoring_dict[score] /= n_tokens

    print("FINAL score : {} tokens were scores : {} ".format(n_tokens, scoring_dict))


if __name__ == "__main__":
    dir = "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/../checkpoints/bert/a30a8-B-84f09-a30a8-B-model_0/predictions/"
    predicted_path = os.path.join(dir, "LAST_ep-prediction-ewt-ud-train-demo-parsing-.conll")
    gold_path = os.path.join(dir, "LAST_ep-gold--ewt-ud-train-demo-parsing-.conll")

    eval_parsing(predicted_path=predicted_path, gold_path=gold_path, list_score=["LAS", "UAS"])

