import re

from env.importing import hmean, os


def eval_norm(src_path, target_path, count_x_token=True):
    """
    evaluating normalization with exact_match only on aligned conll like files gold and prediction
    TODO : - add flexible scoring metric
           - generally : integrate with scoring/ module toolkipt
    :param src_path:
    :param target_path:
    :return:
    """
    src = open(src_path, "r")
    target = open(target_path, "r")

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
    empty_line = 0
    while True:
        src_line = src.readline()
        target_line = target.readline()
        if len(src_line.strip()) == 0:
            empty_line+=1
            assert len(target_line.strip()) == 0, "alignement broken on src:{} target:{}".format(src_line, target_line)
            if empty_line>2:
                break
            else:
                continue
        elif src_line.startswith("#"):
            empty_line = 0
            assert target_line.startswith("#"), "alignement broken on src:{} target:{}".format(src_line, target_line)
        else:
            empty_line = 0
            # should 1 _ .. lines
            src_line = src_line.strip().split("\t")
            target_line = target_line.strip().split("\t")

            src_original_form = src_line[1]
            target_original_form = target_line[1]
            x_token = 0
            if src_original_form.endswith("X") and count_x_token and src_original_form.replace("X", "") :
                x_token = 1
                print("REPLACING src_original_form {} with {}".format(src_original_form, src_original_form.replace("X",
                                                                                                                   "")))
                src_original_form = src_original_form.replace("X", "")
                target_original_form = target_original_form.replace("X", "")

            assert src_original_form == target_original_form, "alignement broken on src:{} target:{}".format(src_line, target_line)
            pred_norm = re.match("^Norm=([^|]+)|.+", src_line[9]).group(1)
            gold_norm = re.match("^Norm=([^|]+)|.+", target_line[9]).group(1)

            if gold_norm == src_original_form:
                # TODO : add pairs of errors + inconsistencies as a list of pair and count by checking it it's in the lsit of possibilities
                if not x_token :
                    match_normed += pred_norm == gold_norm
                    match_normed_flex += weak_match(pred_norm=pred_norm, gold_norm=gold_norm, src=src_original_form)

                n_normed += 1
            else:
                if not x_token:
                    match_need_norm += pred_norm == gold_norm
                    match_need_norm_flex += weak_match(pred_norm=pred_norm, gold_norm=gold_norm, src=src_original_form)

                n_need_norm += 1
            if src_original_form != pred_norm:
                n_pred_need_norm += 1
            if not x_token:
                exact_match += pred_norm == gold_norm
                exact_match_flex += weak_match(pred_norm=pred_norm, gold_norm=gold_norm, src=src_original_form)
            n_tokens += 1

    accuracy = exact_match/n_tokens
    accuracy_flex = exact_match_flex / n_tokens
    recall = match_need_norm/n_need_norm
    recall_flex = match_need_norm_flex / n_need_norm
    precision = match_need_norm/n_pred_need_norm
    precision_flex = match_need_norm_flex/n_pred_need_norm
    f1 = hmean([recall, precision]) if recall > 0 and precision > 0 else None
    f1_flex = hmean([recall_flex, precision_flex]) if recall_flex > 0 and precision_flex > 0 else None

    print("ACCURACY {:0.2f} , RECALL:{:0.2f} , PRECISION:{:0.2f}, F1:{:0.2f} / {} tokens {} need Norm".format(accuracy*100, recall*100,
                                                                                      precision*100, f1*100, n_tokens, n_need_norm))
    print("FLEX ACCURACY {:0.2f} , RECALL:{:0.2f} , PRECISION:{:0.2f}, F1:{:0.2f} / {} tokens {} need Norm".format(
        accuracy_flex * 100, recall_flex * 100,
        precision_flex * 100, f1_flex * 100, n_tokens, n_need_norm))

# defining possible normalization (independent of annotation mistake of TEST)
## - we remove mistake form test
## - we add inconsistencies of DEV/TEST


REF_DIC = {"about": ["about"],
           "bout": ["about"],
           "with": ["with"],
           "you": ["you"],
           "are": ["are"],
           "and": ["and"],
           "babe": ["babe", "baby"],
           "television": ["television", "tv"],
           "family": ["family", "fam"],
           "sister": ["sister"],
           "brother": ["brother"],
           "&": ["and", "&"],
           "congrats": ["congrats", "congratulations"],
           "niggas": ["niggas", "niggers"],
           "yes": ["yes", "ya"]
           }


def weak_match(pred_norm, gold_norm,src, ref_dic=REF_DIC):
    if gold_norm in ref_dic:
        print("{} WEAK MATCH for pred [{}] vs [gold:{}] [src {}]: {} while hard is {}".format(int(pred_norm in ref_dic[gold_norm]) and not int(pred_norm == gold_norm),
              pred_norm, gold_norm, src,int(pred_norm in ref_dic[gold_norm]), int(pred_norm == gold_norm)))
        return int(pred_norm in ref_dic[gold_norm])
    else:
        return pred_norm == gold_norm

dir = "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/checkpoints/bert"

model = "9320927-B-ed1e8-9320927-B-model_0"
# 1 to n pb
model = "9337555-B-a387d-9337555-B-model_5"
# lexnorm fine
model = "9337555-B-57db3-9337555-B-model_2"
#src = os.path.join(dir, model, "predictions", "LAST_ep-prediction-lex_norm2015_test.conll")
#target = os.path.join(dir, model, "predictions", "LAST_ep-gold.conll-lex_norm2015_test")
src = os.path.join(dir, model, "predictions", "LAST_ep-gold.conll-lexnorm-normalize-")
target = os.path.join(dir, model, "predictions", "LAST_ep-prediction-lexnorm-normalize-.conll")


eval_norm(src, target, count_x_token=False)

# ls  ["lol", "idk", "lmfao", "lmao", "tbh", "asap", "omg", "omfg", "wtf", "im", "ima", "youre", "dont", "doesnt","wasnt","cant", "didnt", "its"]