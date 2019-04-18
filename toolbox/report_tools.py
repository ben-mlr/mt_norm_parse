from env.importing import *


def get_score(scores, metric, info_score, task, data):
    for report in scores:
        if report["metric"] == metric and report["info_score"] == info_score and report["task"] == task and report[
            "data"] == data:
            return report
    raise (Exception(
        "REPORT with {} metric {} info_score {} task and {} data not found in {} ".format(metric, info_score, task, data,
                                                                                         scores)))

def checkout_layer_name(name_param, model_parameters, info_epoch=""):
    for name, param in model_parameters:
        if param.requires_grad:
            if name == name_param:
                print("DEBUG END REPLICATION:epoch (checkout_layers_tools) {} ".format(info_epoch), "name", name, param.data)


def pred_word_to_list(pred_word, special_symb_ls):
    index_special_ls = []

    pred_word = [pred_word]
    ind_pred_word = 0
    counter = 0
    while True:
        counter += 1
        index_special_ls = []
        _pred_word = pred_word[ind_pred_word]
        # Looking for all special character (we only look at the first one found)
        for special_symb in special_symb_ls:
            index_special_ls.append(_pred_word.find(special_symb))
        indexes = np.argsort(index_special_ls)
        index_special_char=-1
        # Getting the index and the character of the first special character if nothing we get -1
        for ind, a in enumerate(indexes):
            if index_special_ls[a] >= 0:
                special_symb = special_symb_ls[a]
                index_special_char = index_special_ls[a]
                break
            if ind > len(indexes):
                index_special_char = -1
                special_symb = ""
                break
        # if found a special character
        if (index_special_char) >= 0:
            starting_seq = [_pred_word[:index_special_char]] if index_special_char> 0 else []
            middle = [_pred_word[index_special_char:index_special_char + len(special_symb) ]]
            end_seq = [_pred_word[index_special_char + len(special_symb):]]
            if len(end_seq[0].strip()) == 0:
                end_seq = []
            _pred_word_ls = starting_seq + middle +end_seq
            pred_word[ind_pred_word] = _pred_word_ls[0]
            if len(_pred_word_ls) > 0:
                pred_word.extend(_pred_word_ls[1:])
            ind_pred_word += 1
            pdb.set_trace()
            if len(starting_seq) > 0:
                ind_pred_word += 1
        else:
            ind_pred_word += 1
        pdb.set_trace()
        if ind_pred_word >= len(pred_word):
            break

    new_word = []
    # transform the way we splitted in list of characters (including special ones)
    for word in pred_word:
        if word in special_symb_ls:
            new_word.append(word)
        else:
            new_word.extend(list(word))

    return new_word