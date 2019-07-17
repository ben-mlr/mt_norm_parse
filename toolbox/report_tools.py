from env.importing import *
from io_.info_print import printing


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


def write_args(dir, model_id, checkpoint_dir=None,
               hyperparameters=None,
               info_checkpoint=None, verbose=1):

    args_dir = os.path.join(dir, "{}-args.json".format(model_id))
    if os.path.isfile(args_dir):
        info = "updated"
        args = json.load(open(args_dir, "r"))
        args["checkpoint_dir"] = checkpoint_dir
        args["info_checkpoint"] = info_checkpoint
        json.dump(args, open(args_dir, "w"))
    else:
        assert hyperparameters is not None, "REPORT : args.json created for the first time : hyperparameters dic required "
        assert info_checkpoint is None, "REPORT : args. created for the first time : no checkpoint yet "
        info = "new"
        json.dump(OrderedDict([("checkpoint_dir", checkpoint_dir),
                               ("hyperparameters", hyperparameters),
                               ("info_checkpoint", None)]), open(args_dir, "w"))
    printing("MODEL args.json {} written {} ".format(info, args_dir), verbose_level=1, verbose=verbose)
    return args_dir



def get_hyperparameters_dict(args, case, random_iterator_train, seed, verbose):

    hyperparameters = OrderedDict([("bert_model", args.bert_model), ("lr", args.lr),
                                   ("n_epochs", args.epochs),
                                   ("initialize_bpe_layer", args.initialize_bpe_layer),
                                   ("fine_tuning_strategy", args.fine_tuning_strategy),
                                   ("dropout_input_bpe", args.dropout_input_bpe),
                                   ("heuristic_ls", args.heuristic_ls),
                                   ("gold_error_detection", args.gold_error_detection),
                                   ("dropout_classifier",
                                    args.dropout_classifier if args.dropout_classifier is not None else "UNK"),
                                   ("dropout_bert", args.dropout_bert if args.dropout_bert is not None else "UNK"),
                                   ("tasks", args.tasks),
                                   ("masking_strategy", args.masking_strategy),
                                   ("portion_mask", args.portion_mask),
                                   ("checkpoint_dir", args.checkpoint_dir if args.checkpoint_dir is not None else None),
                                   ("norm_2_noise_training", args.norm_2_noise_training),
                                   ("random_iterator_train", random_iterator_train),
                                   ("aggregating_bert_layer_mode", args.aggregating_bert_layer_mode),
                                   ("tokenize_and_bpe", args.tokenize_and_bpe),
                                   ("SEED", seed), ("case", case), ("bert_module", args.bert_module),
                                   ("freeze_layer_prefix_ls", args.freeze_parameters),
                                   ("layer_wise_attention", args.layer_wise_attention),
                                   ("append_n_mask", args.append_n_mask),
                                   ("multi_task_loss_ponderation", args.multi_task_loss_ponderation)
                                   ])
    printing("HYPERPARAMETERS {} ", var=[hyperparameters], verbose=verbose, verbose_level=1)
    return hyperparameters