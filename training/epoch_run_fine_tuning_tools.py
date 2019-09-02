from env.importing import torch, pdb, re
from env.tasks_settings import TASKS_PARAMETER
from io_.dat.constants import SPECIAL_TOKEN_LS, PAD_ID_BERT
from io_.printout_iterator_as_raw import printing
from io_.dat.normalized_writer import write_conll, write_conll_multitask


def get_casing(case, batch, task_normalize_is):
    if case is not None:
        if case == "lower":
            batch.raw_input = [[word.lower() if word not in SPECIAL_TOKEN_LS else word for word in sent] for sent in batch.raw_input]
            if task_normalize_is:
                batch.raw_output = [[word.lower() if word not in SPECIAL_TOKEN_LS else word for word in sent] for sent in batch.raw_output]
    return batch


def logging_processing_data(_verbose, verbose, verbose_level, batch_raw_input, input_tokens_tensor, batch_raw_output, output_tokens_tensor, inp_bpe_tokenized, out_bpe_tokenized):
    printing("DATA : pre-tokenized input {} ", var=[batch_raw_input], verbose_level=verbose_level, verbose=_verbose)
    printing("DATA : BPEtokenized input ids {}", var=[input_tokens_tensor], verbose_level=3, verbose=verbose)

    printing("DATA : pre-tokenized output {} ", var=[batch_raw_output], verbose_level=verbose_level, verbose=_verbose)
    printing("DATA : BPE tokenized output ids  {}", var=[output_tokens_tensor], verbose_level=4, verbose=verbose)
    # BPE
    printing("DATA : BPE tokenized input  {}", var=[inp_bpe_tokenized], verbose_level=4, verbose=_verbose)
    printing("DATA : BPE tokenized output  {}", var=[out_bpe_tokenized], verbose_level=4, verbose=_verbose)


def logging_scores(perf_prediction, iter, batch_i, pred_detokenized_topk, verbose):
    printing("TRAINING : Score : {} / {} tokens / {} sentences", var=[
        perf_prediction["sum"]["all"]["score"],
        perf_prediction["sum"]["all"]["n_tokens"],
        perf_prediction["sum"]["all"]["n_sents"]
    ],
             verbose=verbose, verbose_level=1)
    printing("TRAINING : eval gold {}-{} {}", var=[iter, batch_i, gold_detokenized],
             verbose=verbose,
             verbose_level=2)
    printing("TRAINING : eval pred {}-{} {}", var=[iter, batch_i, pred_detokenized_topk],
             verbose=verbose,
             verbose_level=2)
    printing("TRAINING : eval src {}-{} {}", var=[iter, batch_i, src_detokenized],
             verbose=verbose, verbose_level=1)
    printing("TRAINING : BPE eval gold {}-{} {}", var=[iter, batch_i, gold],
             verbose=verbose,
             verbose_level=2)
    printing("TRAINING : BPE eval pred {}-{} {}", var=[iter, batch_i, sent_ls_top],
             verbose=verbose,
             verbose_level=2)
    printing("TRAINING : BPE eval src {}-{} {}", var=[iter, batch_i, source_preprocessed],
             verbose=verbose, verbose_level=2)
    printing("TRAINING : BPE eval src {}-{} {}", var=[iter, batch_i, input_alignement_with_raw],
             verbose=verbose, verbose_level=2)


def print_align_bpe(source_preprocessed, gold, input_alignement_with_raw, labels_n_mask_prediction,
                    verbose, verbose_level):
    if labels_n_mask_prediction is None:
        labels_n_mask_prediction = [[None for _ in range(len(sent))] for sent in
                                    input_alignement_with_raw]
    if isinstance(verbose, int) or verbose == "alignement":
        if verbose == "alignement" or verbose >= verbose_level:
            assert len(source_preprocessed) == len(gold), ""
            assert len(input_alignement_with_raw) == len(gold), ""
            for sent_src, sent_gold, index_match_with_src, append_masks in zip(source_preprocessed,
                                                                               gold,
                                                                               input_alignement_with_raw,
                                                                               labels_n_mask_prediction):
                assert len(sent_src) == len(sent_gold)
                assert len(sent_src) == len(sent_gold)
                for src, gold_tok, index, masks in zip(sent_src, sent_gold, index_match_with_src,
                                                       append_masks):
                    printing("{}:{} --> {} (n_masks {})", var=[index, src, gold_tok, masks],
                             verbose=1, verbose_level=1)


def log_warning(counting_failure_parralel_bpe_batch, data_label, batch_i, batch, noisy_under_splitted,
                skipping_batch_n_to_1, aligned, noisy_over_splitted, skip_1_t_n, skipping_evaluated_batch, verbose):
    printing("WARNING {} aignement failure caused by parallel ", var=[counting_failure_parralel_bpe_batch],
             verbose=verbose, verbose_level=1)
    printing(
        "WARNING on {} : Out of {} batch of {} sentences each {} skipped ({} batch aligned ; {} with at least 1 sentence noisy MORE SPLITTED ; {} with  LESS SPLITTED {} + SENT with skipped_1_to_n : {}) ",
        var=[data_label, batch_i, batch.input_seq.size(0), noisy_under_splitted + skipping_batch_n_to_1, aligned,
             noisy_over_splitted, noisy_under_splitted, "SKIPPED" if skip_1_t_n else "", skipping_batch_n_to_1],
        verbose=verbose, verbose_level=0)
    printing("WARNING on {} ON THE EVALUATION SIDE we skipped extra {} batch ",
             var=[data_label, skipping_evaluated_batch], verbose_level=1, verbose=1)


def tensorboard_loss_writer_batch_level_multi(writer, mode, model_id, _loss, batch_i, iter, loss_dic, tasks):

    writer.add_scalars("loss-batch-sum",
                       {"loss-{}-{}-bpe".format(mode, model_id): _loss.clone().cpu().data.numpy()
                       if not isinstance(_loss, int) else 0},
                       iter+batch_i)
    for label in loss_dic:
        writer.add_scalars("loss-batch-{}".format(label),
                           {"loss-{}-{}-bpe".format(mode, model_id): loss_dic[label].detach().clone().cpu().data.numpy()},
                           iter + batch_i)


def tensorboard_loss_writer_batch_level(writer, mode, model_id, _loss, batch_i, iter, loss_dic,task_normalize_is,  append_n_mask, task_pos_is):
    writer.add_scalars("loss-batch-sum",
                       {"loss-{}-{}-bpe".format(mode, model_id): _loss.clone().cpu().data.numpy()
                       if not isinstance(_loss, int) else 0},
                       iter+batch_i)
    if task_pos_is:
        writer.add_scalars("loss-batch-pos",
                           {"loss-{}-{}-bpe".format(mode, model_id): loss_dic["loss_task_2"].detach().clone().cpu().data.numpy()
                           },
                           iter + batch_i)
    if task_normalize_is:
        writer.add_scalars("loss-batch-norm",
                           {"loss-{}-{}-bpe".format(mode, model_id):
                                loss_dic["loss_task_1"].detach().clone().cpu().data.numpy()
                            },
                           iter + batch_i)
        if append_n_mask:

            writer.add_scalars("loss-batch-norm-pred_n_mask",
                               {"loss-{}-{}-pred_n_mask".format(mode, model_id):
                                    loss_dic["loss_task_n_mask_prediction"].detach().clone().cpu().data.numpy()
                                },
                               iter + batch_i)


def update_loss_dic_average(loss_dic_current, loss_dic_total):

    assert set(loss_dic_current.keys()).issubset(set(loss_dic_total.keys())), "ERROR : mismatch keys {} and {} ".format(loss_dic_current, loss_dic_total)

    for loss_label, value in loss_dic_current.items():
        loss_dic_total[loss_label] += value.item()

    return loss_dic_total


def tensorboard_loss_writer_epoch_level_multi(writer, mode, model_id, epoch, loss_dic, n_tokens_dic, data_label):
    """
    NB : loss provided is already supposed to be average per batch
    :param writer:
    :param tasks:
    :param mode:
    :param model_id:
    :param epoch:
    :param n_batch_norm:
    :param n_batch_pos:
    :param append_n_mask:
    :param loss:
    :param loss_norm:
    :param loss_pos:
    :param loss_n_mask_prediction:
    :param batch_i:
    :return:
    """
    try:
        assert set(loss_dic.keys()) == set(n_tokens_dic.keys()), \
        "ERROR keys mismatching between loss and n_tokens {} {}".format(loss_dic, n_tokens_dic)
    except Exception as e:
        print(e)
    for loss_lab, loss_val in loss_dic.items():
        try:
            writer.add_scalars("loss-multitask-epoch-{}-{}".format(loss_lab, mode),  {"{}-{}-{}-{}".format("loss", mode, data_label, model_id): loss_val/n_tokens_dic[loss_lab]}, epoch)
        except:
            print("WARNING : could not report loss in tensorboard for epoch {}, n_token {} , loss {} , loss task {} data {}".format(epoch, n_tokens_dic[loss_lab], loss_val, loss_lab, data_label))


def tensorboard_loss_writer_epoch_level(writer, tasks, mode, model_id, epoch, n_batch_norm, n_batch_pos, append_n_mask, loss, loss_norm, loss_pos, loss_n_mask_prediction, batch_i):
    """
    NB : loss provided is already supposed to be average per batch
    :param writer:
    :param tasks:
    :param mode:
    :param model_id:
    :param epoch:
    :param n_batch_norm:
    :param n_batch_pos:
    :param append_n_mask:
    :param loss:
    :param loss_norm:
    :param loss_pos:
    :param loss_n_mask_prediction:
    :param batch_i:
    :return:
    """
    writer.add_scalars("loss-overall-epoch-{}-{}".format(tasks[0], mode),
                       {"{}-{}-{}".format("loss", mode, model_id): loss/batch_i}, epoch)
    if "normalize" in tasks:
        try:
            writer.add_scalars("loss-norm-epoch",
                       {"loss-{}-{}-bpe".format(mode, model_id): loss_norm.clone().cpu().data.numpy()/n_batch_norm},
                       epoch)
        except Exception as e:
            print("ERROR {} loss_pos is , n_batch_pos is {} coud not log ".format(e, loss_norm, n_batch_norm))
        if append_n_mask:
            writer.add_scalars("loss-n_mask_prediction-epoch",
                               {"loss-{}-{}-n_mask_prediction".format(mode,
                                model_id): loss_n_mask_prediction.clone().cpu().data.numpy()/n_batch_norm},
                               epoch)
    if "pos" in tasks:
        try:
            writer.add_scalars("loss-pos-epoch",
                       {"loss-{}-{}-bpe".format(mode, model_id): loss_pos.clone().cpu().data.numpy()/n_batch_pos},
                       epoch)
        except Exception as e:
            print("ERROR {} loss_pos is , n_batch_pos is {} coud not log ".format(e, loss_pos, n_batch_pos))



def writing_predictions_conll(dir_normalized, dir_normalized_original_only, dir_gold, dir_gold_original_only,
                              src_detokenized, inverse_writing, pred_detokenized_topk, task_pos_is, iter, batch_i,
                              new_file, gold_detokenized, verbose):

    write_conll(format="conll", dir_normalized=dir_normalized,
                dir_original=dir_normalized_original_only,
                src_text_ls=src_detokenized, inverse=inverse_writing,
                text_decoded_ls=pred_detokenized_topk[0],  # pred_pos_ls=None, src_text_pos=None,
                tasks=["pos" if task_pos_is else "normalize"], ind_batch=iter + batch_i, new_file=new_file,
                src_text_pos=src_detokenized, pred_pos_ls=gold_detokenized,
                verbose=verbose)
    write_conll(format="conll", dir_normalized=dir_gold, dir_original=dir_gold_original_only,
                src_text_ls=src_detokenized,
                src_text_pos=src_detokenized, pred_pos_ls=gold_detokenized,
                text_decoded_ls=gold_detokenized,  # pred_pos_ls=None, src_text_pos=None,
                tasks=["pos" if task_pos_is else "normalize"],
                ind_batch=iter + batch_i, new_file=new_file, verbose=verbose)
    new_file = False
    return new_file


def writing_predictions_conll_multi(dir_pred, dir_normalized_original_only,
                                    dir_gold, dir_gold_original_only,
                                    src_detokenized, pred_per_task,
                                    iter, batch_i, tasks, all_indexes,task_parameters,
                                    new_file, gold_per_tasks, verbose):

    write_conll_multitask(format="conll", dir_pred=dir_pred,
                          dir_original=dir_normalized_original_only,
                          src_text_ls=src_detokenized,
                          tasks=tasks, ind_batch=iter + batch_i, new_file=new_file,
                          pred_per_task=pred_per_task,
                          task_parameters=task_parameters,
                          all_indexes=all_indexes,
                          verbose=verbose)
    write_conll_multitask(format="conll", dir_pred=dir_gold,
                          dir_original=dir_gold_original_only, tasks=tasks,
                          src_text_ls=src_detokenized,
                          pred_per_task=gold_per_tasks, gold=True,
                          all_indexes=all_indexes,
                          task_parameters=task_parameters,
                          ind_batch=iter + batch_i, new_file=new_file, verbose=verbose)

    return False


def get_task_name_based_on_logit_label(logit_label, label_processed):
    match = re.match("(.*)-(.*)", logit_label)
    assert match  is not None, "ERROR {}".format(logit_label)
    label = match.group(2)
    task = match .group(1)
    #else:
    #    label = logit_label
    _continue = False
    if label in label_processed:
        _continue = True
    else:
        _continue = False
        label_processed.append(label)
    return label, task, _continue, label_processed


def get_task_label(tasks, task_settings):
    list_label_score = []
    for task in tasks:
        #list_label_score.extend(task_settings[task]["label"])
        list_label_score.extend([task+"-"+labe for labe in task_settings[task]["label"]])
    return list_label_score


def init_score_token_sent_dict(samples_per_task_reporting, tasks, agg_func_ls, compute_intersection_score, task_settings):

    # TODO : make it more systematic (should not hardcode 'normalize' "

    samples = samples_per_task_reporting["normalize"] #["all", "NEED_NORM", "NORMED", "PRED_NEED_NORM", "PRED_NORMED", "InV", "OOV"]
    init_samples = samples.copy()
    init_samples_per_task = {}

    labels = get_task_label(tasks, task_settings)

    for task in labels:
        if task.startswith("mwe"):
            init_samples_per_task[task] = samples_per_task_reporting[task].copy()
        else:
            init_samples_per_task[task] = samples_per_task_reporting["normalize"].copy()

    if compute_intersection_score:
        for task in labels:
            for ind, sam in enumerate(init_samples_per_task[task][1:]):
                for ind_2 in range(ind):
                    init_samples_per_task[task].append(sam+"-n-"+init_samples_per_task[task][ind_2+1])

    score_dic = {task: {agg_func: {sample: 0 for sample in init_samples_per_task[task]} for agg_func in agg_func_ls} for task in labels}

    n_tokens_dic = {task: {agg_func: {sample: 0 for sample in init_samples_per_task[task]} for agg_func in agg_func_ls} for task in labels}
    n_sents_dic = {task: {agg_func: {sample: 0 for sample in init_samples_per_task[task]} for agg_func in agg_func_ls} for task in labels}
    if "normalize" in tasks:
        for extra_label in ["n_masks_pred", "normalize_pred"]:
            score_dic[extra_label] = {"sum": {sample: 0 for sample in samples_per_task_reporting[extra_label]}}
            n_tokens_dic[extra_label] = {"sum": {sample: 0 for sample in samples_per_task_reporting[extra_label]}}
            n_sents_dic[extra_label] = {"sum": {sample: 0 for sample in samples_per_task_reporting[extra_label]}}

    return score_dic, n_tokens_dic, n_sents_dic


def dimension_check_label(label_per_task, input_tokens_tensor):
    for task, labels in label_per_task.items():
        labels.size(0) == input_tokens_tensor.size(
            0), "task {} output_tokens_tensor_aligned.size(0) {} input_tokens_tensor.size() {}".format(task,
                                                                                                       labels.size(),
                                                                                                       input_tokens_tensor.size())
        labels.size(1) == input_tokens_tensor.size(
            1), "task {} output_tokens_tensor_aligned.size(1) {} input_tokens_tensor.size() {}".format(task,
                                                                                                       labels.size(
                                                                                                           1),
                                                                                                       input_tokens_tensor.size(
                                                                                                           1))


def extend_input(masks, input, input_alignement_with_raw, mask_token_index, use_gpu):
    """
    extend input based on predicted masks
    :param masks: predicted number of inputs
    :param input:
    :param input_alignement_with_raw:
    :param mask_token_index:
    :param use_gpu:
    :return:
    """
    assert masks.size(0) == input.size(0)
    assert masks.size(1) == input.size(1)
    extended_input = []
    extended_alignement = []
    max_len = 0
    for ind_sent in range(masks.size(0)):
        extended_input_sent = []
        extended_alignement_sent = []
        for ind_tok in range(input.size(1)):
            # we account 0 prediction as 1 for prediction
            if masks[ind_sent, ind_tok].item() > 1:
                extended_input_sent.append(input[ind_sent, ind_tok].item())
                extended_input_sent.extend([mask_token_index for _ in range(masks[ind_sent, ind_tok]-1)])
                extended_alignement_sent.extend([input_alignement_with_raw[ind_sent, ind_tok].item() for _ in range(masks[ind_sent, ind_tok])])
            else:
                extended_input_sent.append(input[ind_sent, ind_tok].item())
                extended_alignement_sent.extend([input_alignement_with_raw[ind_sent, ind_tok].item() for _ in range(max(masks[ind_sent, ind_tok],1))])
            max_len = max(len(extended_input_sent), max_len)
        extended_input.append(extended_input_sent)
        extended_alignement.append(extended_alignement_sent)
    # add padding
    extended_input = [sent + [0 for _ in range(max_len - len(sent))] for sent in extended_input]
    extended_alignement_sent = [sent_alignement + [1000 for _ in range(max_len - len(sent_alignement))] for
                                sent_alignement in extended_alignement]

    extended_input_torch = torch.tensor(extended_input)
    extended_alignement_sent_torch = torch.tensor(extended_alignement_sent)

    if use_gpu:
        extended_input_torch = extended_input_torch.cuda()
        extended_alignement_sent_torch = extended_alignement_sent_torch.cuda()

    return extended_input_torch, extended_alignement_sent_torch
