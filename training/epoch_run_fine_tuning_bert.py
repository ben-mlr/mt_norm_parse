#from env.importing import *
#from env.project_variables import *

from env.importing import np, torch, OrderedDict, time, pdb, os
from env.project_variables import SAMPLES_PER_TASK_TO_REPORT, AVAILABLE_BERT_MASKING_STRATEGY, MULTITASK_BERT_LABELS_MLM_HEAD
from env.tasks_settings import TASKS_PARAMETER
from io_.dat.constants import PAD_ID_BERT, MASK_BERT, CLS_BERT, SEP_BERT, SPECIAL_TOKEN_LS, NULL_STR, PAD_ID_TAG, PAD_ID_LOSS_STANDART
from io_.info_print import printing
from io_.bert_iterators_tools.string_processing import preprocess_batch_string_for_bert, from_bpe_token_to_str, get_indexes, get_indexes_src_gold
from io_.bert_iterators_tools.get_string_from_bpe import get_prediction, get_bpe_string, get_indexes_src_gold, get_detokenized_str, get_aligned_output
#from io_.bert_iterators_tools.alignement import aligned_output, realigne
import io_.bert_iterators_tools.alignement as alignement
from evaluate.report_writing import report_score_all
from evaluate.scoring.report import overall_word_level_metric_measure
from model.n_masks_predictor import get_n_bpe_pred
from model.bert_tools_from_core_code.modeling import get_multitask_loss
from toolbox.pred_tools.heuristics import predict_with_heuristic
from training.epoch_run_fine_tuning_tools import get_casing, logging_processing_data, logging_scores, log_warning, print_align_bpe, tensorboard_loss_writer_batch_level, tensorboard_loss_writer_batch_level_multi, \
    tensorboard_loss_writer_epoch_level, \
    writing_predictions_conll, writing_predictions_conll_multi, init_score_token_sent_dict, dimension_check_label, extend_input, tensorboard_loss_writer_epoch_level_multi, update_loss_dic_average
from io_.bert_iterators_tools.get_bpe_labels import get_label_per_bpe, get_mask_input
from toolbox.deep_learning_toolbox import dropout_input_tensor
from model.bert_tools_from_core_code.masking import focused_masking


def accumulate_scores_across_sents(agg_func_ls, sample_ls, dic_prediction_score, score_dic, n_tokens_dic, n_sents_dic):
    for agg_func in agg_func_ls:
        for sample in sample_ls:
            score_dic[agg_func][sample] += dic_prediction_score[agg_func][sample]["score"]
            n_tokens_dic[agg_func][sample] += dic_prediction_score[agg_func][sample]["n_tokens"]
            n_sents_dic[agg_func][sample] += dic_prediction_score[agg_func][sample]["n_sents"]
    return score_dic, n_tokens_dic, n_sents_dic


def epoch_run(batchIter, tokenizer,
              args,
              iter, n_iter_max, model, epoch,
              use_gpu, data_label, null_token_index, null_str,
              model_id, early_stoppin_metric=None,
              pos_dictionary=None,
              skip_1_t_n=True,
              writer=None, optimizer=None,
              predict_mode=False, topk=None, metric=None,
              print_pred=False, args_dir=None,
              reference_word_dic=None, dropout_input_bpe=0.,
              writing_pred=False, dir_end_pred=None, extra_label_for_prediction="",
              log_perf=True, remove_mask_str_prediction=False,
              inverse_writing=False,
              norm_2_noise_eval=False,
              compute_intersection_score=False,
              subsample_early_stoping_metric_val=None,
              slang_dic=None, list_reference_heuristic=None,list_candidates=None, index_alphabetical_order=None,
              case=None, threshold_edit=None, edit_module_pred_need_norm_only=True, low_memory_foot_print_batch_mode=False,
              batch_size_real=0, n_epoch=None,
              ponderation_loss_policy="static",
              samples_per_task_reporting=None,
              task_to_eval=None, task_to_label_dictionary=None,
              verbose=0):
    """
    About Evaluation :
    Logic : compare gold and prediction topk using a word level scoring fucntion
            then accumulates for each sentences and foea each batch to get global score
            CAN add SAMPLE Parameter to get scores on specific subsample of the data : e.g. NEED_NORM, NORMED...
            Can also have different aggregation function
    """
    if args.multitask:
        assert task_to_label_dictionary is not None, "ERROR : task_to_label_dictionary should be defined "
    if not args.multitask:
        assert ("pos" in args.tasks[0] or "normalize" in args.tasks[0]), \
            "ERROR : in --multitask 0 mode only normalize and pos supported bu {} tasks passed ".format(args.tasks)
    if samples_per_task_reporting is None:
        samples_per_task_reporting = SAMPLES_PER_TASK_TO_REPORT
    if task_to_eval is not None:
        args.tasks = task_to_eval
        assert task_to_eval in task_to_label_dictionary, "ERROR : {} label was not provided in {}".format(task_to_eval, task_to_label_dictionary)
        printing("WARNING : task_to_eval was provided ", verbose=verbose, verbose_level=1)
    if ponderation_loss_policy == "static":
        if args.multi_task_loss_ponderation is None:
            args.multi_task_loss_ponderation = OrderedDict([("loss_task_1", 1), ("loss_task_2", 1), ("loss_task_n_mask_prediction", 1)])
            printing("TRAINING : setting default args.multi_task_loss_ponderation {} ",
                     var=[args.multi_task_loss_ponderation], verbose=verbose, verbose_level=1)
    else:
        raise(Exception("Only static strategy supported so far"))

    if subsample_early_stoping_metric_val is None:
        subsample_early_stoping_metric_val = "all"
    if low_memory_foot_print_batch_mode:
        assert batch_size_real > 0, "ERROR have to define batch_size_real in low_memory_foot_print_batch_mode"

    if args.heuristic_ls is not None:
        for edit_rule in ["all", "ref", "data"]:
            if "edit_check-"+edit_rule in args.heuristic_ls:
                assert threshold_edit is not None, "ERROR threshold_edit required as args.heuristic_ls is {}".format(args.heuristic_ls)
    if case is not None:
        AVAILABLE_CASE_OPTIONS = ["lower"]
        assert case in AVAILABLE_CASE_OPTIONS
    assert args.norm_2_noise_training is None or not norm_2_noise_eval, "only one of the two should be triggered but we have args.norm_2_noise_training : {} norm_2_noise_eval:{}".format(args.norm_2_noise_training, norm_2_noise_eval)
    if args.norm_2_noise_training is not None:
        printing("WARNING : {} args.norm_2_noise_training is on ", var=[args.norm_2_noise_training],
                 verbose=verbose, verbose_level=1)
    if norm_2_noise_eval:
        printing("WARNING : {} norm_2_noise_eval is on ", var=[norm_2_noise_eval],
                 verbose=verbose, verbose_level=1)
    assert len(args.tasks) <= 2
    evaluated_task = []
    skip_score = 0
    skipping = 0
    mean_end_pred = 0
    label_heuristic = ""
    if args.gold_error_detection:
        label_heuristic += "-gold"
    if args.heuristic_ls is not None:
        label_heuristic += "-"+"_".join(args.heuristic_ls)
    if norm_2_noise_eval:
        label_heuristic += "-noise_generation"
    printing("INFO : HEURISTIC used {} {}", var=[args.heuristic_ls, label_heuristic], verbose=verbose, verbose_level=1)
    if args.masking_strategy is not None:
        if "start_stop" not in args.masking_strategy:
            assert "normalize" in args.tasks[0], "SO FAR : inconsistency between task {} and masking strategy {}".format(args.tasks, args.masking_strategy)
        if isinstance(args.masking_strategy, list):
            assert len(args.masking_strategy) <= 2, \
                "first element should be strategy, second should be portion or first element only ".format(args.masking_strategy)
            if len(args.masking_strategy) == 2:
                args.portion_mask = eval(str(args.masking_strategy[1]))
                args.masking_strategy = args.masking_strategy[0]
            else:
                args.masking_strategy = args.masking_strategy[0]
        assert args.masking_strategy in AVAILABLE_BERT_MASKING_STRATEGY, "args.masking_strategy {} should be in {}".format(args.masking_strategy, AVAILABLE_BERT_MASKING_STRATEGY)
        if args.masking_strategy == "normed":
            printing("INFO : Portion mask was found to {}", var=[args.portion_mask], verbose=verbose, verbose_level=1)
    if predict_mode:
        if topk is None:
            topk = 1
            printing("PREDICTION MODE : setting top-k to default 1 ", verbose_level=1, verbose=verbose)
        print_pred = False
        if metric is None:
            metric = "exact_match"
            printing("PREDICTION MODE : setting metric to default 'exact_match' ", verbose_level=1, verbose=verbose)

    if writing_pred:
        assert dir_end_pred is not None
        if extra_label_for_prediction != "":
            extra_label_for_prediction = "-"+extra_label_for_prediction
        extra_label_for_prediction += "-"+label_heuristic
        dir_normalized = os.path.join(dir_end_pred, "{}_ep-prediction{}.conll".format(epoch,
                                                                                      extra_label_for_prediction))
        dir_normalized_original_only = os.path.join(dir_end_pred, "{}_ep-prediction_src{}.conll".format(epoch,
                                                                                                        extra_label_for_prediction))
        dir_gold = os.path.join(dir_end_pred, "{}_ep-gold-{}.conll".format(epoch,
                                                                          extra_label_for_prediction))
        dir_gold_original_only = os.path.join(dir_end_pred, "{}_ep-gold_src{}.conll".format(epoch,
                                                                                            extra_label_for_prediction))
    mask_token_index = tokenizer.convert_tokens_to_ids([MASK_BERT])[0]
    cls_token_index = tokenizer.convert_tokens_to_ids([CLS_BERT])[0]
    sep_token_index = tokenizer.convert_tokens_to_ids([SEP_BERT])[0]

    space_token_index = tokenizer.convert_tokens_to_ids([null_str])[0]
    printing("WARNING : [MASK] set to {} [CLS] {} [SEP] {}", var=[mask_token_index, cls_token_index, sep_token_index],
             verbose=verbose, verbose_level=1)

    batch_i = 0
    noisy_over_splitted = 0
    noisy_under_splitted = 0
    aligned = 0
    skipping_batch_n_to_1 = 0

    loss = 0

    agg_func_ls = ["sum"]

    score_dic, n_tokens_dic, n_sents_dic = init_score_token_sent_dict(samples_per_task_reporting, [task for tasks in args.tasks for task in tasks],
                                                                      agg_func_ls, compute_intersection_score,
                                                                      task_settings=TASKS_PARAMETER)

    _samples_per_task_reporting = list(samples_per_task_reporting.keys())+["all"]
    n_tokens_counter_per_task = OrderedDict((a, 0) for a in _samples_per_task_reporting)
    loss_dic_epoch = OrderedDict((a, 0) for a in _samples_per_task_reporting)

    # TODO : should be removed (everuthing should go through samples_per_task_reporting)
    samples = samples_per_task_reporting["normalize"]
    # vocab_index_except_pad_cls_sep = [i for i in range(1, len(tokenizer.vocab)) if i not in [mask_token_index, sep_token_index, cls_token_index]]
    # pad is the first index
    skipping_evaluated_batch = 0
    mode = "?"
    new_file = True

    labels_n_mask_prediction = None
    loss_norm = 0
    loss_pos = 0
    loss_n_mask_prediction = 0
    n_batch_pos = 0
    n_batch_norm = 0
    n_task_pos_sanity = 0
    n_task_normalize_sanity = 0
    input_token_mask = None

    counting_failure_parralel_bpe_batch = 0
    while True:

        try:
            batch_i += 1
            batch = batchIter.__next__()
            n_tokens_counter_current_per_task = OrderedDict()
            # if no normalization found : should have pos
            task_pos_is = len(batch.raw_output[0]) == 0
            # only one task supported at a time per batch so far based on the input batch
            task_normalize_is = not args.multitask
            task_pos_is = False#"pos" in args.tasks and len(args.tasks) == 1
            if task_pos_is:
               print("WARNING : task_pos_is  {} ".format(task_pos_is))
            # case the batches if case is 'lower'
            batch = get_casing(case, batch, task_normalize_is)
            #print("ITERATING on {} task".format("pos" if task_pos_is else "normalize"))
            n_task_pos_sanity += int(task_pos_is)
            n_task_normalize_sanity += int(task_normalize_is)
            norm2noise_bool = False
            batch_raw_output = None
            # Handling input
            if (args.norm_2_noise_training is not None or norm_2_noise_eval) and task_normalize_is:
                portion_norm2noise = args.norm_2_noise_training if args.norm_2_noise_training is not None else 1.
                args.norm_2_noise_training = portion_norm2noise is not None
                rand = np.random.uniform(low=0, high=1, size=1)[0]
                norm2noise_bool = portion_norm2noise >= rand
                if norm2noise_bool:
                    batch_raw_input = preprocess_batch_string_for_bert(batch.raw_output)
                    printing("WARNING : input is gold norm", verbose_level=2, verbose=1)
                else:
                    printing("WARNING : input is input", verbose_level=2, verbose=1)
                    batch_raw_input = preprocess_batch_string_for_bert(batch.raw_input)
            else:
                printing("WARNING : input is input ", verbose_level=2, verbose=1)
                batch_raw_input = preprocess_batch_string_for_bert(batch.raw_input)

            group_to_mask = None

            if args.masking_strategy == "cls":
                # we trick batch.output_norm_not_norm : set all 1 to 0 (not to touch padding)
                # we set first element to 1
                batch.output_norm_not_norm[batch.output_norm_not_norm == 1] = 0
                batch.output_norm_not_norm[:, 0] = 1
                group_to_mask = batch.output_norm_not_norm
            elif args.masking_strategy == "normed":
                rand = np.random.uniform(low=0, high=1, size=1)[0]
                group_to_mask = np.array(batch.output_norm_not_norm.cpu()) if args.portion_mask >= rand else None
            if not args.tokenize_and_bpe:
                input_tokens_tensor, input_segments_tensors, inp_bpe_tokenized, input_alignement_with_raw, input_mask = \
                    get_indexes(batch_raw_input, tokenizer, verbose, use_gpu, word_norm_not_norm=group_to_mask)
            if args.masking_strategy == "start_stop":
                input_mask[input_tokens_tensor == sep_token_index] = 0
                input_mask[input_tokens_tensor == cls_token_index] = 0

            if task_normalize_is:

                if norm2noise_bool or norm_2_noise_eval:
                    printing("WARNING : output is noisy input", verbose_level=2, verbose=1)
                    batch_raw_output = preprocess_batch_string_for_bert(batch.raw_input)
                else:
                    printing("WARNING : output is output", verbose_level=2, verbose=1)
                    batch_raw_output = preprocess_batch_string_for_bert(batch.raw_output, rp_space=True)

                if args.tokenize_and_bpe:
                    try:
                        tokens_tensor_dic, segments_tensors_dic, tokenized_dic, aligned_index_padded_dic, mask_dic = \
                            get_indexes_src_gold(list_pretokenized_str_source=batch_raw_input,
                                                 list_pretokenized_str_gold=batch_raw_output,
                                                 tokenizer=tokenizer, verbose=verbose, use_gpu=use_gpu)

                        output_tokens_tensor, output_segments_tensors, out_bpe_tokenized, output_alignement_with_raw, output_mask = \
                            tokens_tensor_dic["gold"], segments_tensors_dic["gold"], tokenized_dic["gold"], \
                            aligned_index_padded_dic["gold"], mask_dic["gold"]

                        input_tokens_tensor, input_segments_tensors, inp_bpe_tokenized, input_alignement_with_raw, input_mask = \
                            tokens_tensor_dic["src"], segments_tensors_dic["src"], tokenized_dic["src"], \
                            aligned_index_padded_dic["src"], mask_dic["src"]
                    except Exception as e:
                        print("FAILLING error {} TO ALIGN batch_raw_input {} with "
                              "batch_raw_output {} so using the old method".format(e, batch_raw_input, batch_raw_output))
                        input_tokens_tensor, input_segments_tensors, inp_bpe_tokenized, input_alignement_with_raw, input_mask = \
                            get_indexes(batch_raw_input, tokenizer, verbose, use_gpu, word_norm_not_norm=None)
                        output_tokens_tensor, output_segments_tensors, out_bpe_tokenized, output_alignement_with_raw, output_mask = \
                            get_indexes(batch_raw_output, tokenizer, verbose, use_gpu)
                        counting_failure_parralel_bpe_batch += 1
                else:
                    output_tokens_tensor, output_segments_tensors, out_bpe_tokenized, output_alignement_with_raw, output_mask = get_indexes(batch_raw_output, tokenizer, verbose, use_gpu, word_norm_not_norm=None)
                printing("DATA dim : {} input {} output ", var=[input_tokens_tensor.size(), output_tokens_tensor.size()],
                         verbose_level=2, verbose=verbose)

            if args.multitask:
                out_bpe_tokenized = None
                # TODO : should have a task specific input_mask and head_masks : only considering word level tasks and bpe level tasks for now
                input_mask = get_mask_input(input_tokens_tensor, use_gpu)
                head_masks, input_tokens_tensor, token_type_ids, label_per_task, input_tokens_tensor_per_task, input_mask_per_task = get_label_per_bpe(args.tasks, batch, input_tokens_tensor, input_alignement_with_raw,  use_gpu, tasks_parameters=TASKS_PARAMETER)
                # NB : token_type_ids not used in MultiTask (no needed, just use 0 everywhere )
                dimension_check_label(label_per_task, input_tokens_tensor)

                # NB : we use the aligned input with the
            # logging
            verbose_level = verbose if verbose in ["raw_data", "alignement"] else "raw_data"
            #logging_processing_data(verbose, verbose, verbose_level, batch_raw_input, input_tokens_tensor,
            #                        batch_raw_output, output_tokens_tensor_aligned, inp_bpe_tokenized, out_bpe_tokenized)

            _1_to_n_token = 0
            if task_normalize_is:
                # aligning output BPE with input
                # (we are rejecting batch with at least one 1 to n case # (that we don't want to handle)
                try:
                    output_tokens_tensor_aligned, input_tokens_tensor_aligned, input_alignement_with_raw, input_mask, _1_to_n_token = \
                        alignement.aligned_output(input_tokens_tensor, output_tokens_tensor, input_alignement_with_raw, output_alignement_with_raw,
                                                  mask_token_index=mask_token_index, input_mask=input_mask, use_gpu=use_gpu,null_token_index=null_token_index, verbose=verbose)
                except Exception as e:
                    print("ERROR : (epoch_run_fine_tuning_bert.py) alignement.aligned_output failed {}".format(e))
                    pdb.set_trace()
                    raise(e)

                assert output_tokens_tensor_aligned.size(0) == input_tokens_tensor_aligned.size(0), \
                    "output_tokens_tensor_aligned.size(0) {} input_tokens_tensor.size() {}".format(output_tokens_tensor_aligned.size(), input_tokens_tensor_aligned.size())
                assert output_tokens_tensor_aligned.size(1) == input_tokens_tensor_aligned.size(1), \
                    "output_tokens_tensor_aligned.size(1) {} input_tokens_tensor.size() {}".format(output_tokens_tensor_aligned.size(1), input_tokens_tensor_aligned.size(1))

                input_token_mask = input_tokens_tensor.clone()
                input_tokens_tensor = input_tokens_tensor_aligned
                token_type_ids = torch.zeros_like(input_tokens_tensor)
                if use_gpu:
                    token_type_ids = token_type_ids.cuda()
                #
                #TODO : creaate a tensor same dim as output_tokens_tensor based on output_alignement_with_raw
                # number of repetition in output_alignement_with_raw
                # or number of bpe tokens related to each bpe

            if batch_i == n_iter_max:
                break
            if batch_i % 1000 == 0:
                printing("TRAINING : iteration finishing {}/{} batch", var=[batch_i, n_iter_max], verbose=verbose, verbose_level=1)
            if _1_to_n_token:
                skipping_batch_n_to_1 += _1_to_n_token
                #continue
            # sanity checking alignement
            # we consider only 1 sentence case
            #printing("CUDA SANITY CHECK input_tokens:{}  type:{}input_mask:{}  label:{}", var=[input_tokens_tensor.is_cuda, token_type_ids.is_cuda, input_mask.is_cuda, output_tokens_tensor_aligned.is_cuda], verbose=verbose, verbose_level="cuda")
            # we have to recompute the mask based on aligned input
            if dropout_input_bpe > 0:
                input_tokens_tensor, mask_dropout, dropout_applied = dropout_input_tensor(input_tokens_tensor,
                                                                                          mask_token_index,
                                                                                          sep_token_index=sep_token_index,
                                                                                          dropout=dropout_input_bpe,
                                                                                          applied_dropout_rate=True)
            if args.append_n_mask and task_normalize_is:
                # --> the label needs to be he same shape of input : input_token_mask
                # with -1 for pad
                labels_n_mask_prediction = get_n_bpe_pred(input_tokens_tensor == mask_token_index)
                # - handle both output and both input , should it be calling the same model api ??
                # - Backpropagate in one step
                # - predict in two steps
                # sanity test : are mask correectly encoded as -1
                assert (((input_tokens_tensor == mask_token_index).nonzero()
                         == (labels_n_mask_prediction == -1).nonzero())).all()
                # Assigning padded input to label -1 for loss ignore
                labels_n_mask_prediction[input_tokens_tensor == 0] = -1

            # TODO : to factorize
            if not args.multitask and optimizer is not None:
                input_tokens_tensor, feeding_the_model_with_label = \
                    focused_masking(args.masking_strategy, input_tokens_tensor, output_tokens_tensor_aligned, dropout_input_bpe,
                                    mask_token_index, sep_token_index, use_gpu, epoch, n_epoch, args.portion_mask,
                                    input_mask, tokenizer,
                                    verbose)
            elif not args.multitask:
                feeding_the_model_with_label = output_tokens_tensor_aligned.clone()
            elif args.multitask:
                assert args.masking_strategy is None, "ERROR : {} not supported in multitask mode ".format(args.masking_strategy)
            if not args.multitask:
                assert len(args.tasks) == 1 and ("normalize" in args.tasks[0]), \
                    "ERROR : as args.multitask False mode only pos or normalize (pos now supported in args.multitask ONLY) " \
                    "supported (single) task while we have {} ".format(args.multitask)
                # is meant to be completely removed
                feeding_the_model_with_label[feeding_the_model_with_label == 0] = -1
                assert len(args.tasks) == 1, "ERROR : when args.multitask not True : only allowing one task at the time "
                loss_dic, layer_wise_weights = model(input_ids=input_tokens_tensor,
                                                     token_type_ids=token_type_ids, attention_mask=input_mask,
                                                     labels=feeding_the_model_with_label if "normalize" in args.tasks[0] else None,
                                                     labels_n_masks=labels_n_mask_prediction,
                                                     #input_token_mask=input_token_mask,
                                                     labels_task_2=output_tokens_tensor_aligned if "pos" in args.tasks[0] else None,
                                                     aggregating_bert_layer_mode=args.aggregating_bert_layer_mode,
                                                     mask_token_index=mask_token_index,
                                                     multi_task_loss_ponderation=args.multi_task_loss_ponderation)

                _loss = loss_dic["loss"]

                # report the loss per args.tasks
                if task_normalize_is:
                    loss_norm += loss_dic["loss_task_1"].detach()
                    n_batch_norm += 1
                    if args.append_n_mask:
                        if not isinstance(loss_dic["loss_task_n_mask_prediction"], int):
                            loss_n_mask_prediction += loss_dic["loss_task_n_mask_prediction"].detach()
                if task_pos_is:
                    loss_pos += loss_dic["loss_task_2"].detach()
                    n_batch_pos += 1
                if predict_mode:
                    start_pred = time.time()
                    predictions_topk = {}
                    # if predict more : will evaluate the model and write its predictions
                    # TODO : add mapping_info between task_id to model and task name necessary to iterator
                    logits, layer_wise_weights = model(input_tokens_tensor, token_type_ids, input_mask,
                                                       aggregating_bert_layer_mode=args.aggregating_bert_layer_mode,
                                                       mask_token_index=mask_token_index,
                                                       multi_task_loss_ponderation=args.multi_task_loss_ponderation)

                    predicted_task = "normalize"
                    logits_task_label = MULTITASK_BERT_LABELS_MLM_HEAD[predicted_task]

                    # add prediction n_masks -->
                    if args.append_n_mask and task_normalize_is:
                        # TODO : --> should add a : simultaneous task module !
                        assert logits["logits_n_mask_prediction"] is not None, \
                            "ERROR : args.append_n_mask is {} while logits['logits_n_mask_prediction'] is None".format(args.append_n_mask)
                        prediction_n_mask = torch.argsort(logits["logits_n_mask_prediction"],
                                                          dim=-1, descending=True)[:, :, 0]
                        # TODO : extend  input_tokens_tensor based on prediction_n_mask : append masks
                        pred_inputs, extended_input_alignement_with_raw = extend_input(prediction_n_mask, input_tokens_tensor,
                                                                                       input_alignement_with_raw, mask_token_index, use_gpu)

                    if logits_task_label != "parsing":
                        predictions_topk[logits_task_label] = torch.argsort(logits[logits_task_label], dim=-1,
                                                                            descending=True)[:, :, :topk]
                    # from bpe index to string
                    sent_ls_top = from_bpe_token_to_str(predictions_topk[logits_task_label], topk, tokenizer=tokenizer,
                                                        pred_mode=True, pos_dictionary=pos_dictionary,
                                                        null_token_index=null_token_index, null_str=null_str)

                    gold = from_bpe_token_to_str(output_tokens_tensor_aligned, topk, tokenizer=tokenizer,
                                                 pos_dictionary=pos_dictionary,
                                                 pred_mode=False, null_token_index=null_token_index, null_str=null_str)
                    source_preprocessed = from_bpe_token_to_str(input_tokens_tensor, topk, tokenizer=tokenizer,
                                                                pos_dictionary=pos_dictionary,
                                                                pred_mode=False, null_token_index=null_token_index,
                                                                null_str=null_str, verbose=verbose)
                    # de-BPE-tokenize
                    src_detokenized = alignement.realigne(source_preprocessed, input_alignement_with_raw,
                                                          null_str=null_str, tasks=["normalize"],
                                                          # normalize means we deal wiht bpe input not pos
                                                          mask_str=MASK_BERT, remove_mask_str=remove_mask_str_prediction)
                    gold_detokenized = alignement.realigne(gold, input_alignement_with_raw, remove_null_str=True,
                                                           null_str=null_str, tasks=args.tasks[0], mask_str=MASK_BERT)


                    pred_detokenized_topk = []
                    pred_n_masks_detokenized_topk = []

                    if args.append_n_mask and task_normalize_is:
                        # TODO : factorize somewhere !
                        sent_ls_pred_n_masks_top = from_bpe_token_to_str(pred_inputs, topk, tokenizer=tokenizer, pred_mode=False,
                                                                         pos_dictionary=pos_dictionary,
                                                                         null_token_index=null_token_index, null_str=null_str)

                        pred_n_masks_detokenized_topk.append(alignement.realigne(sent_ls_pred_n_masks_top,
                                                                                 extended_input_alignement_with_raw,
                                                                                 remove_null_str=True,
                                                                                 tasks=args.tasks[0],
                                                                                 remove_mask_str=True,
                                                                                 remove_extra_predicted_token=True,
                                                                                 null_str=null_str, mask_str=MASK_BERT))
                        printing("DEBUG (training/epoch_run_fine_tuning_bert.py) INPUT input_tokens_tensor {} "
                              "\n DEBUG : prediction_n_mask {}  PRED "
                              "\n_masks predicted inputs {}  "
                              "\n DEBUG : pred inputs as str  {} "
                              "\n DEBUG : aligned pred_input {} "
                              "\n DEBUG : labels n masks {} ".format(input_tokens_tensor, prediction_n_mask, pred_inputs, sent_ls_pred_n_masks_top, pred_n_masks_detokenized_topk, labels_n_mask_prediction), verbose_level="pred", verbose=verbose)

                    for sent_ls in sent_ls_top:
                        pred_detokenized_topk.append(alignement.realigne(sent_ls, input_alignement_with_raw,
                                                                         remove_null_str=True,
                                                                         tasks=args.tasks[0], remove_extra_predicted_token=True,
                                                                         null_str=null_str, mask_str=MASK_BERT))

                        # NB : applying those successively might overlay heuristic
                        if task_normalize_is:
                            if args.heuristic_ls is not None:
                                # NB : if the rules in args.heuristic_ls are not exclusive their order matters !!
                                # the last one will be the one that is applied
                                pred_detokenized_topk = predict_with_heuristic(src_detokenized=src_detokenized,
                                                                               pred_detokenized_topk=pred_detokenized_topk,
                                                                               list_reference=list_reference_heuristic, list_candidates=list_candidates,
                                                                               slang_dic=slang_dic,
                                                                               index_alphabetical_order=index_alphabetical_order,
                                                                               heuristic_ls=args.heuristic_ls,
                                                                               threshold_edit=threshold_edit,
                                                                               edit_module_pred_need_norm_only=edit_module_pred_need_norm_only,
                                                                               verbose=verbose)
                            # NB : we overlay prediction with args.gold_error_detection
                            if args.gold_error_detection:
                                pred_detokenized_topk = predict_with_heuristic(src_detokenized=src_detokenized,
                                                                               gold_detokenized=gold_detokenized,
                                                                               pred_detokenized_topk=pred_detokenized_topk,
                                                                               heuristic_ls=["gold_detection"], verbose=verbose)
                                #print("PRED after gold",gold_detokenized, pred_detokenized_topk)
                    if writing_pred:
                        # TODO : if you do multitask leaning
                        #  you'll have to adapt here (you're passing twice the same parameters)
                        new_file = writing_predictions_conll(dir_normalized, dir_normalized_original_only, dir_gold,
                                                             dir_gold_original_only,
                                                             src_detokenized, inverse_writing, pred_detokenized_topk,
                                                             task_pos_is, iter, batch_i, new_file,  gold_detokenized,
                                                             verbose)
                    try:

                        if task_normalize_is and args.append_n_mask:
                            # Masks
                            perf_prediction_n_mask, skipping_n_mask, _ = \
                                overall_word_level_metric_measure(
                                                                  task_label="n_masks_pred",
                                                                  gold_sent_ls_dict={"n_masks_pred": labels_n_mask_prediction.tolist()},
                                                                  pred_sent_ls_topk_dict={"n_masks_pred": [prediction_n_mask.tolist()]},
                                                                  topk=1,
                                                                  metric=metric,
                                                                  samples=samples_per_task_reporting["n_masks_pred"],
                                                                  agg_func_ls=agg_func_ls,
                                                                  reference_word_dic=reference_word_dic,
                                                                  compute_intersection_score=False,
                                                                  src_detokenized=None)

                            score_dic["n_masks_pred"], n_tokens_dic["n_masks_pred"], n_sents_dic["n_masks_pred"] = \
                                accumulate_scores_across_sents(agg_func_ls=agg_func_ls,
                                                               sample_ls=samples_per_task_reporting["n_masks_pred"],
                                                               dic_prediction_score=perf_prediction_n_mask,
                                                               score_dic=score_dic["n_masks_pred"],
                                                               n_tokens_dic=n_tokens_dic["n_masks_pred"],
                                                               n_sents_dic=n_sents_dic["n_masks_pred"])
                            # token based on predicted masks
                            perf_detok_prediction_on_n_mask, skipping_n_mask, _ = \
                                overall_word_level_metric_measure(
                                                                task_label="normalize_pred",
                                                                gold_sent_ls_dict={"normalize_pred": gold_detokenized},
                                                                pred_sent_ls_topk_dict={"normalize_pred": pred_n_masks_detokenized_topk},
                                                                topk=1,
                                                                metric=metric,
                                                                samples=samples_per_task_reporting["normalize_pred"],
                                                                agg_func_ls=agg_func_ls,
                                                                reference_word_dic=reference_word_dic,
                                                                compute_intersection_score=False,
                                                                src_detokenized=src_detokenized)
                            score_dic["normalize_pred"], n_tokens_dic["normalize_pred"], n_sents_dic["normalize_pred"] = \
                                accumulate_scores_across_sents(agg_func_ls=agg_func_ls,
                                                               sample_ls=samples_per_task_reporting["normalize_pred"],
                                                               dic_prediction_score=perf_detok_prediction_on_n_mask,
                                                               score_dic=score_dic["normalize_pred"],
                                                               n_tokens_dic=n_tokens_dic["normalize_pred"],
                                                               n_sents_dic=n_sents_dic["normalize_pred"])
                            evaluated_task.append("normalize_pred")
                            evaluated_task.append("n_masks_pred")

                        elif task_normalize_is:
                            # we fill it with an empty report for simplifying reporting
                            accumulate_scores_across_sents(agg_func_ls=agg_func_ls, sample_ls=["all"],
                                                           dic_prediction_score={agg_func_ls[0]: {"all": {"agg_func": agg_func_ls[0],"metric": "exact_match", "score": 0, "n_sents": 0, "n_tokens": 0 }}},
                                                           score_dic=score_dic["n_masks_pred"], n_tokens_dic=n_tokens_dic["n_masks_pred"], n_sents_dic=n_sents_dic["n_masks_pred"])

                            accumulate_scores_across_sents(agg_func_ls=agg_func_ls, sample_ls=["all"],
                                                           dic_prediction_score={agg_func_ls[0]:{"all": {"agg_func": agg_func_ls[0], "metric": "exact_match", "score": 0, "n_sents": 0, "n_tokens": 0}}},
                                                           score_dic=score_dic["normalize_pred"], n_tokens_dic=n_tokens_dic["normalize_pred"], n_sents_dic=n_sents_dic["normalize_pred"])
                            evaluated_task.append("n_masks_pred")
                            evaluated_task.append("normalize_pred")


                        evaluated_task.append(predicted_task)

                        perf_prediction, skipping, _samples = overall_word_level_metric_measure(task_label=predicted_task,
                                                                                                gold_sent_ls_dict={predicted_task: gold_detokenized},
                                                                                                pred_sent_ls_topk_dict={predicted_task: pred_detokenized_topk},
                                                                                                topk=topk,
                                                                                                metric=metric, samples=samples,
                                                                                                agg_func_ls=agg_func_ls,
                                                                                                reference_word_dic=reference_word_dic,
                                                                                                compute_intersection_score=compute_intersection_score,
                                                                                                src_detokenized=src_detokenized)

                        score_dic[predicted_task], n_tokens_dic[predicted_task], n_sents_dic[predicted_task] = \
                            accumulate_scores_across_sents(agg_func_ls=agg_func_ls, sample_ls=_samples,
                                                           dic_prediction_score=perf_prediction,
                                                           score_dic=score_dic[predicted_task],
                                                           n_tokens_dic=n_tokens_dic[predicted_task],
                                                           n_sents_dic=n_sents_dic[predicted_task])
                    except Exception as e:
                        skip_score += 1
                        print("SKIPPED {} evaluation current error : {} ".format(skip_score, e))
                        raise(e)
                    skipping_evaluated_batch += skipping

                    if print_pred:
                        logging_scores(perf_prediction, iter, batch_i, pred_detokenized_topk, verbose)
                    end_pred = time.time()-start_pred
                    mean_end_pred += end_pred/60

                    print_align_bpe(source_preprocessed, gold, input_alignement_with_raw, labels_n_mask_prediction,
                                    verbose=verbose, verbose_level=4)
            # multitask :
            elif args.multitask:
                #   - for all word level tasks : normalization, tagging, edit, parsing :
                #       handle realignement to words
                # --> evaluate them based on labels
                # HANDLE THE CASE FOR WHICH ONLY PREDICTION
                # TODO:
                # - factorize   masking
                assert "normalize" not in args.tasks[0], "ERROR : input and output not supported yet for 'normalize' task in this setting "
                for label in label_per_task:
                    # make mask for the loss padding
                    # TODO handle task specific index pad
                    # NB : maybe factorize with prediction
                    #assert len(set(args.tasks) & set(["parsing", "pos"])) == len(args.tasks), \
                    #    "ERROR need to handle tasks agnostic pad index for allowing other tasks {} ".format(args.tasks)
                    # we transform the padded labels according to the loss ignore mask parameters
                    if label not in ["parsing_heads", "mwe_prediction", "n_masks_mwe", "mwe_detection"]:
                        label_per_task[label][label_per_task[label] == PAD_ID_TAG] = PAD_ID_LOSS_STANDART
                    # we do the token counting using labels
                    n_tokens_counter_per_task[label] += (label_per_task[label] != PAD_ID_LOSS_STANDART).sum().item()
                    # NB : do you account for CLS and SEQ HERE ?
                    n_tokens_counter_current_per_task[label] = (label_per_task[label] != PAD_ID_LOSS_STANDART).sum().item()
                # TODO : handle in a more standart way
                n_tokens_counter_per_task["all"] += n_tokens_counter_current_per_task[label]
                logits_dic, loss_dic, _ = model(input_tokens_tensor_per_task,
                                                token_type_ids=None,
                                                labels=label_per_task, head_masks=head_masks, attention_mask=input_mask_per_task)

                if len(list(loss_dic.keys() & set(TASKS_PARAMETER.keys()))) != len(loss_dic.keys()):
                    # it means a given task has several set of labels (e.g parsing)
                    # should do same for logits
                    pass

                predictions_topk_dic = get_prediction(logits_dic, topk=topk)
                assert "normalize" not in args.tasks, "ERROR : following line () was needed apparently for normalize being supported"
                #output_tokens_tensor_aligned_dic = get_aligned_output(label_per_task)
                # for parsing heads will leave heads untouched
                source_preprocessed_dict, label_dic, predict_dic = get_bpe_string(predictions_topk_dic,
                                                                                 label_per_task,
                                                                                 input_tokens_tensor_per_task, topk, tokenizer,
                                                                                 task_to_label_dictionary, null_str,
                                                                                 null_token_index, verbose)

                # for parsing and tagging : will simply remove non-first bpe of each token
                src_detokenized_dic, label_detokenized_dic, predict_detokenize_dic = get_detokenized_str(source_preprocessed_dict,
                                                                                                         input_alignement_with_raw,
                                                                                                         label_dic,
                                                                                                         predict_dic,
                                                                                                         null_str,
                                                                                                         remove_mask_str_prediction,
                                                                                                         batch=batch)

                for label in label_detokenized_dic:
                    # TODO make more standart
                    if label in ["pos"] or label.startswith("parsing"):
                        src_detokenized = src_detokenized_dic["mwe_prediction"]
                    elif label in ["mwe_prediction"]:
                        src_detokenized = src_detokenized_dic["wordpieces_raw_aligned_with_words"]
                    elif label in ["n_masks_mwe", "mwe_detection"]:
                        src_detokenized = src_detokenized_dic["wordpieces_inputs_raw_tokens"]
                    else:
                        raise(Exception("label {} not found".format(label)))
                    perf_prediction, skipping, _samples = overall_word_level_metric_measure(task_label=label,
                                                                                            gold_sent_ls_dict=label_detokenized_dic,
                                                                                            pred_sent_ls_topk_dict=predict_detokenize_dic,
                                                                                            topk=topk,
                                                                                            metric=metric,
                                                                                            samples=samples,
                                                                                            agg_func_ls=agg_func_ls,
                                                                                            reference_word_dic=reference_word_dic,
                                                                                            compute_intersection_score=compute_intersection_score,
                                                                                            src_detokenized=src_detokenized)

                    printing("PREDICTION epoch {} task {} score all {}/{} total "
                             "gold {} gold token {} pred {} pred token {} ",
                             var=[epoch, label, perf_prediction["sum"]["all"]["score"], perf_prediction["sum"]["all"]["n_tokens"], label_detokenized_dic[label], label_per_task[label], predict_detokenize_dic[label], predictions_topk_dic[label][:, :, 0]],
                             verbose=verbose, verbose_level="pred")

                    score_dic[label], n_tokens_dic[label], n_sents_dic[label] = \
                        accumulate_scores_across_sents(agg_func_ls=agg_func_ls, sample_ls=_samples,
                                                       dic_prediction_score=perf_prediction,
                                                       score_dic=score_dic[label], n_tokens_dic=n_tokens_dic[label],
                                                       n_sents_dic=n_sents_dic[label])

                    evaluated_task.append(label)

                if writing_pred:
                    new_file = writing_predictions_conll_multi(
                                            dir_pred=dir_normalized,
                                            dir_normalized_original_only=dir_normalized_original_only,
                                            dir_gold=dir_gold, dir_gold_original_only=dir_gold_original_only,
                                            src_detokenized=src_detokenized_dic, pred_per_task=predict_detokenize_dic,
                                            iter=iter, batch_i=batch_i, new_file=new_file, gold_per_tasks=label_detokenized_dic,
                                            all_indexes=batch.all_indexes, task_parameters=TASKS_PARAMETER,
                                            tasks=args.tasks, verbose=verbose)

                _loss = get_multitask_loss(loss_dic, args.multi_task_loss_ponderation)
                loss_dic["all"] = _loss
                loss_dic_epoch = update_loss_dic_average(loss_dic, loss_dic_epoch)

                # temporary
                task_normalize_is = False
                task_pos_is = False
                # based on a policy : handle batch, epoch, batch weights, simultanuously
                # assert the policy is consistent with the available labels fed to the model
            # training :
            loss += _loss.detach()
            if optimizer is not None:
                _loss.backward()
                if (low_memory_foot_print_batch_mode and batch_i % batch_size_real == 0) or not low_memory_foot_print_batch_mode:
                    if low_memory_foot_print_batch_mode:
                        printing("OPTIMIZING in low_memory_foot_print_batch_mode cause batch index {}"
                                 " is batch_size_real",
                                 var=[batch_i, batch_size_real], verbose=verbose, verbose_level=1)
                    for opti in optimizer:
                        opti.step()
                        opti.zero_grad()
                    mode = "train"
                    printing("MODE data {} optimizing".format(data_label), verbose=verbose, verbose_level=4)
            else:
                mode = "dev"
                printing("MODE data {} not optimizing".format(data_label), verbose=verbose, verbose_level=4)
            if writer is not None:
                tensorboard_loss_writer_batch_level(writer, mode, model_id, _loss, batch_i, iter, loss_dic, task_normalize_is,  args.append_n_mask, task_pos_is)
                if args.multitask:
                    tensorboard_loss_writer_batch_level_multi(writer, mode, model_id, _loss, batch_i, iter, loss_dic, tasks=args.tasks)
        except StopIteration:
            printing("BREAKING ITERATION", verbose_level=1, verbose=1)
            printing("TIME : {:0.3f} min with / without {} n_masks predictions ", var=[mean_end_pred / batch_i, args.append_n_mask], verbose_level=1, verbose=verbose)
            break

    log_warning(counting_failure_parralel_bpe_batch, data_label, batch_i, batch, noisy_under_splitted, skipping_batch_n_to_1, aligned, noisy_over_splitted, skip_1_t_n, skipping_evaluated_batch, verbose)

    early_stoppin_metric_val = 999
    evaluated_task = list(set(evaluated_task))
    if predict_mode:
        if writer is not None:
            # n_tokens_counter_per_task
            if args.multitask:
                tensorboard_loss_writer_epoch_level_multi(writer,  mode, model_id, epoch, loss_dic_epoch, n_tokens_counter_per_task)
            tensorboard_loss_writer_epoch_level(writer, args.tasks, mode, model_id, epoch, n_batch_norm, n_batch_pos, args.append_n_mask, loss, loss_norm, loss_pos, loss_n_mask_prediction, batch_i)
        reports = []
        printing("TRAINING : evaluating on {} args.tasks ", var=[evaluated_task], verbose_level=1, verbose=verbose)
        reports, early_stoppin_metric_val, score, n_tokens = report_score_all(evaluated_task, agg_func_ls, samples_per_task_reporting,
                                                                              label_heuristic, score_dic, n_tokens_dic,
                                                                              n_sents_dic, model_id, args.tasks, args_dir,
                                                                              data_label, reports,  writer, log_perf,
                                                                              early_stoppin_metric_val, early_stoppin_metric,
                                                                              mode, subsample_early_stoping_metric_val,
                                                                              epoch)
    else:
        reports = None
    iter += batch_i
    if writing_pred:
        printing("DATA WRITTEN TO {} ", var=[dir_end_pred], verbose=verbose, verbose_level=1)
    printing("END EPOCH {} mode, iterated {} on pos {} on normalisation ", var=[mode, n_task_pos_sanity, n_task_normalize_sanity], verbose_level=1, verbose=verbose)
    try:
        if early_stoppin_metric is not None:
            assert early_stoppin_metric_val is not None, \
                "ERROR : early_stoppin_metric_val should have been found but was not {} sample metric {}" \
                " not found in {} (NB : MIGHT ALSO BECAUSE THE PERF DID NOT DECREASED AT ALL ) ".format(early_stoppin_metric, subsample_early_stoping_metric_val, reports)
    except Exception as e:
        print(e)
    if early_stoppin_metric_val is None:
        print("WARNING : early_stoppin_metric_val is None, score {} n_tokens {}".format(score, n_tokens))
    return loss/batch_i, iter, reports, early_stoppin_metric_val
