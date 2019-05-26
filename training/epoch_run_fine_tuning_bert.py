from env.importing import *
from env.project_variables import *
from io_.dat.constants import PAD_ID_BERT, MASK_BERT, CLS_BERT, SEP_BERT, SPECIAL_TOKEN_LS
from io_.info_print import printing
from io_.dat.normalized_writer import write_conll
from io_.bert_iterators_tools.string_processing import preprocess_batch_string_for_bert, from_bpe_token_to_str, get_indexes, get_indexes_src_gold
from io_.bert_iterators_tools.alignement import aligned_output, realigne

from evaluate.scoring.report import overall_word_level_metric_measure
from evaluate.scoring.confusion_matrix_rates import get_perf_rate

from toolbox.pred_tools.heuristics import predict_with_heuristic

from toolbox.deep_learning_toolbox import dropout_input_tensor
sys.path.insert(0, os.path.join(PROJECT_PATH, "..", "experimental_pipe"))
from reporting.write_to_performance_repo import report_template, write_dic


def accumulate_scores_across_sents(agg_func_ls, sample_ls, dic_prediction_score, score_dic, n_tokens_dic, n_sents_dic):
    for agg_func in agg_func_ls:
        for sample in sample_ls:
            score_dic[agg_func][sample] += dic_prediction_score[agg_func][sample]["score"]
            n_tokens_dic[agg_func][sample] += dic_prediction_score[agg_func][sample]["n_tokens"]
            n_sents_dic[agg_func][sample] += dic_prediction_score[agg_func][sample]["n_sents"]
    return score_dic, n_tokens_dic, n_sents_dic


def epoch_run(batchIter, tokenizer,
              iter, n_iter_max, bert_with_classifier, epoch,
              use_gpu, data_label, null_token_index, null_str,
              model_id, tasks, early_stoppin_metric=None,
              pos_dictionary=None,
              skip_1_t_n=True,
              writer=None, optimizer=None,
              predict_mode=False, topk=None, metric=None,
              print_pred=False, args_dir=None,
              heuristic_ls=None, gold_error_detection=False,
              reference_word_dic=None, dropout_input_bpe=0.,
              writing_pred=False, dir_end_pred=None, extra_label_for_prediction="",
              log_perf=True, masking_strategy=None, portion_mask=None, remove_mask_str_prediction=False,
              inverse_writing=False,
              norm_2_noise_eval=False, norm_2_noise_training=None, aggregating_bert_layer_mode="sum",
              compute_intersection_score = False,
              subsample_early_stoping_metric_val="all",
              slang_dic=None, list_reference_heuristic=None,list_candidates=None, index_alphabetical_order=None,
              case=None, threshold_edit=None, edit_module_pred_need_norm_only=True, low_memory_foot_print_batch_mode=False,
              batch_size_real=0, tokenize_and_bpe=False,
              verbose=0):
    """
    About Evaluation :
    Logic : compare gold and prediction topk using a word level scoring fucntion
            then accumulates for each sentences and foea each batch to get global score
            CAN add SAMPLE Parameter to get scores on specific subsample of the data : e.g. NEED_NORM, NORMED...
            Can also have different aggregation function
            TODO : TEST those scoring fucntions
    """
    if low_memory_foot_print_batch_mode:
        assert batch_size_real>0, "ERROR have to define batch_size_real in low_memory_foot_print_batch_mode"

    if heuristic_ls is not None:
        for edit_rule in ["all", "ref", "data"]:
            if "edit_check-"+edit_rule in heuristic_ls:
                assert threshold_edit is not None, "ERROR threshold_edit required as heuristic_ls is {}".format(heuristic_ls)
    if case is not None:
        AVAILABLE_CASE_OPTIONS = ["lower"]
        assert case in AVAILABLE_CASE_OPTIONS
    assert norm_2_noise_training is None or not norm_2_noise_eval, "only one of the two should be triggered but we" \
                                                                   " have norm_2_noise_training : {} norm_2_noise_" \
                                                                   "eval:{}".format(norm_2_noise_training,
                                                                                    norm_2_noise_eval)
    if norm_2_noise_training is not None:
        printing("WARNING : {} norm_2_noise_training is on ", var=[norm_2_noise_training],
                 verbose=verbose, verbose_level=1)
    if norm_2_noise_eval:
        printing("WARNING : {} norm_2_noise_eval is on ", var=[norm_2_noise_eval],
                 verbose=verbose, verbose_level=1)
    assert len(tasks) <= 2
    label_heuristic = ""
    if gold_error_detection:
        label_heuristic += "-gold"
    if heuristic_ls is not None:
        label_heuristic += "-"+"_".join(heuristic_ls)
    if norm_2_noise_eval:
        label_heuristic += "-noise_generation"
    print("HEURISTIC", heuristic_ls, label_heuristic)
    if masking_strategy is not None:
        if "start_stop" not in masking_strategy:
            assert "normalize" in tasks, "SO FAR : inconsistency between task {} and masking strategy {}".format(tasks, masking_strategy)
        if isinstance(masking_strategy, list):
            assert len(masking_strategy) <= 2, \
                "first element should be strategy, second should be portion or first element only ".format(masking_strategy)
            if len(masking_strategy) == 2:
                portion_mask = eval(str(masking_strategy[1]))
                masking_strategy = masking_strategy[0]
            else:
                masking_strategy = masking_strategy[0]
        assert masking_strategy in AVAILABLE_BERT_MASKING_STRATEGY, "masking_strategy {} should be in {}".format(AVAILABLE_BERT_MASKING_STRATEGY)
        if masking_strategy == "normed":
            printing("INFO : Portion mask was found to {}", var=[portion_mask], verbose=verbose, verbose_level=1)
    if predict_mode:
        if topk is None:
            topk = 1
            printing("PREDICITON MODE : setting topk to default 1 ", verbose_level=1, verbose=verbose)
        print_pred = False
        if metric is None:
            metric = "exact_match"
            printing("PREDICITON MODE : setting metric to default 'exact_match' ", verbose_level=1, verbose=verbose)

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
    printing("WARNING : [MASK] set to {} [CLS] {} [SEP] {}", var=[mask_token_index,cls_token_index, sep_token_index],
             verbose=verbose, verbose_level=1)

    batch_i = 0
    noisy_over_splitted = 0
    noisy_under_splitted = 0
    aligned = 0
    skipping_batch_n_to_1 = 0

    loss = 0
    samples = ["all", "NEED_NORM", "NORMED", "PRED_NEED_NORM", "PRED_NORMED", "InV", "OOV"]
    init_samples = samples.copy()
    if compute_intersection_score:
        for ind,sam in enumerate(samples[1:]):
            for ind_2 in range(ind):
                init_samples.append(sam+"-n-"+samples[ind_2+1])
    agg_func_ls = ["sum"]
    score_dic = {agg_func: {sample: 0 for sample in init_samples} for agg_func in agg_func_ls }
    n_tokens_dic = {agg_func: {sample: 0 for sample in init_samples} for agg_func in agg_func_ls}
    n_sents_dic = {agg_func: {sample: 0 for sample in init_samples} for agg_func in agg_func_ls}
    skipping_evaluated_batch = 0
    mode = "?"
    new_file = True

    loss_norm = 0
    loss_pos = 0
    n_batch_pos = 0
    n_batch_norm = 0
    n_task_pos_sanity = 0
    n_task_normalize_sanity = 0

    counting_failure_parralel_bpe_batch = 0
    while True:

        try:
            batch_i += 1

            batch = batchIter.__next__()
            # if no normalization found : should have pos
            task_pos_is = len(batch.raw_output[0]) == 0
            task_normalize_is = not task_pos_is
            if case is not None:
                if case == "lower":
                    batch.raw_input = [[word.lower() if word not in SPECIAL_TOKEN_LS else word for word in sent] for sent in batch.raw_input]
                    if task_normalize_is:
                        batch.raw_output = [[word.lower() if word not in SPECIAL_TOKEN_LS else word for word in sent] for sent in batch.raw_output]
            #print("ITERATING on {} task".format("pos" if task_pos_is else "normalize"))
            n_task_pos_sanity += int(task_pos_is)
            n_task_normalize_sanity += int(task_normalize_is)
            norm2noise_bool = False
            batch_raw_output = None
            # Handling input
            if (norm_2_noise_training is not None or norm_2_noise_eval) and task_normalize_is:
                portion_norm2noise = norm_2_noise_training if norm_2_noise_training is not None else 1.
                norm_2_noise_training = portion_norm2noise is not None
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

            if masking_strategy == "cls":
                # we trick batch.output_norm_not_norm : set all 1 to 0 (not to touch padding)
                # we set first element to 1
                batch.output_norm_not_norm[batch.output_norm_not_norm == 1] = 0
                batch.output_norm_not_norm[:, 0] = 1
                group_to_mask = batch.output_norm_not_norm
            elif masking_strategy == "normed":
                rand = np.random.uniform(low=0, high=1, size=1)[0]
                group_to_mask = np.array(batch.output_norm_not_norm.cpu()) if portion_mask >= rand else None
            if not tokenize_and_bpe:
                input_tokens_tensor, input_segments_tensors, inp_bpe_tokenized, input_alignement_with_raw, input_mask = \
                get_indexes(batch_raw_input, tokenizer, verbose, use_gpu, word_norm_not_norm=group_to_mask)
            if masking_strategy == "start_stop":
                input_mask[input_tokens_tensor == sep_token_index] = 0
                input_mask[input_tokens_tensor == cls_token_index] = 0
            #if "normalize" in tasks:
            if task_normalize_is:
                if norm2noise_bool or norm_2_noise_eval:
                    printing("WARNING : output is noisy innput", verbose_level=2, verbose=1)
                    batch_raw_output = preprocess_batch_string_for_bert(batch.raw_input)
                else:
                    printing("WARNING : output is output", verbose_level=2, verbose=1)
                    batch_raw_output = preprocess_batch_string_for_bert(batch.raw_output, rp_space=True)

                if tokenize_and_bpe:
                    try:
                        tokens_tensor_dic, segments_tensors_dic, tokenized_dic, aligned_index_padded_dic, mask_dic = \
                            get_indexes_src_gold(list_pretokenized_str_source=batch_raw_input,
                                             list_pretokenized_str_gold=batch_raw_output,
                                             tokenizer=tokenizer,
                                             verbose=verbose, use_gpu= use_gpu)

                        output_tokens_tensor, output_segments_tensors, out_bpe_tokenized, output_alignement_with_raw, output_mask = \
                            tokens_tensor_dic["gold"], segments_tensors_dic["gold"], tokenized_dic["gold"], \
                            aligned_index_padded_dic["gold"], mask_dic["gold"]

                        input_tokens_tensor, input_segments_tensors, inp_bpe_tokenized, input_alignement_with_raw, input_mask = \
                            tokens_tensor_dic["src"], segments_tensors_dic["src"], tokenized_dic["src"], \
                            aligned_index_padded_dic["src"], mask_dic["src"]

                    except Exception as e:
                        print("FAILLING error {} TO ALIGN batch_raw_input {} with batch_raw_output {} so using the old method".format(e, batch_raw_input, batch_raw_output))
                        input_tokens_tensor, input_segments_tensors, inp_bpe_tokenized, input_alignement_with_raw, input_mask = \
                            get_indexes(batch_raw_input, tokenizer, verbose, use_gpu, word_norm_not_norm=group_to_mask)
                        output_tokens_tensor, output_segments_tensors, out_bpe_tokenized, output_alignement_with_raw, output_mask = \
                            get_indexes(batch_raw_output, tokenizer, verbose, use_gpu)
                        counting_failure_parralel_bpe_batch += 1

                else:
                    output_tokens_tensor, output_segments_tensors, out_bpe_tokenized, output_alignement_with_raw, output_mask =\
                    get_indexes(batch_raw_output, tokenizer, verbose, use_gpu)
                printing("DATA dim : {} input {} output ", var=[input_tokens_tensor.size(), output_tokens_tensor.size()],
                         verbose_level=2, verbose=verbose)
            #elif "pos" in tasks:
            if task_pos_is:
                # TODO : factorize all this
                #  should be done in pytorch + reducancies with get_index + factorize is somewhwere
                output_tokens_tensor = np.array(batch.pos.cpu())
                out_bpe_tokenized = None
                #inde, _ = torch.min((torch.Tensor(input_alignement_with_raw) == 2).nonzero()[:, 1], dim=0)
                new_input = np.array(input_tokens_tensor.cpu())
                #new_input = [[input_tokens_tensor[ind_sent, ind] for ind in range(len(sent)) if sent[ind] != sent[ind - 1]] for ind_sent, sent in enumerate(input_alignement_with_raw)]

                len_max = max([len(sent) for sent in new_input])
                new_input = [[inp for inp in sent]+[PAD_ID_BERT for _ in range(len_max-len(sent))] for sent in new_input]
                #_input_mask = [[1 if input != PAD_ID_BERT else 0 for input in inp] for inp in zip(new_input)]
                # we mask bpe token that have been split (we don't mask the first bpe token of each word)
                _input_mask = [[0 if new_input[ind_sent][ind_tok] == PAD_ID_BERT
                                     or input_alignement_with_raw[ind_sent][ind_tok-1] == input_alignement_with_raw[ind_sent][ind_tok]
                                else 1 for ind_tok in range(len(new_input[ind_sent]))]
                               for ind_sent in range(len(new_input))]
                output_tokens_tensor_new = []
                for ind_sent in range(len(_input_mask)):
                    output_tokens_tensor_new_ls = []
                    shift = 0
                    for ind_tok in range(len(_input_mask[ind_sent])):
                        mask = _input_mask[ind_sent][ind_tok]
                        try:
                            label = output_tokens_tensor[ind_sent, ind_tok-shift]
                        except Exception as e:
                            print("ERROR ind_send:{} ind_tok {} shift {} output_tokens_tensor {} {}".format(ind_sent, ind_tok, shift, output_tokens_tensor, e))
                            print("ERROR ind_send ", batch.raw_input, batch.raw_output)
                            label = output_tokens_tensor[ind_sent, output_tokens_tensor.shape[1]-1]

                        if mask != 0:
                            output_tokens_tensor_new_ls.append(label)
                        else:
                            # 1 for _PAD_POS
                            output_tokens_tensor_new_ls.append(1)
                            shift += 1
                    output_tokens_tensor_new.append(output_tokens_tensor_new_ls)

                output_tokens_tensor = torch.Tensor(output_tokens_tensor_new).long()
                input_mask = torch.Tensor(_input_mask).long()
                input_tokens_tensor = torch.Tensor(new_input).long()

                if use_gpu:
                    input_mask = input_mask.cuda()
                    output_tokens_tensor = output_tokens_tensor.cuda()
                    input_tokens_tensor = input_tokens_tensor.cuda()
            _verbose = verbose

            # logging
            printing("DATA : pre-tokenized input {} ", var=[batch_raw_input], verbose_level="raw_data",
                     verbose=_verbose)
            printing("DATA : BPEtokenized input ids {}", var=[input_tokens_tensor], verbose_level=3,
                     verbose=verbose)

            printing("DATA : pre-tokenized output {} ", var=[batch_raw_output],
                     verbose_level="raw_data",
                     verbose=_verbose)
            printing("DATA : BPE tokenized output ids  {}", var=[output_tokens_tensor],
                     verbose_level=4,
                     verbose=verbose)
            # BPE
            printing("DATA : BPE tokenized input  {}", var=[inp_bpe_tokenized], verbose_level=4,
                     verbose=_verbose)
            printing("DATA : BPE tokenized output  {}", var=[out_bpe_tokenized], verbose_level=4,
                     verbose=_verbose)
            _1_to_n_token = 0
            #if "normalize" in tasks:
            if task_normalize_is:
                # aligning output BPE with input (we are rejecting batch with at least one 1 to n case
                # (that we don't want to handle

                output_tokens_tensor_aligned, input_tokens_tensor_aligned, input_alignement_with_raw, input_mask, _1_to_n_token = \
                    aligned_output(input_tokens_tensor, output_tokens_tensor,
                                   input_alignement_with_raw,
                                   output_alignement_with_raw, mask_token_index=mask_token_index,
                                   input_mask=input_mask, use_gpu=use_gpu,
                                   null_token_index=null_token_index, verbose=verbose)
                input_tokens_tensor = input_tokens_tensor_aligned
                #
                #TODO : creaate a tensor same dim as output_tokens_tensor based on output_alignement_with_raw
                # number of repetition in output_alignement_with_raw
                # or number of bpe tokens related to each bpe
            #elif "pos" in tasks:
            elif task_pos_is:
                # NB : we use the aligned input with the
                output_tokens_tensor_aligned = output_tokens_tensor[:, : input_tokens_tensor.size(1)]
                output_tokens_tensor_aligned = output_tokens_tensor_aligned.contiguous()
                if use_gpu:
                    output_tokens_tensor_aligned = output_tokens_tensor_aligned.cuda()
                    #segments_ids = [[0 for _ in range(len(tokenized))] for tokenized in tokenized_ls]
            #mask = [[1 for _ in inp] + [0 for _ in range(max_sent_len - len(inp))] for inp in segments_ids]

            if batch_i == n_iter_max:
                break
            if _1_to_n_token:
                skipping_batch_n_to_1 += _1_to_n_token
                #continue
            # CHECKING ALIGNEMENT
            # PADDING TO HANDLE !!
            assert output_tokens_tensor_aligned.size(0) == input_tokens_tensor.size(0),\
                "output_tokens_tensor_aligned.size(0) {} input_tokens_tensor.size() {}".format(output_tokens_tensor_aligned.size(), input_tokens_tensor.size())
            assert output_tokens_tensor_aligned.size(1) == input_tokens_tensor.size(1), \
                "output_tokens_tensor_aligned.size(1) {} input_tokens_tensor.size() {}".format(output_tokens_tensor_aligned.size(1),
                                                                                               input_tokens_tensor.size(1))
            # we consider only 1 sentence case
            token_type_ids = torch.zeros_like(input_tokens_tensor)
            if use_gpu:
                token_type_ids = token_type_ids.cuda()
            printing("CUDA SANITY CHECK input_tokens:{}  type:{}input_mask:{}  label:{}",
                     var=[input_tokens_tensor.is_cuda, token_type_ids.is_cuda, input_mask.is_cuda,
                          output_tokens_tensor_aligned.is_cuda],
                     verbose=verbose, verbose_level="cuda")
            # we have to recompute the mask based on aligned input
            if dropout_input_bpe > 0:
                input_tokens_tensor = dropout_input_tensor(input_tokens_tensor, mask_token_index,
                                                           sep_token_index=sep_token_index,
                                                           dropout=dropout_input_bpe)
            try:
                printing("MASK mask:{}\nMASK input:{}\nMASK output:{}", var=[input_mask, input_tokens_tensor,
                                                                         output_tokens_tensor_aligned],
                         verbose_level="raw_data", verbose=verbose)
                loss_dic = bert_with_classifier(input_tokens_tensor, token_type_ids, input_mask,
                                                labels=output_tokens_tensor_aligned if task_normalize_is else None, #tasks[0] == "normalize" else None,
                                                labels_task_2=output_tokens_tensor_aligned if task_pos_is else None, #tasks[0] == "pos" else None
                                                aggregating_bert_layer_mode=aggregating_bert_layer_mode)
            except Exception as e:
                print(e)
                print(" MAX ", torch.max(output_tokens_tensor_aligned), input_tokens_tensor, input_mask)
                loss_dic = bert_with_classifier(input_tokens_tensor, token_type_ids, input_mask,
                                                aggregating_bert_layer_mode=aggregating_bert_layer_mode,
                                                labels=output_tokens_tensor_aligned if task_normalize_is else None,#if tasks[0] == "normalize" else None,
                                                labels_task_2=output_tokens_tensor_aligned if task_pos_is else None)#if tasks[0] == "pos" else None)
            _loss = loss_dic["loss"]
            if task_normalize_is:
                loss_norm += loss_dic["loss_task_1"].detach()
                n_batch_norm += 1
            if task_pos_is:
                loss_pos += loss_dic["loss_task_2"].detach()
                n_batch_pos += 1
            if predict_mode:
                # if predict more : will evaluate the model and write its predictions
                # TODO : add mapping_info between task_id to model and task name necessary to iterator
                logits = bert_with_classifier(input_tokens_tensor, token_type_ids, input_mask,
                                              aggregating_bert_layer_mode=aggregating_bert_layer_mode,
                                              )["logits_task_2" if task_pos_is else "logits_task_1"]
                predictions_topk = torch.argsort(logits, dim=-1, descending=True)[:, :, :topk]
                # from bpe index to string
                sent_ls_top = from_bpe_token_to_str(predictions_topk, topk, tokenizer=tokenizer, pred_mode=True,
                                                    pos_dictionary=pos_dictionary, task=tasks[0],
                                                    null_token_index=null_token_index, null_str=null_str)
                gold = from_bpe_token_to_str(output_tokens_tensor_aligned, topk, tokenizer=tokenizer,
                                             pos_dictionary=pos_dictionary, task=tasks[0],
                                             pred_mode=False, null_token_index=null_token_index, null_str=null_str)

                source_preprocessed = from_bpe_token_to_str(input_tokens_tensor, topk, tokenizer=tokenizer,
                                                            pos_dictionary=pos_dictionary,
                                                            pred_mode=False, null_token_index=null_token_index,
                                                            null_str=null_str, verbose=verbose)

                # de-BPE-tokenize
                src_detokenized = realigne(source_preprocessed, input_alignement_with_raw, null_str=null_str,
                                           tasks=["normalize"],# normalize means we deal wiht bpe input not pos
                                           mask_str=MASK_BERT, remove_mask_str=remove_mask_str_prediction)
                gold_detokenized = realigne(gold, input_alignement_with_raw, remove_null_str=True, null_str=null_str,
                                            tasks=tasks,
                                            mask_str=MASK_BERT)
                if task_pos_is:
                    # we remove padding here based on src that is corectly padded
                    gold_detokenized = [gold_sent[:len(src_sent)] for gold_sent, src_sent in zip(gold_detokenized, src_detokenized)]
                pred_detokenized_topk = []
                for sent_ls in sent_ls_top:
                    pred_detokenized_topk.append(realigne(sent_ls, input_alignement_with_raw, remove_null_str=True,
                                                          tasks=tasks, remove_extra_predicted_token=True,
                                                          null_str=null_str, mask_str=MASK_BERT))
                    # NB : applying those successively might overlay heuristic
                    if task_normalize_is:
                        if heuristic_ls is not None:
                            # NB : if the rules in heuristic_ls are not exclusive their order matters !!
                            # the last one will be the one that is applied
                            pred_detokenized_topk = predict_with_heuristic(src_detokenized=src_detokenized,
                                                                           pred_detokenized_topk=pred_detokenized_topk,
                                                                           list_reference=list_reference_heuristic, list_candidates=list_candidates,
                                                                           slang_dic=slang_dic,
                                                                           index_alphabetical_order=index_alphabetical_order,
                                                                           heuristic_ls=heuristic_ls,
                                                                           threshold_edit=threshold_edit,
                                                                           edit_module_pred_need_norm_only=edit_module_pred_need_norm_only,
                                                                           verbose=verbose)
                        # NB : we overlay prediction with gold_error_detection
                        if gold_error_detection:
                            pred_detokenized_topk = predict_with_heuristic(src_detokenized=src_detokenized,
                                                                           gold_detokenized=gold_detokenized,
                                                                           pred_detokenized_topk=pred_detokenized_topk,
                                                                           heuristic_ls=["gold_detection"], verbose=verbose)
                            #print("PRED after gold",gold_detokenized, pred_detokenized_topk)
                if writing_pred:
                    # TODO : if you do multitask leaning
                    #  you'll have to adapt here (you're passing twice the same parameters)
                    write_conll(format="conll", dir_normalized=dir_normalized,
                                dir_original=dir_normalized_original_only,
                                src_text_ls=src_detokenized, inverse=inverse_writing,
                                text_decoded_ls=pred_detokenized_topk[0], #pred_pos_ls=None, src_text_pos=None,
                                tasks=["pos" if task_pos_is else "normalize"], ind_batch=iter+batch_i, new_file=new_file,
                                src_text_pos=src_detokenized, pred_pos_ls=gold_detokenized,
                                verbose=verbose)
                    write_conll(format="conll", dir_normalized=dir_gold, dir_original=dir_gold_original_only,
                                src_text_ls=src_detokenized,
                                src_text_pos=src_detokenized, pred_pos_ls=gold_detokenized,
                                text_decoded_ls=gold_detokenized, #pred_pos_ls=None, src_text_pos=None,
                                tasks=["pos" if task_pos_is else "normalize"],
                                ind_batch=iter + batch_i, new_file=new_file, verbose=verbose)
                    new_file = False
                perf_prediction, skipping, _samples = overall_word_level_metric_measure(gold_detokenized, pred_detokenized_topk,
                                                                                        topk,
                                                                                        metric=metric,
                                                                                        samples=samples,
                                                                                        agg_func_ls=agg_func_ls,
                                                                                        reference_word_dic=reference_word_dic,
                                                                                        compute_intersection_score=compute_intersection_score,
                                                                                        src_detokenized=src_detokenized)

                score_dic, n_tokens_dic, n_sents_dic = accumulate_scores_across_sents(agg_func_ls=agg_func_ls,
                                                                                      sample_ls=_samples,
                                                                                      dic_prediction_score=perf_prediction,
                                                                                      score_dic=score_dic,
                                                                                      n_tokens_dic=n_tokens_dic,
                                                                                      n_sents_dic=n_sents_dic)

                skipping_evaluated_batch += skipping

                if print_pred:
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

                def print_align_bpe(source_preprocessed, gold, input_alignement_with_raw, verbose,verbose_level):
                    if isinstance(verbose, int) or verbose == "alignement":
                        if verbose == "alignement" or verbose >= verbose_level:
                            assert len(source_preprocessed) == len(gold), ""
                            assert len(input_alignement_with_raw) == len(gold), ""
                            for sent_src, sent_gold, index_match_with_src in zip(source_preprocessed, gold, input_alignement_with_raw):
                                assert len(sent_src) == len(sent_gold)
                                assert len(sent_src) == len(sent_gold)
                                for src, gold_tok, index in zip(sent_src, sent_gold, index_match_with_src):
                                    printing("{}:{} --> {} ", var=[index, src, gold_tok],
                                             verbose=1, verbose_level=1)
                                    #printing("{}:{} --> {} ", var=[index, src, gold_tok],
                                    #         verbose=verbose, verbose_level=verbose_level)

                print_align_bpe(source_preprocessed, gold, input_alignement_with_raw, verbose=verbose, verbose_level=4)

            loss += _loss.detach()

            if optimizer is not None:
                _loss.backward()
                if (low_memory_foot_print_batch_mode and batch_i % batch_size_real==0) or not low_memory_foot_print_batch_mode:
                    if low_memory_foot_print_batch_mode:
                        printing("OPTIMIZING in low_memory_foot_print_batch_mode cause batch index {} is batch_size_real",
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
                writer.add_scalars("loss-alteranate",
                                   {"loss-{}-{}-bpe".format(mode, model_id): _loss.clone().cpu().data.numpy()
                                   if not isinstance(_loss, int) else 0},
                                   iter+batch_i)
                if task_pos_is:
                    writer.add_scalars("loss-pos",
                                       {"loss-{}-{}-bpe".format(mode, model_id): loss_dic["loss_task_2"].detach().clone().cpu().data.numpy()
                                    },
                                       iter + batch_i)
                if task_normalize_is:
                    writer.add_scalars("loss-norm",
                                       {"loss-{}-{}-bpe".format(mode, model_id): loss_dic["loss_task_1"].detach().clone().cpu().data.numpy()},
                                       iter + batch_i)
        except StopIteration:
            break
    printing("WARNING {} aignement failure caused by parallel ", var=[counting_failure_parralel_bpe_batch], verbose=verbose, verbose_level=1)
    printing("WARNING on {} : Out of {} batch of {} sentences each {} skipped ({} batch aligned ; "
             "{} with at least 1 sentence "
             "noisy MORE SPLITTED "
             "; {} with  LESS SPLITTED {} + SENT with skipped_1_to_n : {}) ",
             var=[data_label, batch_i, batch.input_seq.size(0), noisy_under_splitted+skipping_batch_n_to_1, aligned,
                  noisy_over_splitted, noisy_under_splitted,
                  "SKIPPED" if skip_1_t_n else "",
                  skipping_batch_n_to_1],
             verbose=verbose, verbose_level=0)
    printing("WARNING on {} ON THE EVALUATION SIDE we skipped extra {} batch ", var=[data_label, skipping_evaluated_batch], verbose_level=1, verbose=1)

    early_stoppin_metric_val = None
    samples = _samples
    print("CHECKING SAMPLES", _samples)
    if predict_mode:
        if writer is not None:
            writer.add_scalars("loss-overall-mean-{}-{}".format(tasks[0], mode),
                           {"{}-{}-{}".format("loss", mode, model_id): loss/batch_i
                            }, epoch)
            if "normalize" in tasks:
                try:
                    writer.add_scalars("loss-norm",
                               {"loss-{}-{}-bpe".format(mode, model_id): loss_norm.clone().cpu().data.numpy()/n_batch_norm},
                               epoch)
                except Exception as e:
                    print("ERROR {} loss_pos is , n_batch_pos is {} coud not log ".format(e, loss_norm, n_batch_norm))
            if "pos" in tasks:
                try:
                    writer.add_scalars("loss-pos",
                               {"loss-{}-{}-bpe".format(mode, model_id): loss_pos.clone().cpu().data.numpy()/n_batch_pos},
                               epoch)
                except Exception as e:
                    print("ERROR {} loss_pos is , n_batch_pos is {} coud not log ".format(e, loss_pos, n_batch_pos))

        reports = []
        for agg_func in agg_func_ls:
            for sample in samples:
                print("sample", sample)
                # for binary classification : having 3 samples define [class Positive, class Negative, All]
                #  e.g [NORMED, NEED_NORM , all] for a given agg_func
                # TP : score_dic[agg_func][Positive Class]
                # TN : score_dic[agg_func][Negative Class]
                # P observations = n_tokens_dic[agg_func][Positive Class]
                # N observations  = n_tokens_dic[agg_func][Negative Class]
                # PP predictions = FP + TP = (N-TN) + TP
                # NP predictions = FN + TN = (P-TP) + TN
                # recall = TP/P , precision = TP/PP,  tnr = TN/N , npr = TN/NP
                # f1 = hmean(recall, precision) , accuracy = (TN+TP)/(N+P)
                score = score_dic[agg_func][sample]
                n_tokens = n_tokens_dic[agg_func][sample]
                n_sents = n_sents_dic[agg_func][sample]
                metric_val = "accuracy-exact-{}".format(tasks[1] if len(tasks)>1 else tasks[0])
                report = report_template(metric_val=metric_val, subsample=sample+label_heuristic, info_score_val=None,
                                         score_val=score/n_tokens if n_tokens > 0 else None,
                                         n_sents=n_sents,
                                         avg_per_sent=0,
                                         n_tokens_score=n_tokens,
                                         model_full_name_val=model_id, task=tasks,
                                         evaluation_script_val="exact_match",
                                         model_args_dir=args_dir,
                                         token_type="word",
                                         report_path_val=None,
                                         data_val=data_label)

                if early_stoppin_metric is not None:
                    if metric_val == early_stoppin_metric and subsample_early_stoping_metric_val == sample+label_heuristic:
                        early_stoppin_metric_val = -score/n_tokens
                if writer is not None and log_perf:
                    writer.add_scalars("perf-{}-{}".format(tasks[0], mode),
                                       {"{}-{}-{}-{}".format(metric_val, mode, model_id, sample):
                                            score/n_tokens if n_tokens>0 and score is not None else 0
                                        }, epoch)
                reports.append(report)
            # class negative 0 , class positive 1
            # TODO : make that more consistent with user needs !
            if "normalize" in tasks:
                if "all" in samples and TASKS_PARAMETER["normalize"]["predicted_classes"][0] in samples \
                        and TASKS_PARAMETER["normalize"]["predicted_classes"][1] in samples:

                    # then we can compute all the confusion matrix rate
                    # TODO : factore with TASKS_2_METRICS_STR

                    for metric_val in ["recall", "precision", "f1", "tnr", "npv", "accuracy"]:
                        metric_val += "-"+tasks[0]
                        score, n_rate_universe = get_perf_rate(metric=metric_val, n_tokens_dic=n_tokens_dic,
                                                               score_dic=score_dic, agg_func=agg_func)

                        report = report_template(metric_val=metric_val, subsample="rates"+label_heuristic,
                                                 info_score_val=None,
                                                 score_val=score, n_sents=n_sents_dic[agg_func]["all"],
                                                 avg_per_sent=0,
                                                 n_tokens_score=n_rate_universe,
                                                 model_full_name_val=model_id, task=tasks,
                                                 evaluation_script_val="exact_match",
                                                 model_args_dir=args_dir,
                                                 token_type="word",
                                                 report_path_val=None,
                                                 data_val=data_label)
                        if early_stoppin_metric is not None:
                            if metric_val == early_stoppin_metric and subsample_early_stoping_metric_val == "rates"+label_heuristic:
                                early_stoppin_metric_val = -score
                        reports.append(report)

                        if writer is not None and log_perf:
                            writer.add_scalars("perf-{}-{}".format(tasks[0], mode),
                                               {"{}-{}-{}-bpe".format(metric_val, mode, model_id):
                                                    score if score is not None else 0
                                                }, epoch)

    else:
        reports = None
    iter += batch_i

    if writing_pred:
        printing("DATA WRITTEN TO {} ", var=[dir_end_pred], verbose=verbose, verbose_level=1)
    printing("END EPOCH {} mode, iterated {} on pos {} on normalisation ",
             var=[mode, n_task_pos_sanity, n_task_normalize_sanity], verbose_level=1, verbose=verbose)
    if early_stoppin_metric is not None:
        assert early_stoppin_metric_val is not None, "ERROR : early_stoppin_metric_val should have been found " \
                                                     "but was not {} sample metric {} not found in {}  ".format(early_stoppin_metric, subsample_early_stoping_metric_val, reports)
    return loss/batch_i, iter, reports, early_stoppin_metric_val
