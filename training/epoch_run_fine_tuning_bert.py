from env.importing import *
from env.project_variables import *
from io_.dat.constants import PAD_ID_BERT, MASK_BERT
from io_.info_print import printing
from io_.dat.normalized_writer import write_conll
from io_.bert_iterators_tools.string_processing import preprocess_batch_string_for_bert, from_bpe_token_to_str, get_indexes
from io_.bert_iterators_tools.alignement import aligned_output, realigne

from evaluate.scoring.report import overall_word_level_metric_measure
from evaluate.scoring.confusion_matrix_rates import get_perf_rate

from toolbox.predictions.heuristics import predict_with_heuristic
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
              model_id, tasks,
              pos_dictionary=None,
              skip_1_t_n=True,
              writer=None, optimizer=None,
              predict_mode=False, topk=None, metric=None,
              print_pred=False, args_dir=None,
              heuristic_ls=None, gold_error_detection=False,
              reference_word_dic=None, dropout_input_bpe=0.,
              writing_pred=False, dir_end_pred=None, extra_label_for_prediction="",
              log_perf=True, masking_strategy=None, portion_mask=None,
              norm_2_noise_eval=False,  norm_2_noise_training=None,
              verbose=0):
    """
    About Evaluation :
    Logic : compare gold and prediction topk using a word level scoring fucntion
            then accumulates for each sentences and foea each batch to get global score
            CAN add SAMPLE Parameter to get scores on specific subsample of the data : e.g. NEED_NORM, NORMED...
            Can also have different aggregation function
            TODO : TEST those scoring fucntions
    :param batchIter:
    :param tokenizer:
    :param iter:
    :param n_iter_max:
    :param bert_with_classifier:
    :param epoch:
    :param use_gpu:
    :param data_label:
    :param null_token_index:
    :param null_str:
    :param model_id:
    :param skip_1_t_n:
    :param writer:
    :param optimizer:
    :param predict_mode:
    :param topk:
    :param metric:
    :param print_pred:
    :param args_dir:
    :param writing_pred:
    :param dir_end_pred:
    :param extra_label_for_prediction:
    :param verbose:
    :return:
    """
    assert len(tasks) == 1, "only one task supported so far"
    assert norm_2_noise_training is None or not norm_2_noise_eval, "only one of the two should be triggered but we" \
                                                                   " have norm_2_noise_training : {} norm_2_noise_" \
                                                                   "eval:{}".format(norm_2_noise_training,
                                                                                    norm_2_noise_eval)
    if masking_strategy is not None:
        assert "normalize" in tasks, "SO FAR : inconsistency between task {} and masking strategy {}".format(tasks,
                                                                                                    masking_strategy)
        if isinstance(masking_strategy, list):
            assert len(masking_strategy) <= 2, \
                "first element should be strategy, second should be portion or first element only ".format(masking_strategy)
            if len(masking_strategy) == 2:
                portion_mask = eval(str(masking_strategy[1]))
                masking_strategy = masking_strategy[0]
            else:
                masking_strategy = masking_strategy[0]
        assert masking_strategy in AVAILABLE_BERT_MASKING_STRATEGY , "masking_strategy {} should be in {}".format(AVAILABLE_BERT_MASKING_STRATEGY)
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
        dir_normalized = os.path.join(dir_end_pred, "{}_ep-prediction{}.conll".format(epoch,
                                                                                      extra_label_for_prediction))
        dir_normalized_original_only = os.path.join(dir_end_pred, "{}_ep-prediction_src{}.conll".format(epoch,
                                                                                                        extra_label_for_prediction))
        dir_gold = os.path.join(dir_end_pred, "{}_ep-gold.conll{}".format(epoch,
                                                                          extra_label_for_prediction))
        dir_gold_original_only = os.path.join(dir_end_pred, "{}_ep-gold_src{}.conll".format(epoch,
                                                                                            extra_label_for_prediction))

    mask_token_index = tokenizer.convert_tokens_to_ids([MASK_BERT])[0]
    printing("WARNING : [MASK] set to {}", var=[mask_token_index],
             verbose=verbose, verbose_level=1)

    batch_i = 0
    noisy_over_splitted = 0
    noisy_under_splitted = 0
    aligned = 0
    skipping_batch_n_to_1 = 0

    loss = 0
    samples = ["all", "NEED_NORM", "NORMED", "InV", "OOV"]
    agg_func_ls = ["sum"]
    score_dic = {agg_func: {sample: 0 for sample in samples} for agg_func in agg_func_ls }
    n_tokens_dic = {agg_func: {sample: 0 for sample in samples} for agg_func in agg_func_ls}
    n_sents_dic = {agg_func: {sample: 0 for sample in samples} for agg_func in agg_func_ls}
    skipping_evaluated_batch = 0
    mode = "?"
    new_file = True

    while True:

        try:
            batch_i += 1

            batch = batchIter.__next__()
            norm2noise_bool = False

            if norm_2_noise_training is not None or norm_2_noise_eval:
                portion_norm2noise = norm_2_noise_training
                norm_2_noise_training = portion_norm2noise is not None
                rand = np.random.uniform(low=0, high=1, size=1)[0]
                norm2noise_bool = portion_norm2noise >= rand
                if norm2noise_bool or norm_2_noise_eval:
                    batch_raw_input = preprocess_batch_string_for_bert(batch.raw_output)
                    print("WARNING : input is gold norm")
                else:
                    print("WARNING : input is input")
                    batch_raw_input = preprocess_batch_string_for_bert(batch.raw_input)
            else:
                batch_raw_input = preprocess_batch_string_for_bert(batch.raw_input)

            if masking_strategy is None:
                group_to_mask = None
            elif masking_strategy == "cls":
                # we trick batch.output_norm_not_norm : set all 1 to 0 (not to touch padding)
                # we set first element to 1
                batch.output_norm_not_norm[batch.output_norm_not_norm == 1] = 0
                batch.output_norm_not_norm[:, 0] = 1
                group_to_mask = batch.output_norm_not_norm
            elif masking_strategy == "normed":
                rand = np.random.uniform(low=0, high=1, size=1)[0]
                group_to_mask = np.array(batch.output_norm_not_norm.cpu()) if portion_mask >= rand else None

            input_tokens_tensor, input_segments_tensors, inp_bpe_tokenized, input_alignement_with_raw, input_mask = \
                get_indexes(batch_raw_input, tokenizer, verbose, use_gpu,
                            word_norm_not_norm=group_to_mask)
            pdb.set_trace()
            if "normalize" in tasks:
                if norm2noise_bool or norm_2_noise_eval:
                    print("WARNING : output is gold norm")
                    pdb.set_trace()
                    batch_raw_output = preprocess_batch_string_for_bert(batch.raw_input)
                else:
                    print("WARNING : output is output")
                    batch_raw_output = preprocess_batch_string_for_bert(batch.raw_output, rp_space=True)
                output_tokens_tensor, output_segments_tensors, out_bpe_tokenized, output_alignement_with_raw, output_mask =\
                    get_indexes(batch_raw_output, tokenizer, verbose, use_gpu)
                printing("DATA dim : {} input {} output ", var=[input_tokens_tensor.size(), output_tokens_tensor.size()],
                         verbose_level=2, verbose=verbose)
            elif "pos" in tasks:
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
                        label = output_tokens_tensor[ind_sent, ind_tok-shift]
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
            if "normalize" in tasks:
                # aligning output BPE with input (we are rejecting batch with at least one 1 to n case
                # (that we don't want to handle

                output_tokens_tensor_aligned, input_tokens_tensor_aligned, input_alignement_with_raw, input_mask, _1_to_n_token = \
                    aligned_output(input_tokens_tensor, output_tokens_tensor,
                                   input_alignement_with_raw,
                                   output_alignement_with_raw, mask_token_index=mask_token_index,
                                   input_mask=input_mask,use_gpu=use_gpu,
                                   null_token_index=null_token_index, verbose=verbose)
                input_tokens_tensor = input_tokens_tensor_aligned
            elif "pos" in tasks:
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
                "output_tokens_tensor_aligned.size(0) {} input_tokens_tensor.size() {}".format(output_tokens_tensor_aligned.size(),
                                                                                               input_tokens_tensor.size())
            assert output_tokens_tensor_aligned.size(1) == input_tokens_tensor.size(1), \
                "output_tokens_tensor_aligned.size(1) {} input_tokens_tensor.size() {}".format(output_tokens_tensor_aligned.size(1),
                                                                                               input_tokens_tensor.size(1))
            # we consider only 1 sentence case
            token_type_ids = torch.zeros_like(input_tokens_tensor)
            if use_gpu:
                token_type_ids = token_type_ids.cuda()
            printing("CUDA SANITY CHECK input_tokens:{}  type:{} input_mask:{}  label:{}",
                     var=[input_tokens_tensor.is_cuda, token_type_ids.is_cuda, input_mask.is_cuda,
                          output_tokens_tensor_aligned.is_cuda],
                     verbose=verbose, verbose_level="cuda")
            # we have to recompute the mask based on aligned input
            if "normalize" in tasks:
                pass
                #input_mask = torch.Tensor([[1 if token_id != PAD_ID_BERT else 0 for token_id in sent_token]
                #                           for sent_token in input_tokens_tensor]).long()
                #if input_tokens_tensor.is_cuda:
                #    input_mask = input_mask.cuda()
            if dropout_input_bpe > 0:
                input_tokens_tensor = dropout_input_tensor(input_tokens_tensor, mask_token_index,
                                                           dropout=dropout_input_bpe)
            try:
                printing("MASK mask:{} input:{} ", var=[input_mask, input_tokens_tensor], verbose_level="mask", verbose=verbose)
                _loss = bert_with_classifier(input_tokens_tensor, token_type_ids, input_mask,
                                             labels=output_tokens_tensor_aligned if tasks[0] == "normalize" else None,
                                             labels_task_2=output_tokens_tensor_aligned if tasks[0] == "pos" else None)
            except Exception as e:
                print(e)
                print(output_tokens_tensor_aligned, input_tokens_tensor, input_mask)
                _loss = bert_with_classifier(input_tokens_tensor, token_type_ids, input_mask,
                                             labels=output_tokens_tensor_aligned if tasks[0] == "normalize" else None,
                                             labels_task_2=output_tokens_tensor_aligned if tasks[0] == "pos" else None)
            _loss = _loss["loss"]

            if predict_mode:
                # if predict more : will evaluate the model and write its predictions
                # TODO : add mapping_info between task_id to model and task name necessary to iterator
                logits = bert_with_classifier(input_tokens_tensor, token_type_ids, input_mask)["logits_task_2" if tasks[0] == "pos" else "logits_task_1"]
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
                                                            null_str=null_str)

                # de-BPE-tokenize
                src_detokenized = realigne(source_preprocessed, input_alignement_with_raw, null_str=null_str,
                                           tasks=["normalize"],# normalize means we deal wiht bpe input not pos
                                           mask_str=MASK_BERT, remove_mask_str=False)
                gold_detokenized = realigne(gold, input_alignement_with_raw, remove_null_str=True, null_str=null_str,
                                            tasks=tasks,
                                            mask_str=MASK_BERT)
                pred_detokenized_topk = []
                for sent_ls in sent_ls_top:
                    pred_detokenized_topk.append(realigne(sent_ls, input_alignement_with_raw, remove_null_str=True,
                                                          tasks=tasks, remove_extra_predicted_token=True,
                                                          null_str=null_str, mask_str=MASK_BERT))
                    # NB : applying those successively might overlay heuristic
                    if "normalize" in tasks:
                        if heuristic_ls is not None:
                            pred_detokenized_topk = predict_with_heuristic(src_detokenized=src_detokenized,
                                                                           pred_detokenized_topk=pred_detokenized_topk,
                                                                           heuristic_ls=heuristic_ls, verbose=verbose)
                            #print("PRED after @ and #",src_detokenized, pred_detokenized_topk)
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
                                src_text_ls=src_detokenized,
                                text_decoded_ls=pred_detokenized_topk[0], #pred_pos_ls=None, src_text_pos=None,
                                tasks=tasks, ind_batch=iter+batch_i, new_file=new_file,
                                src_text_pos=src_detokenized, pred_pos_ls=gold_detokenized,
                                verbose=verbose)
                    write_conll(format="conll", dir_normalized=dir_gold, dir_original=dir_gold_original_only,
                                src_text_ls=src_detokenized, src_text_pos=src_detokenized, pred_pos_ls=gold_detokenized,
                                text_decoded_ls=gold_detokenized, #pred_pos_ls=None, src_text_pos=None,
                                tasks=tasks, ind_batch=iter + batch_i, new_file=new_file, verbose=verbose)
                    new_file = False
                perf_prediction, skipping = overall_word_level_metric_measure(gold_detokenized, pred_detokenized_topk,
                                                                              topk,
                                                                              metric=metric,
                                                                              samples=samples,
                                                                              agg_func_ls=agg_func_ls,
                                                                              reference_word_dic=reference_word_dic,
                                                                              src_detokenized=src_detokenized)

                score_dic, n_tokens_dic, n_sents_dic = accumulate_scores_across_sents(agg_func_ls=agg_func_ls,
                                                                                      sample_ls=samples,
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
                                    for src, gold_tok, index in zip(sent_src,sent_gold, index_match_with_src):
                                        printing("{}:{} --> {} ", var=[index, src, gold_tok],
                                                 verbose=verbose, verbose_level="alignement")
                                        printing("{}:{} --> {} ", var=[index, src, gold_tok],
                                                 verbose=verbose, verbose_level=verbose_level)

                    print_align_bpe(source_preprocessed, gold, input_alignement_with_raw, verbose=verbose, verbose_level=4)

                    # TODO : detokenize
                    #  write to conll
                    #  compute prediction score

            loss += _loss.detach()
            #if writer is not None:

            if optimizer is not None:
                _loss.backward()
                for opti in optimizer:
                    opti.step()
                    opti.zero_grad()
                mode = "train"
                print("MODE data {} optimizing".format(data_label))
            else:
                mode = "dev"
                print("MODE data {} not optimizing".format(data_label))

            if writer is not None:
                writer.add_scalars("loss",
                                   {"loss-{}-{}-bpe".format(mode, model_id): _loss.clone().cpu().data.numpy() if not isinstance(_loss,int) else 0},
                                   iter+batch_i)
        except StopIteration:
            break

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

    label_heuristic = ""
    if gold_error_detection:
        label_heuristic += "-gold"
    if heuristic_ls is not None:
        label_heuristic += "-#-@"
    if norm_2_noise_eval:
        label_heuristic += "-noise_generation"

    if predict_mode:
        assert len(tasks) == 1
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
                report = report_template(metric_val="accuracy-exact-{}".format(tasks[0]), subsample=sample+label_heuristic, info_score_val=None,
                                         score_val=score/n_tokens if n_tokens > 0 else None, n_sents=n_sents,
                                         avg_per_sent=0,
                                         n_tokens_score=n_tokens,
                                         model_full_name_val=model_id, task=tasks,
                                         evaluation_script_val="exact_match",
                                         model_args_dir=args_dir,
                                         token_type="word",
                                         report_path_val=None,
                                         data_val=data_label)
                reports.append(report)
            # class negative 0 , class positive 1
            # TODO : make that more consistent with user needs !
            if "normalize" in tasks:
                if "all" in samples and TASKS_PARAMETER["normalize"]["predicted_classes"][0] in samples and TASKS_PARAMETER["normalize"]["predicted_classes"][1] in samples:

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
                        reports.append(report)

                        if writer is not None and log_perf:
                            writer.add_scalars("perf-{}".format(mode),
                                               {"{}-{}-{}-bpe".format(metric_val, mode, model_id):
                                                    score if score is not None else 0
                                                },
                                           iter + batch_i)


    else:
        reports = None
    iter += batch_i
    return loss, iter, reports
