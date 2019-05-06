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
              model_id,
              skip_1_t_n=True,
              writer=None, optimizer=None,
              predict_mode=False, topk=None, metric=None,
              print_pred=False, args_dir=None,
              heuristic_ls=None, gold_error_detection=False,
              reference_word_dic=None,
              writing_pred=False, dir_end_pred=None, extra_label_for_prediction="",
              log_perf=True,
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
    if predict_mode:
        if topk is None:
            topk = 1
            printing("PREDICITON MODE : setting topk to default 1 ", verbose_level=1, verbose=verbose)
        print_pred = True
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
            batch.raw_input = preprocess_batch_string_for_bert(batch.raw_input)
            batch.raw_output = preprocess_batch_string_for_bert(batch.raw_output)
            input_tokens_tensor, input_segments_tensors, inp_bpe_tokenized, input_alignement_with_raw, input_mask = get_indexes(batch.raw_input, tokenizer, verbose, use_gpu)
            output_tokens_tensor, output_segments_tensors, out_bpe_tokenized, output_alignement_with_raw, output_mask =\
                get_indexes(batch.raw_output, tokenizer, verbose, use_gpu)

            printing("DATA dim : {} input {} output ", var=[input_tokens_tensor.size(), output_tokens_tensor.size()],
                     verbose_level=2, verbose=verbose)

            _verbose = verbose
            if input_tokens_tensor.size(1) != output_tokens_tensor.size(1):
                printing("-------------- Alignement broken --------------", verbose=verbose, verbose_level=2)
                if input_tokens_tensor.size(1) > output_tokens_tensor.size(1):
                    printing("N to 1 : NOISY splitted MORE than standard", verbose=verbose, verbose_level=2)
                    noisy_over_splitted += 1
                elif input_tokens_tensor.size(1) < output_tokens_tensor.size(1):
                    printing("1 to N : NOISY splitted LESS than standard", verbose=verbose, verbose_level=2)
                    noisy_under_splitted += 1
                    if skip_1_t_n:
                        printing("WE SKIP IT ", verbose=verbose, verbose_level=1)
                        continue
                if isinstance(verbose, int):
                    _verbose += 1
            else:
                aligned += 1

            # logging
            printing("DATA : pre-tokenized input {} ", var=[batch.raw_input], verbose_level="raw_data",
                     verbose=_verbose)
            printing("DATA : BPEtokenized input ids {}", var=[input_tokens_tensor], verbose_level=3,
                     verbose=verbose)

            printing("DATA : pre-tokenized output {} ", var=[batch.raw_output],
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
            # aligning output BPE with input (we are rejecting batch with at least one 1 to n case
            # (that we don't want to handle)
            output_tokens_tensor_aligned, input_tokens_tensor_aligned, input_alignement_with_raw, _1_to_n_token = \
                aligned_output(input_tokens_tensor, output_tokens_tensor,
                               input_alignement_with_raw,
                               output_alignement_with_raw, mask_token_index=mask_token_index,
                               null_token_index=null_token_index, verbose=verbose)

            #segments_ids = [[0 for _ in range(len(tokenized))] for tokenized in tokenized_ls]
            #mask = [[1 for _ in inp] + [0 for _ in range(max_sent_len - len(inp))] for inp in segments_ids]

            if batch_i == n_iter_max:
                break
            if _1_to_n_token:
                skipping_batch_n_to_1 += 1
                #continue
            # CHECKING ALIGNEMENT
            input_tokens_tensor = input_tokens_tensor_aligned
            assert output_tokens_tensor_aligned.size(0) == input_tokens_tensor.size(0)
            assert output_tokens_tensor_aligned.size(1) == input_tokens_tensor.size(1)
            #assert output_tokens_tensor_aligned.size(0) == input_tokens_tensor_aligned.size(0)
            #assert output_tokens_tensor_aligned.size(1) == input_tokens_tensor_aligned.size(1)
            # we consider only 1 sentence case
            token_type_ids = torch.zeros_like(input_tokens_tensor)
            if input_tokens_tensor.is_cuda:
                token_type_ids = token_type_ids.cuda()
            printing("CUDA SANITY CHECK input_tokens:{}  type:{} input_mask:{}  label:{}",
                     var=[input_tokens_tensor.is_cuda, token_type_ids.is_cuda, input_mask.is_cuda,
                          output_tokens_tensor_aligned.is_cuda],
                     verbose=verbose, verbose_level="cuda")
            # we have to recompute the mask based on aligned input
            input_mask = torch.Tensor([[1 if token_id != PAD_ID_BERT else 0 for token_id in sent_token] for sent_token in input_tokens_tensor]).long()

            _loss = bert_with_classifier(input_tokens_tensor, token_type_ids, input_mask,
                                          labels=output_tokens_tensor_aligned)

            if predict_mode:
                # if predict more : will evaluate the model and write its predictions 
                logits = bert_with_classifier(input_tokens_tensor, token_type_ids, input_mask)
                predictions_topk = torch.argsort(logits, dim=-1, descending=True)[:, :, :topk]
                # from bpe index to string

                sent_ls_top = from_bpe_token_to_str(predictions_topk, topk, tokenizer=tokenizer, pred_mode=True,
                                                    null_token_index=null_token_index, null_str=null_str)
                gold = from_bpe_token_to_str(output_tokens_tensor_aligned, topk, tokenizer=tokenizer,
                                             pred_mode=False, null_token_index=null_token_index,null_str=null_str)
                source_preprocessed = from_bpe_token_to_str(input_tokens_tensor, topk, tokenizer=tokenizer,
                                                            pred_mode=False, null_token_index=null_token_index,null_str=null_str)

                # de-BPE-tokenize
                src_detokenized = realigne(source_preprocessed, input_alignement_with_raw, null_str=null_str,
                                           mask_str=MASK_BERT, remove_mask_str=False)
                gold_detokenized = realigne(gold, input_alignement_with_raw, remove_null_str=True, null_str=null_str,
                                            mask_str=MASK_BERT)
                pred_detokenized_topk = []
                for sent_ls in sent_ls_top:
                    pred_detokenized_topk.append(realigne(sent_ls, input_alignement_with_raw,
                                                          remove_null_str=True,
                                                          remove_extra_predicted_token=True,
                                                          null_str=null_str, mask_str=MASK_BERT))
                    # NB : applying those successivly might overlay heuristic
                    if heuristic_ls is not None:
                        pred_detokenized_topk = predict_with_heuristic(src_detokenized=src_detokenized,
                                                                   pred_detokenized_topk=pred_detokenized_topk,
                                                                   heuristic_ls=heuristic_ls, verbose=verbose)
                        print("PRED after @ and #",src_detokenized, pred_detokenized_topk)
                    if gold_error_detection:
                        pred_detokenized_topk = predict_with_heuristic(src_detokenized=src_detokenized,
                                                                       gold_detokenized=gold_detokenized,
                                                                       pred_detokenized_topk=pred_detokenized_topk,
                                                                       heuristic_ls=["gold_detection"], verbose=verbose)
                        print("PRED after gold",gold_detokenized, pred_detokenized_topk)


                if writing_pred:
                    write_conll(format="conll", dir_normalized=dir_normalized,
                                dir_original=dir_normalized_original_only,
                                src_text_ls=src_detokenized,
                                text_decoded_ls=pred_detokenized_topk[0], pred_pos_ls=None, src_text_pos=None,
                                tasks=["normalize"], ind_batch=iter+batch_i, new_file=new_file,
                                verbose=verbose)
                    write_conll(format="conll", dir_normalized=dir_gold, dir_original=dir_gold_original_only,
                                src_text_ls=src_detokenized,
                                text_decoded_ls=gold_detokenized, pred_pos_ls=None, src_text_pos=None,
                                tasks=["normalize"], ind_batch=iter + batch_i, new_file=new_file, verbose=verbose)
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
                    print("SKIP",_1_to_n_token,  input_tokens_tensor.size(1) < output_tokens_tensor.size(1))
                    printing("TRAINING : Score : {} / {} tokens / {} sentences", var=[
                                                                                      perf_prediction["sum"]["all"]["score"],
                                                                                      perf_prediction["sum"]["all"]["n_tokens"],
                                                                                      perf_prediction["sum"]["all"]["n_sents"]
                                                                                      ],
                             verbose=verbose, verbose_level=1)
                    printing("TRAINING : eval gold {}-{} {}", var=[iter, batch_i, gold_detokenized],
                             verbose=1,
                             verbose_level=1)
                    printing("TRAINING : eval pred {}-{} {}", var=[iter, batch_i, pred_detokenized_topk],
                             verbose=verbose,
                             verbose_level=1)
                    printing("TRAINING : eval src {}-{} {}", var=[iter, batch_i, src_detokenized],
                             verbose=1, verbose_level=1)
                    printing("TRAINING : BPE eval gold {}-{} {}", var=[iter, batch_i, gold],
                             verbose=1,
                             verbose_level=1)
                    printing("TRAINING : BPE eval pred {}-{} {}", var=[iter, batch_i, sent_ls_top],
                             verbose=verbose,
                             verbose_level=1)
                    printing("TRAINING : BPE eval src {}-{} {}", var=[iter, batch_i, source_preprocessed],
                             verbose=1, verbose_level=1)
                    printing("TRAINING : BPE eval src {}-{} {}", var=[iter, batch_i, input_alignement_with_raw],
                             verbose=verbose, verbose_level=1)

                    def print_align_bpe(source_preprocessed, gold, input_alignement_with_raw, verbose):
                        assert len(source_preprocessed)==len(gold), ""
                        assert len(input_alignement_with_raw) == len(gold), ""
                        for sent_src, sent_gold, index_match_with_src in zip(source_preprocessed, gold, input_alignement_with_raw):
                            assert len(sent_src) == len(sent_gold)
                            assert len(sent_src) == len(sent_gold)
                            for src, gold_tok, index in zip(sent_src,sent_gold, index_match_with_src):
                                printing("{}:{} --> {} ", var=[index, src, gold_tok],
                                         verbose=verbose, verbose_level="alignement")

                    print_align_bpe(source_preprocessed, gold, input_alignement_with_raw, verbose)



                    # TODO : detokenize
                    #  write to conll
                    #  compute prediction score

            loss += _loss.detach()
            #if writer is not None:

            if optimizer is not None:
                _loss.backward()
                optimizer.step()
                optimizer.zero_grad()
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
             "; {} with  LESS SPLITTED {} + BATCH with skipped_1_to_n : {}) ",
             var=[data_label, batch_i, batch.input_seq.size(0), noisy_under_splitted+skipping_batch_n_to_1, aligned,
                  noisy_over_splitted, noisy_under_splitted,
                  "SKIPPED" if skip_1_t_n else "",
                  skipping_batch_n_to_1],
             verbose=verbose, verbose_level=0)
    printing("WARNING on {} ON THE EVALUATION SIDE we skipped extra {} batch ", var=[data_label, skipping_evaluated_batch], verbose_level=1, verbose=1)
    if predict_mode:
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
                report = report_template(metric_val="accuracy-exact", subsample=sample, info_score_val=None,
                                         score_val=score/n_tokens if n_tokens>0 else None, n_sents=n_sents, avg_per_sent=0,
                                         n_tokens_score=n_tokens,
                                         model_full_name_val=model_id, task=["normalize"],
                                         evaluation_script_val="exact_match",
                                         model_args_dir=args_dir,
                                         token_type="word",
                                         report_path_val=None,
                                         data_val=data_label)
                reports.append(report)
            # class negative 0 , class positive 1
            # TODO : make that more consistent with user needs !
            if "all" in samples and TASKS_PARAMETER["normalize"]["predicted_classes"][0] in samples and TASKS_PARAMETER["normalize"]["predicted_classes"][1] in samples:

                # then we can compute all the confusion matrix rate
                # TODO : factore with TASKS_2_METRICS_STR
                for metric_val in ["recall-normalize", "precision-normalize", "f1-normalize", "tnr-normalize", "npv-normalize", "accuracy-normalize"]:
                    score, n_rate_universe = get_perf_rate(metric=metric_val, n_tokens_dic=n_tokens_dic,
                                                           score_dic=score_dic, agg_func=agg_func)
                    report = report_template(metric_val=metric_val, subsample="rates", info_score_val=None,
                                             score_val=score, n_sents=n_sents_dic[agg_func]["all"],
                                             avg_per_sent=0,
                                             n_tokens_score=n_rate_universe,
                                             model_full_name_val=model_id, task=["normalize"],
                                             evaluation_script_val="exact_match",
                                             model_args_dir=args_dir,
                                             token_type="word",
                                             report_path_val=None,
                                             data_val=data_label)
                    reports.append(report)
                    pdb.set_trace()

                    if writer is not None and log_perf:
                        print("-->", mode, iter+batch_i, iter, batch_i)
                        writer.add_scalars("perf-{}".format(mode),
                                           {"{}-{}-{}-bpe".format(metric_val, mode, model_id):
                                                score if score is not None else 0
                                            },
                                       iter + batch_i)


    else:
        reports = None
    iter += batch_i
    return loss, iter, reports
