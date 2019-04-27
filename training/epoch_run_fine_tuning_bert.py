from env.importing import *
from env.project_variables import *

from io_.info_print import printing
from io_.dat.normalized_writer import write_conll
from io_.bert_iterators_tools.string_processing import preprocess_batch_string_for_bert, from_bpe_token_to_str, get_indexes
from io_.bert_iterators_tools.alignement import aligned_output, realigne

from evaluate.scoring.report import overall_word_level_metric_measure


def epoch_run(batchIter, tokenizer,
              iter, n_iter_max, bert_with_classifier, epoch,
              use_gpu, data_label, null_token_index, null_str,
              skip_1_t_n=True,
              writer=None, optimizer=None,
              predict_mode=False, topk=None, metric=None,
              print_pred=False,
              writing_pred=False, dir_end_pred=None,
              verbose=0):

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
        dir_normalized = os.path.join(dir_end_pred, "{}_ep-prediction.conll".format(epoch))
        dir_normalized_original_only = os.path.join(dir_end_pred, "{}_ep-prediction_src.conll".format(epoch))
        dir_gold = os.path.join(dir_end_pred, "{}_ep-gold.conll".format(epoch))
        dir_gold_original_only = os.path.join(dir_end_pred, "{}_ep-gold_src.conll".format(epoch))

    batch_i = 0
    noisy_over_splitted = 0
    noisy_under_splitted = 0
    aligned = 0
    skipping_batch_n_to_1 = 0

    loss = 0
    score = 0
    n_tokens = 0
    n_sents = 0
    skipping_evaluated_batch = 0
    mode = "?"
    new_file = True
    while True:

        try:
            batch_i += 1

            batch = batchIter.__next__()
            batch.raw_input = preprocess_batch_string_for_bert(batch.raw_input)
            batch.raw_output = preprocess_batch_string_for_bert(batch.raw_output)

            pdb.set_trace()

            input_tokens_tensor, input_segments_tensors, inp_bpe_tokenized, input_alignement_with_raw, input_mask = get_indexes(batch.raw_input, tokenizer, verbose, use_gpu)
            output_tokens_tensor, output_segments_tensors, out_bpe_tokenized, output_alignement_with_raw, output_mask = get_indexes(batch.raw_output,
                                                                                                                                    tokenizer,
                                                                                                                                    verbose,
                                                                                                                                    use_gpu)

            printing("DATA dim : {} input {} output ", var=[input_tokens_tensor.size(), output_tokens_tensor.size()],
                     verbose_level=2, verbose=verbose)

            _verbose = verbose if isinstance(verbose, int) else 0

            if input_tokens_tensor.size(1) != output_tokens_tensor.size(1):
                printing("-------------- Alignement broken", verbose=verbose, verbose_level=2)
                if input_tokens_tensor.size(1) > output_tokens_tensor.size(1):
                    printing("N to 1 like : NOISY splitted MORE than standard", verbose=verbose, verbose_level=2)
                    noisy_over_splitted += 1
                elif input_tokens_tensor.size(1) < output_tokens_tensor.size(1):
                    printing("1 to N : NOISY splitted LESS than standard", verbose=verbose, verbose_level=2)
                    noisy_under_splitted += 1
                    if skip_1_t_n:
                        printing("WE SKIP IT ", verbose=verbose, verbose_level=2)
                        continue
                _verbose += 1
            else:
                aligned += 1

            # logging
            printing("DATA : pre-tokenized input {} ", var=[batch.raw_input], verbose_level=3,
                     verbose=_verbose)
            printing("DATA : BPEtokenized input ids {}", var=[input_tokens_tensor], verbose_level=3,
                     verbose=verbose)

            printing("DATA : pre-tokenized output {} ", var=[batch.raw_output],
                     verbose_level=4,
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
            output_tokens_tensor_aligned, _1_to_n_token = aligned_output(input_tokens_tensor, output_tokens_tensor,
                                                                         input_alignement_with_raw,
                                                                         output_alignement_with_raw,
                                                                         null_token_index=null_token_index)

            if batch_i == n_iter_max:
                break
            if _1_to_n_token:
                skipping_batch_n_to_1 += 1
                continue
            # CHECKING ALIGNEMENT
            assert output_tokens_tensor_aligned.size(0) == input_tokens_tensor.size(0)
            assert output_tokens_tensor_aligned.size(1) == input_tokens_tensor.size(1)
            # we consider only 1 sentence case
            output_tokens_tensor_aligned = output_tokens_tensor_aligned
            token_type_ids = torch.zeros_like(input_tokens_tensor)
            if input_tokens_tensor.is_cuda:
                token_type_ids = token_type_ids.cuda()
            printing("CUDA SANITY CHECK input_tokens:{}  type:{} input_mask:{}  label:{}",
                     var=[input_tokens_tensor.is_cuda, token_type_ids.is_cuda, input_mask.is_cuda,
                          output_tokens_tensor_aligned.is_cuda],
                     verbose=verbose, verbose_level="cuda")
            try:
                _loss = bert_with_classifier(input_tokens_tensor, token_type_ids, input_mask, labels=output_tokens_tensor_aligned)
                if predict_mode:
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
                    src_detokenized = realigne(source_preprocessed, input_alignement_with_raw, null_str=null_str)
                    gold_detokenized = realigne(gold, input_alignement_with_raw, remove_null_str=True,null_str=null_str)
                    pred_detokenized_topk = []
                    for sent_ls in sent_ls_top:
                        pred_detokenized_topk.append(realigne(sent_ls, input_alignement_with_raw, remove_null_str=True,
                                                              remove_extra_predicted_token=True,null_str=null_str))

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

                    perf_prediction, skipping = overall_word_level_metric_measure(gold_detokenized, pred_detokenized_topk, topk, metric=metric, agg_func_ls=["sum"])
                    skipping_evaluated_batch += skipping
                    score += perf_prediction[0]["score"]
                    n_tokens += perf_prediction[0]["n_tokens"]
                    n_sents += perf_prediction[0]["n_sents"]

                    if print_pred:
                        printing("TRAINING : Score : {} / {} tokens / {} sentences", var=[perf_prediction[0]["score"],
                                                                                          perf_prediction[0]["n_tokens"],
                                                                                          perf_prediction[0]["n_sents"]],
                                 verbose=verbose, verbose_level=1)
                        printing("TRAINING : eval gold {}-{} {}", var=[iter, batch_i, gold_detokenized], verbose=verbose,
                                 verbose_level=1)
                        printing("TRAINING : eval pred {}-{} {}", var=[iter, batch_i, pred_detokenized_topk], verbose=verbose,
                                 verbose_level=1)
                        printing("TRAINING : eval src {}-{} {}", var=[iter, batch_i, src_detokenized],
                                 verbose=verbose, verbose_level=1)
                        # TODO : detokenize
                        #  write to conll
                        #  compute prediction score

            except RuntimeError as e:
                print(e)
                pdb.set_trace()

            loss += _loss
            _loss.backward()
            if optimizer is not None:
                optimizer.step()
                optimizer.zero_grad()
                mode = "train"
                print("MODE data {} optimizing".format(data_label))
            else:
                mode = "dev"
                print("MODE data {} optimizing".format(data_label))

            if writer is not None:
                writer.add_scalars("loss",
                                    {"loss-{}-bpe".format(mode):
                                     _loss.clone().cpu().data.numpy()}, iter+batch_i)
        except StopIteration:
            break

    printing("WARNING on {} : Out of {} batch of {} sentences each : {} batch aligned ; {} with at least 1 sentence "
             "noisy MORE SPLITTED "
             "; {} with  LESS SPLITTED {} + BATCH with skipped_1_to_n : {} ",
             var=[data_label, batch_i, batch.input_seq.size(0), aligned, noisy_over_splitted, noisy_under_splitted,
                  "SKIPPED" if skip_1_t_n else "",
                  skipping_batch_n_to_1],
             verbose=verbose, verbose_level=0)
    printing("WARNING on {} ON THE EVALUATION SIDE we skipped extra {} batch ", var=[data_label,skipping_evaluated_batch], verbose_level=1, verbose=1)
    if predict_mode:
        try:
            report = {"score": score/n_tokens, "agg_func": "mean",
                      "subsample": "all", "data": data_label,
                      "metric": metric, "n_tokens": n_tokens,"n_sents": n_sents}
        except Exception as e:
            print(e)
            report = []
        if writer is not None:
            writer.add_scalars("prediction_score",
                               {"exact_match-all-{}".format(mode):
                                    report["score"]}, epoch)

    else:
        report = None
    iter += batch_i
    return loss, iter, report
