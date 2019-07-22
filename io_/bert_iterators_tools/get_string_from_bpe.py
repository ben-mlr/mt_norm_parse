from env.importing import OrderedDict, torch, pdb
from io_.bert_iterators_tools.string_processing import preprocess_batch_string_for_bert, from_bpe_token_to_str, get_indexes, get_indexes_src_gold
import io_.bert_iterators_tools.alignement as alignement
from io_.dat.constants import MASK_BERT


def get_prediction(logits_dic, topk):
    assert topk == 1
    predictions_topk_dic = OrderedDict()
    for logit_label, logits in logits_dic.items():
        predictions_topk_dic[logit_label] = torch.argsort(logits, dim=-1, descending=True)[:, :, :topk]
    return predictions_topk_dic


def get_bpe_string(predictions_topk_dic, output_tokens_tensor_aligned_dic, input_tokens_tensor, topk,
                   tokenizer, task_to_label_dictionary, tasks, null_str, null_token_index, verbose):

    predict_dic = OrderedDict()
    label_dic = OrderedDict()

    for task in tasks:
        sent_ls_top = from_bpe_token_to_str(predictions_topk_dic[task], topk, tokenizer=tokenizer,
                                            pred_mode=True, label_dictionary=task_to_label_dictionary[task],
                                            get_string=False, task=task,
                                            null_token_index=null_token_index, null_str=null_str)

        gold = from_bpe_token_to_str(output_tokens_tensor_aligned_dic[task], topk, tokenizer=tokenizer,
                                     label_dictionary=task_to_label_dictionary[task],
                                     pred_mode=False,
                                     get_string=False, task=task,
                                     null_token_index=null_token_index, null_str=null_str)

        predict_dic[task] = sent_ls_top
        label_dic[task] = gold

    source_preprocessed = from_bpe_token_to_str(input_tokens_tensor, topk, tokenizer=tokenizer,
                                                label_dictionary=task_to_label_dictionary[task],
                                                pred_mode=False,
                                                null_token_index=null_token_index, null_str=null_str,
                                                get_string=True, verbose=verbose)

    return source_preprocessed, label_dic, predict_dic


def get_detokenized_str(source_preprocessed, input_alignement_with_raw, label_dic, predict_dic,
                        null_str, tasks, remove_mask_str_prediction):

    # de-BPE-tokenize
    predict_detokenize_dic = OrderedDict()
    label_detokenized_dic = OrderedDict()

    src_detokenized = alignement.realigne_multi(source_preprocessed, input_alignement_with_raw, null_str=null_str,
                                                task="normalize",
                                                # normalize means we deal wiht bpe input not pos
                                                mask_str=MASK_BERT,
                                                remove_mask_str=remove_mask_str_prediction)
    for task in tasks:
        label_detokenized_dic[task] = alignement.realigne_multi(label_dic[task], input_alignement_with_raw,
                                                                remove_null_str=True,
                                                                null_str=null_str,
                                                                task=task,
                                                                mask_str=MASK_BERT)

        #label_detokenized_dic[task] = label_detokenized_dic[task][0]
        if task == "pos":
            #gold_detokenized = [gold_sent[:len(src_sent)] for gold_sent, src_sent in zip(gold_detokenized, src_detokenized)]
            # we remove padding here based on src that is correctly padded
            label_detokenized_dic[task] = [gold_sent[:len(src_sent)] for gold_sent, src_sent in zip(label_detokenized_dic[task], src_detokenized)]

        predict_detokenize_dic[task] = []
        for sent_ls in predict_dic[task]:
            predict_detokenize_dic[task].append(alignement.realigne_multi(sent_ls, input_alignement_with_raw,
                                                remove_null_str=True, task=task, remove_extra_predicted_token=True,
                                                null_str=null_str, mask_str=MASK_BERT))
    pdb.set_trace()

    return src_detokenized, label_detokenized_dic, predict_detokenize_dic
