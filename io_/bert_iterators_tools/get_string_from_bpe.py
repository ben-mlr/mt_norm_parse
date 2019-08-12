from env.importing import OrderedDict, torch, pdb
from io_.dat.constants import PAD_ID_LOSS_STANDART
from io_.bert_iterators_tools.string_processing import preprocess_batch_string_for_bert, from_bpe_token_to_str, get_indexes, get_indexes_src_gold
import io_.bert_iterators_tools.alignement as alignement
from io_.dat.constants import MASK_BERT


def get_prediction(logits_dic, topk):
    assert topk == 1
    predictions_topk_dic = OrderedDict()
    for logit_label, logits in logits_dic.items():
        #we handle parsing_types in a specific way
        if logit_label == "parsing_types":
            batch_size = logits.size(0)
            # getting predicted heads (to know which labels of the graph which should look at
            pred_heads = predictions_topk_dic["parsing_heads"][:, :, 0]
            # we extract from the logits only the one of the predicted heads (that are not PAD_ID_LOSS_STANDART : useless)
            logits = logits[(pred_heads != PAD_ID_LOSS_STANDART).nonzero()[:, 0], (pred_heads != PAD_ID_LOSS_STANDART).nonzero()[:,1], pred_heads[pred_heads != PAD_ID_LOSS_STANDART]]
            # we take the argmax label of this heads
            predictions_topk_dic[logit_label] = torch.argsort(logits, dim=-1, descending=True)[:, :topk]
            # only keeping the top 1 prediction
            # predictions_topk_dic[logit_label] = predictions_topk_dic[logit_label][:, 0]
            # reshaping
            predictions_topk_dic[logit_label] = predictions_topk_dic[logit_label].view(batch_size, -1, topk)
        else:
            predictions_topk_dic[logit_label] = torch.argsort(logits, dim=-1, descending=True)[:, :, :topk]
            print("--")
            #pdb.set_trace()

    return predictions_topk_dic


def get_bpe_string(predictions_topk_dic,
                   input_alignement_with_raw,
                   output_tokens_tensor_aligned_dic, input_tokens_tensor, topk,
                   tokenizer, task_to_label_dictionary, null_str, null_token_index, verbose):

    predict_dic = OrderedDict()
    label_dic = OrderedDict()

    for label in predictions_topk_dic:
        if label == "parsing_heads":
            #pdb.set_trace()
            pass
        sent_ls_top = from_bpe_token_to_str(predictions_topk_dic[label], topk, tokenizer=tokenizer,
                                            pred_mode=True, label_dictionary=task_to_label_dictionary[label],
                                            get_string=False, label=label,
                                            null_token_index=null_token_index, null_str=null_str)

        gold = from_bpe_token_to_str(output_tokens_tensor_aligned_dic[label], topk, tokenizer=tokenizer,
                                     label_dictionary=task_to_label_dictionary[label], pred_mode=False,
                                     get_string=False, label=label, null_token_index=null_token_index,
                                     null_str=null_str)


        predict_dic[label] = sent_ls_top
        label_dic[label] = gold
    source_preprocessed = from_bpe_token_to_str(input_tokens_tensor, topk, tokenizer=tokenizer,
                                                label_dictionary=task_to_label_dictionary[label],
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
    for label in label_dic:
        label_detokenized_dic[label] = alignement.realigne_multi(label_dic[label], input_alignement_with_raw,
                                                                 remove_null_str=True,
                                                                 null_str=null_str,
                                                                 task=label,
                                                                 mask_str=MASK_BERT)

        #label_detokenized_dic[task] = label_detokenized_dic[task][0]
        if label == "pos" or label.startswith("parsing"):
            #gold_detokenized = [gold_sent[:len(src_sent)] for gold_sent, src_sent in zip(gold_detokenized, src_detokenized)]
            # we remove padding here based on src that is correctly padded
            label_detokenized_dic[label] = [gold_sent[:len(src_sent)] for gold_sent, src_sent in zip(label_detokenized_dic[label], src_detokenized)]

        predict_detokenize_dic[label] = []
        # handle several prediction
        for sent_ls in predict_dic[label]:
            predict_detokenize_dic[label].append(alignement.realigne_multi(sent_ls, input_alignement_with_raw,
                                                 remove_null_str=True, task=label, remove_extra_predicted_token=True,
                                                 null_str=null_str, mask_str=MASK_BERT))
    return src_detokenized, label_detokenized_dic, predict_detokenize_dic


def get_aligned_output(label_per_task):
    output_tokens_tensor_aligned_dict = OrderedDict()
    for label in label_per_task:
        if label != "normalize" :
            output_tokens_tensor_aligned_dict[label] = label_per_task[label]


    return output_tokens_tensor_aligned_dict
