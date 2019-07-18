from env.importing import *
from env.models_dir import BERT_MODEL_DIC
from model.bert_tools_from_core_code.tokenization import BertTokenizer
from env.project_variables import *
from io_.dat.constants import *
from env.project_variables import EN_LINES_EWT_TRAIN, LIU_DEV, TRAINING, DEMO
from env.tasks_settings import TASKS_PARAMETER
from io_.dat import conllu_data
from io_.info_print import printing, print_char_seq, disable_tqdm_level
from io_.bert_iterators_tools.alignement import aligned_output, realigne

from io_.bert_iterators_tools.string_processing import preprocess_batch_string_for_bert, from_bpe_token_to_str, get_indexes

from io_.data_iterator import data_gen_multi_task_sampling_batch, readers_load


def update_dic(bpe_counter, bpe_normed_counter, bpe_need_norm_counter, input_bpe, output_bpe,
               input_alignement_with_raw, norm_not_norm):
    assert len(input_bpe) == 1
    assert len(input_bpe) == len(output_bpe)
    assert len(input_bpe[0]) == len(output_bpe[0])

    input_alignement_with_raw = input_alignement_with_raw[0]
    input_bpe = input_bpe[0]
    output_bpe = output_bpe[0]
    norm_not_norm = norm_not_norm[0]

    for ind, (b_inp, b_out) in enumerate(zip(input_bpe, output_bpe)):
        if b_inp in ["[CLS]", "[SEP]"]:
            continue
        original_index = input_alignement_with_raw[ind]
        bool_norm = norm_not_norm[original_index]
        if b_inp not in bpe_counter:
            bpe_counter[b_inp] = {b_out: 1}
            bpe_normed_counter[b_inp] = {b_out: bool_norm.item()}
            bpe_need_norm_counter[b_inp] = {b_out: 1 - bool_norm.item()}
        else:
            if b_out not in bpe_counter[b_inp]:
                bpe_counter[b_inp][b_out] = 1
                bpe_normed_counter[b_inp][b_out] = bool_norm.item()
                bpe_need_norm_counter[b_inp][b_out] = 1 - bool_norm.item()
            else:
                bpe_counter[b_inp][b_out] += 1
                bpe_normed_counter[b_inp][b_out] += bool_norm.item()
                bpe_need_norm_counter[b_inp][b_out] += (1 - bool_norm.item())

    return bpe_counter, bpe_normed_counter, bpe_need_norm_counter


def stat_bpe(iterator_multi,tokenizer, mask_token_index, null_token_index=1,verbose=1):

    bpe_counter = OrderedDict()
    bpe_normed_counter = OrderedDict()
    bpe_need_norm_counter = OrderedDict()

    while True:
        try:
            batch = iterator_multi.__next__()

            norm_bool = batch.output_norm_not_norm
            batch.raw_input = preprocess_batch_string_for_bert(batch.raw_input)
            batch.raw_output = preprocess_batch_string_for_bert(batch.raw_output)
            input_tokens_tensor, input_segments_tensors, inp_bpe_tokenized, input_alignement_with_raw, input_mask = get_indexes(
                batch.raw_input, tokenizer, verbose, False)
            output_tokens_tensor, output_segments_tensors, out_bpe_tokenized, output_alignement_with_raw, output_mask =\
                get_indexes(batch.raw_output, tokenizer, verbose, False)

            output_tokens_tensor_aligned, input_tokens_tensor_aligned, new_alignement_with_input_ls, _1_to_n_token= aligned_output(input_tokens_tensor,
                                                                         output_tokens_tensor,
                                                                         input_alignement_with_raw,
                                                                         output_alignement_with_raw,
                                                                         null_token_index=null_token_index,
                                                                         mask_token_index=mask_token_index,
                                                                         verbose=verbose)

            if not _1_to_n_token:
                output_tokens_tensor_aligned = \
                    from_bpe_token_to_str(output_tokens_tensor_aligned, topk=1, tokenizer=tokenizer,
                                          pred_mode=False, null_token_index=null_token_index,
                                          null_str=NULL_STR_TO_SHOW)
                bpe_counter, bpe_normed_counter, bpe_need_norm_counter = \
                    update_dic(bpe_counter, bpe_normed_counter, bpe_need_norm_counter,
                               input_bpe=inp_bpe_tokenized,
                               output_bpe=output_tokens_tensor_aligned,
                               norm_not_norm=norm_bool, input_alignement_with_raw=input_alignement_with_raw)
        except StopIteration as e:
            print(Exception(e))
            break

    return bpe_counter, bpe_normed_counter, bpe_need_norm_counter


def get_embedding_mat(iterator_multi,tokenizer,bert_token_classification, mask_token_index,null_token_index=1,
                      early_breaking=None,
                      verbose=0):

    bpe_counter = OrderedDict()
    bpe_normed_counter = OrderedDict()
    bpe_need_norm_counter = OrderedDict()

    embedding_output_ls = []
    embedding_inpput_ls = []
    batch_ind = 0
    while True:
        try:
            batch = iterator_multi.__next__()
            batch_ind+=1
            norm_bool = batch.output_norm_not_norm
            batch.raw_input = preprocess_batch_string_for_bert(batch.raw_input)
            batch.raw_output = preprocess_batch_string_for_bert(batch.raw_output)

            # noisy
            input_tokens_tensor, input_segments_tensors, inp_bpe_tokenized, input_alignement_with_raw, input_mask = get_indexes(
                batch.raw_input, tokenizer, verbose, False)
            # gold
            output_tokens_tensor, output_segments_tensors, out_bpe_tokenized, output_alignement_with_raw, output_mask =\
                get_indexes(batch.raw_output, tokenizer, verbose, False)
            # get embedding for noisy
            token_type_ids = torch.zeros_like(input_tokens_tensor)
            embedding_input, _ = bert_token_classification.bert(input_tokens_tensor, token_type_ids, input_mask,
                                                                output_all_encoded_layers=False)
            embedding_inpput_ls.append(embedding_input[:,0,:])
            # get embedding for gold
            token_type_output_ids = torch.zeros_like(output_tokens_tensor)
            embedding_output, _ = bert_token_classification.bert(output_tokens_tensor, token_type_output_ids , output_mask,
                                                                 output_all_encoded_layers=False)
            embedding_output_ls.append(embedding_output[:,0,:])
            if False:
                aligned_output(input_tokens_tensor,output_tokens_tensor,
                               input_alignement_with_raw,
                               output_alignement_with_raw,
                               mask_token_index=mask_token_index,
                               null_token_index=null_token_index,
                               verbose=verbose)
            if early_breaking is not None and batch_ind>early_breaking:
                print("BREAKING after {} ".format(batch_ind))
                break
        except StopIteration as e:
            print(Exception(e))
            break

    out = {"input": np.array(torch.cat(embedding_output_ls).detach()),
           "gold": np.array(torch.cat(embedding_inpput_ls).detach())
           }

    return out




def bpe_statistics_on_data(data_set, dict_path, mask_token_index, bert_model="bert-cased",
                           bert_token_classification=None,print_raw=False,early_breaking=None,
                           output="bpe_stat",verbose=1):

    voc_tokenizer = BERT_MODEL_DIC[bert_model]["vocab"]

    tokenizer = BertTokenizer.from_pretrained(voc_tokenizer)

    word_dictionary, word_dictionary_norm, char_dictionary, pos_dictionary, \
    xpos_dictionary, type_dictionary = conllu_data.create_dict(dict_path=dict_path,
                                                               train_path=data_set,
                                                               dev_path=data_set,
                                                               test_path=None,
                                                               word_embed_dict={},
                                                               expand_vocab_bool=False,
                                                               word_normalization=True,
                                                               tasks=["normalize"],
                                                               dry_run=False,
                                                               pos_specific_data_set=None,
                                                               add_start_char=1)

    data_set = [data_set]
    tasks = ["norm_not_norm"]
    readers = readers_load(datasets=data_set, tasks=tasks, word_dictionary=word_dictionary,
                           word_dictionary_norm=word_dictionary_norm, char_dictionary=char_dictionary,
                           pos_dictionary=pos_dictionary, xpos_dictionary=xpos_dictionary,
                           type_dictionary=type_dictionary, use_gpu=None,
                           norm_not_norm=True, word_decoder=True,
                           add_start_char=1, add_end_char=1, symbolic_end=True, symbolic_root=True,
                           verbose=1)

    iterator_multi = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers, batch_size=1,
                                                        word_dictionary=word_dictionary,
                                                        char_dictionary=char_dictionary,
                                                        pos_dictionary=pos_dictionary,
                                                        word_dictionary_norm=word_dictionary_norm,
                                                        extend_n_batch=1, print_raw=print_raw,
                                                        get_batch_mode=False,
                                                        verbose=1)
    if output == "bpe_stat":
        bpe_counter, bpe_normed_counter, bpe_need_norm_counter = stat_bpe(iterator_multi, tokenizer,mask_token_index=mask_token_index,
                                                                          null_token_index=BERT_MODEL_DIC[bert_model]["vocab_size"])
        return bpe_counter, bpe_normed_counter, bpe_need_norm_counter
    elif output == "embedding":
        out = get_embedding_mat(iterator_multi,tokenizer,bert_token_classification, null_token_index=1,
                                early_breaking=early_breaking,
                                mask_token_index=mask_token_index, verbose=verbose)
        return out


def n_token_with_k_norm(dic):
    n_norm_stat = {}
    n_norm_normed_stat = {}
    for norm in dic:
        n_norm = len(norm)
        if n_norm in n_norm_stat:
            n_norm_stat[n_norm] += 1
        else:
            n_norm_stat[n_norm] = 1
    total = 0
    for norm, count in n_norm_stat.items():
        total += count
    for norm, count in n_norm_stat.items():
        n_norm_normed_stat[norm] = count/total
    return n_norm_stat, n_norm_normed_stat


if __name__ == "__main__":

    bpe_counter, bpe_normed_counter, bpe_need_norm_counter = bpe_statistics_on_data(LIU_DEV, "../dictionaries")


    pdb.set_trace()

