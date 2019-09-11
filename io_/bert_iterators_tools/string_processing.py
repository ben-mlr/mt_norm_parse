#from env.importing import *
from env.importing import pdb, torch
from io_.dat.constants import TOKEN_BPE_BERT_SEP, TOKEN_BPE_BERT_START, PAD_ID_BERT, PAD_BERT, PAD_ID_NORM_NOT_NORM
from io_.info_print import printing

from toolbox.sanity_check import sanity_check_data_len


def rp_space_func(sent, replace_space_with=""):
    sent = [word if " " not in word else word.replace(" ", replace_space_with) for word in sent]
    return sent


def preprocess_batch_string_for_bert(batch,
                                     rp_space=False):
    """
    adding starting and ending token in raw sentences
    :param batch:
    :return:
    """
    for i in range(len(batch)):
        try:
            batch[i][0] = TOKEN_BPE_BERT_START
        except:
            pdb.set_trace()
        batch[i][-1] = TOKEN_BPE_BERT_SEP
        if rp_space:
            batch[i] = rp_space_func(batch[i])
        batch[i] = " ".join(batch[i])
    return batch


def get_indexes_src_gold(list_pretokenized_str_source, list_pretokenized_str_gold, tokenizer,
                         verbose, use_gpu, word_norm_not_norm=None):
    " mostly a copy of get_indexes adapted to handle both src and gold sequence in parrallel"
    assert word_norm_not_norm is None, "ERROR not possible in tokenize_and_bpe mode"

    # final tensors
    segments_tensors_dic = {}
    tokens_tensor_dic = {}
    printing("SOURCE {} TARGET {} ", var=[list_pretokenized_str_source, list_pretokenized_str_gold],
             verbose=verbose, verbose_level="alignement")

    all_tokenized_ls = [tokenizer.tokenize(src, gold, aligne=True) for src, gold in zip(list_pretokenized_str_source, list_pretokenized_str_gold)]

    tokenized_dic = {}
    aligned_index_dic = {}
    segments_ids_dic = {}
    ids_ls_dic = {}
    max_sent_len_dic = {}
    ids_padded_dic = {}
    aligned_index_padded_dic = {}
    segments_padded_dic = {}
    mask_dic = {}

    tokenized_dic["src"] = [tup[0] for tup in all_tokenized_ls]
    aligned_index_dic["src"] = [tup[1] for tup in all_tokenized_ls]
    tokenized_dic["gold"] = [tup[2] for tup in all_tokenized_ls]
    aligned_index_dic["gold"] = [tup[3] for tup in all_tokenized_ls]

    for sequence in ["src", "gold"]:
        segments_ids_dic[sequence] = [[0 for _ in range(len(tokenized))] for tokenized in tokenized_dic[sequence]]

        printing("DATA : bpe tokenized {}", var=[tokenized_dic[sequence]], verbose=verbose, verbose_level="raw_data")

        ids_ls_dic[sequence] = [tokenizer.convert_tokens_to_ids(inp) for inp in tokenized_dic[sequence]]
        max_sent_len_dic[sequence] = max([len(inp) for inp in tokenized_dic[sequence]])
        ids_padded_dic[sequence] = [inp + [PAD_ID_BERT for _ in range(max_sent_len_dic[sequence] - len(inp))] for inp in ids_ls_dic[sequence]]
        aligned_index_padded_dic[sequence] = [[e for e in inp] + [1000 for _ in range(max_sent_len_dic[sequence] - len(inp))] for inp in aligned_index_dic[sequence]]
        segments_padded_dic[sequence] = [inp + [PAD_ID_BERT for _ in range(max_sent_len_dic[sequence] - len(inp))] for inp in segments_ids_dic[sequence]]
        mask_dic[sequence] = [[1 for _ in inp] + [0 for _ in range(max_sent_len_dic[sequence] - len(inp))] for inp in segments_ids_dic[sequence]]

        mask_dic[sequence] = torch.LongTensor(mask_dic[sequence])
        tokens_tensor_dic[sequence] = torch.LongTensor(ids_ls_dic[sequence])
        segments_tensors_dic[sequence] = torch.LongTensor(segments_padded_dic[sequence])
        if use_gpu:
            mask_dic[sequence] = mask_dic[sequence].cuda()
            tokens_tensor_dic[sequence] = tokens_tensor_dic[sequence].cuda()
            segments_tensors_dic[sequence] = segments_tensors_dic[sequence].cuda()
        sanity_check_data_len(tokens_tensor_dic[sequence], segments_tensors_dic[sequence], tokens_tensor_dic[sequence],
                              aligned_index_padded_dic[sequence], raising_error=True)

    return tokens_tensor_dic, segments_tensors_dic, tokenized_dic, aligned_index_padded_dic, mask_dic


def mask_group(norm_not_norm, bpe_aligned_index):
    """
    norm_not_norm : 1 if group to mask (need_norm) 0 if normed
    can be use with any group of token to mask
    """
    mask_batch = []
    for i_sent, sent in enumerate(bpe_aligned_index):
        mask_sent = []
        for i in range(len(sent)):
            original_index = sent[i]
            # if original_index 1000 --> means we reached padding : mask should be 0
            norm_not = norm_not_norm[i_sent, original_index] if original_index != 1000 else 1
            mask_sent.append(1 - norm_not if norm_not != PAD_ID_NORM_NOT_NORM
                             else PAD_ID_NORM_NOT_NORM)
        if len(mask_sent) == sum([1 for mask in mask_sent if mask == 0]):
            mask_sent[1] = 1
            print("FORCING UNMASKING FOR SENT")
        mask_batch.append(mask_sent)
    return mask_batch


def get_indexes(list_pretokenized_str, tokenizer, verbose, use_gpu,
                word_norm_not_norm=None):
    """
    from pretokenized string : it will bpe-tokenize it using BERT 'tokenizer'
    and then convert it to tokens ids
    :param list_pretokenized_str:
    :param tokenizer:
    :param verbose:
    :param use_gpu:
    :return:
    """
    all_tokenized_ls = [tokenizer.tokenize_origin(inp,) for inp in list_pretokenized_str]
    tokenized_ls = [tup[0] for tup in all_tokenized_ls]

    aligned_index = [tup[1] for tup in all_tokenized_ls]
    segments_ids = [[0 for _ in range(len(tokenized))] for tokenized in tokenized_ls]

    printing("DATA : bpe tokenized {} , {} {} ", var=[tokenized_ls, len(tokenized_ls),len(tokenized_ls[0])], verbose=verbose, verbose_level="raw_data")
    printing("DATA : bpe tokenized {} , {} {} ", var=[tokenized_ls, len(tokenized_ls),len(tokenized_ls[0])], verbose=verbose, verbose_level="alignement")
    pdb.set_trace()
    ids_ls = [tokenizer.convert_tokens_to_ids(inp) for inp in tokenized_ls]
    max_sent_len = max([len(inp) for inp in tokenized_ls])
    ids_padded = [inp + [PAD_ID_BERT for _ in range(max_sent_len - len(inp))] for inp in ids_ls]
    aligned_index_padded = [[e for e in inp] + [1000 for _ in range(max_sent_len - len(inp))] for inp in aligned_index]
    segments_padded = [inp + [PAD_ID_BERT for _ in range(max_sent_len - len(inp))] for inp in segments_ids]

    if word_norm_not_norm is not None:
        mask = mask_group(word_norm_not_norm, bpe_aligned_index=aligned_index_padded)
    else:
        mask = [[1 for _ in inp]+[0 for _ in range(max_sent_len - len(inp))] for inp in segments_ids]
    mask = torch.LongTensor(mask)
    tokens_tensor = torch.LongTensor(ids_padded)
    segments_tensors = torch.LongTensor(segments_padded)
    if use_gpu:
        mask = mask.cuda()
        tokens_tensor = tokens_tensor.cuda()
        segments_tensors = segments_tensors.cuda()

    printing("DATA {}", var=[tokens_tensor], verbose=verbose, verbose_level=3)

    sanity_check_data_len(tokens_tensor, segments_tensors, tokenized_ls, aligned_index, raising_error=True)

    return tokens_tensor, segments_tensors, tokenized_ls, aligned_index_padded, mask


def from_bpe_token_to_str(bpe_tensor,  topk, pred_mode, null_token_index, null_str, task, tokenizer=None,
                          bpe_tensor_src=None,
                          pos_dictionary=None, label="normalize",
                          label_dictionary=None, mask_index=None,
                          get_string=False, verbose=1):
    """
    it actually supports not only bpe token but also pos-token
    pred_mode allow to handle gold data also (which only have 2 dim and not three)
    :param bpe_tensor:
    :param topk: int : number of top prediction : will arrange them with all the top1 all the 2nd all the third...
    :param pred_mode: book
    :return:
    """
    assert label is not None or get_string, \
        "ERROR : task {} get_string {} : one of them should be defined or True".format(label, get_string)
    if task == "mlm" and pred_mode:
        assert bpe_tensor_src is not None and mask_index is not None, "ERROR bpe_tensor_src is needed to get not-predicted token as well as mask_index "
        predictions_topk_ls = [[[bpe_tensor[sent, word, top].item() if bpe_tensor_src[sent, word].item() == mask_index else bpe_tensor_src[sent, word].item() for word in range(bpe_tensor.size(1))] for sent in range(bpe_tensor.size(0))] for top in range(topk)]
    else:
        predictions_topk_ls = [[[bpe_tensor[sent, word, top].item() if pred_mode else bpe_tensor[sent, word].item() for word in range(bpe_tensor.size(1))] for sent in range(bpe_tensor.size(0))] for top in range(topk)]

    # here all labels that require the tokenizer (should factorize it in some way)
    if label in ["normalize", "mwe_prediction", "input_masked"] or get_string:
        assert tokenizer is not None
        # requires task specific here : mlm only prediction we are interested in are
        sent_ls_top = [[tokenizer.convert_ids_to_tokens(sent_bpe, special_extra_token=null_token_index, special_token_string=null_str) for sent_bpe in predictions_topk] for predictions_topk in predictions_topk_ls]

        printing("DATA : bpe string again {}", var=[sent_ls_top], verbose=verbose, verbose_level="raw_data")
    else:
        dictionary = label_dictionary

        if label_dictionary == "index":
            sent_ls_top = [[[token_ind for token_ind in sent_bpe] for sent_bpe in predictions_topk] for predictions_topk in predictions_topk_ls]
        else:
            try:
                sent_ls_top = [[[dictionary.instances[token_ind - 1] if token_ind > 0 else "UNK" for token_ind in sent_bpe] for sent_bpe in predictions_topk] for predictions_topk in predictions_topk_ls]
            # adding more information about the exe
            except Exception as e:
                print("{} : dictionary : {} and prediction {} (POS specificity was removed )".format(e, dictionary.instances, predictions_topk_ls))
                raise(e)

    if not pred_mode:
        sent_ls_top = sent_ls_top[0]

    return sent_ls_top
