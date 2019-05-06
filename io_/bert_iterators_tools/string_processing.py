from env.importing import *
from io_.dat.constants import TOKEN_BPE_BERT_SEP, TOKEN_BPE_BERT_START, PAD_ID_BERT, PAD_BERT
from io_.info_print import printing

from toolbox.sanity_check import sanity_check_data_len

def preprocess_batch_string_for_bert(batch):
    """
    adding starting and ending token in raw sentences
    :param batch:
    :return:
    """
    for i in range(len(batch)):
        batch[i][0] = TOKEN_BPE_BERT_START
        batch[i][-1] = TOKEN_BPE_BERT_SEP
        batch[i] = " ".join(batch[i])
    return batch


def get_indexes(list_pretokenized_str, tokenizer, verbose, use_gpu):
    """
    from pretokenized string : it will bpe-tokenize it using BERT 'tokenizer'
    and then convert it to tokens ids
    :param list_pretokenized_str:
    :param tokenizer:
    :param verbose:
    :param use_gpu:
    :return:
    """
    all_tokenized_ls = [tokenizer.tokenize(inp) for inp in list_pretokenized_str]
    tokenized_ls = [tup[0] for tup in all_tokenized_ls]
    aligned_index = [tup[1] for tup in all_tokenized_ls]
    segments_ids = [[0 for _ in range(len(tokenized))] for tokenized in tokenized_ls]

    printing("DATA : bpe tokenized {}", var=[tokenized_ls], verbose=verbose, verbose_level="raw_data")

    ids_ls = [tokenizer.convert_tokens_to_ids(inp) for inp in tokenized_ls]
    max_sent_len = max([len(inp) for inp in tokenized_ls])
    ids_padded = [inp + [PAD_ID_BERT for _ in range(max_sent_len - len(inp))] for inp in ids_ls]
    aligned_index_padded = [[e for e in inp] + [1000 for _ in range(max_sent_len - len(inp))] for inp in aligned_index]
    segments_padded = [inp + [PAD_ID_BERT for _ in range(max_sent_len - len(inp))] for inp in segments_ids]
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


def from_bpe_token_to_str(bpe_tensor, topk, pred_mode,
                          tokenizer, null_token_index, null_str):
    """
    pred_mode allow to handle gold data also (which only have 2 dim and not three)
    :param bpe_tensor:
    :param topk: int : number of top prediction : will arrange them with all the top1 all the 2nd all the third...
    :param pred_mode: book
    :return:
    """
    predictions_topk_ls = [[[bpe_tensor[sent, word, top].item() if pred_mode else bpe_tensor[sent, word].item()
                             for word in range(bpe_tensor.size(1))] for sent in range(bpe_tensor.size(0))]
                           for top in range(topk)]
    sent_ls_top = [[tokenizer.convert_ids_to_tokens(sent_bpe, special_extra_token=null_token_index,
                                                    special_token_string=null_str)
                    for sent_bpe in predictions_topk] for predictions_topk in predictions_topk_ls]

    if not pred_mode:
        sent_ls_top = sent_ls_top[0]
    return sent_ls_top
