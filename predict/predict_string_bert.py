from env.importing import *
from model.bert_tools_from_core_code.tokenization import BertTokenizer
from model.bert_normalize import get_indexes, from_bpe_token_to_str, realigne
from io_.info_print import printing


def interact_bert(bert_token_classification,  tokenizer, topk=1, verbose=1, use_gpu=False):

    printing("INFO : input_string should be white space tokenized", verbose=verbose, verbose_level=1)

    input_string = input("What would you like to normalize ? type STOP to stop ")
    if input_string == "STOP":
        print("ENDING interaction")
        return None, 0

    input_string = ["[CLS] " + input_string + " [SEP]"]

    input_tokens_tensor, input_segments_tensors, inp_bpe_tokenized, \
    input_alignement_with_raw, input_mask = get_indexes(input_string, tokenizer, verbose, use_gpu)
    token_type_ids = torch.zeros_like(input_tokens_tensor)

    logits = bert_token_classification(input_tokens_tensor, token_type_ids, input_mask)

    predictions_topk = torch.argsort(logits, dim=-1, descending=True)[:, :, :topk]

    sentence_pred = from_bpe_token_to_str(predictions_topk, topk, tokenizer=tokenizer, pred_mode=True)

    sentence_pred_aligned = []

    for top in range(topk):
        realign_sent = realigne(sentence_pred[top], input_alignement_with_raw, remove_null_str=True)
        assert len(realign_sent) == 1, "ERROR : only batch len 1 accepted here (we are doing interaction)"
        printing("{} top-pred : bpe {}", var=[top, realign_sent],
                 verbose_level=2, verbose=verbose)
        realign_sent = " ".join(realign_sent[0])
        sentence_pred_aligned.append(realign_sent)

    return input_string, sentence_pred_aligned

