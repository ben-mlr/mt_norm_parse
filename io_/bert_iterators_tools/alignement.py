from env.importing import *
from io_.info_print import printing
from io_.dat.constants import NULL_STR_TO_SHOW, TOKEN_BPE_BERT_START, TOKEN_BPE_BERT_SEP


def aligned_output(input_tokens_tensor, output_tokens_tensor,
                   input_alignement_with_raw, output_alignement_with_raw,
                   null_token_index,
                   verbose=1):
    """
    realigning and de-tokenizng tokens (e.g : words) that have been splitted based on indexes
    :param input_tokens_tensor:
    :param output_tokens_tensor:
    :param input_alignement_with_raw:
    :param output_alignement_with_raw:
    :param verbose:
    :return:
    """
    output_tokens_tensor_aligned = torch.empty_like(input_tokens_tensor)

    for ind_sent, (_input_alignement_with_raw, _output_alignement_with_raw) in enumerate(zip(input_alignement_with_raw,
                                                                                             output_alignement_with_raw)):
        _i_input = 0
        _i_output = 0
        _1_to_n_token = False
        not_the_end_of_input = True
        output_tokens_tensor_aligned_sent = []

        padded_reached_ind = 0
        while not_the_end_of_input:

            padded_reach = _input_alignement_with_raw[_i_input] == 1000
            if not (padded_reach and len(_output_alignement_with_raw) ==_i_output):
                # usual case
                n_to_1_token = _input_alignement_with_raw[_i_input] < _output_alignement_with_raw[_i_output]
                _1_to_n_token = _input_alignement_with_raw[_i_input] > _output_alignement_with_raw[_i_output]
                end_output_with_padded_reach = 0
            else:
                # we reach padding on input and the end on the output
                end_output_with_padded_reach = 1
                n_to_1_token, _1_to_n_token = 0, 0
            # if the otuput token don't change we have to shift the input of one
            if _1_to_n_token:
                printing("WARNING : _1_to_n_token --> next batch ",
                         verbose=verbose, verbose_level="raw_data")

                break
            if padded_reach and not n_to_1_token:
                # we assert we also reached padding in the output
                # if we are in n_to_1_token it's different maybe not true
                #  same if we reached the end we handle the case with end_output_with_padded_reach
                if len(_output_alignement_with_raw) != _i_output:
                    assert _output_alignement_with_raw[_i_output] == 1000
                padded_reached_ind = 1
            if n_to_1_token:
                appending = null_token_index
                output_tokens_tensor_aligned_sent.append(appending)
            elif not end_output_with_padded_reach:
                appending = output_tokens_tensor[ind_sent, _i_output]
                output_tokens_tensor_aligned_sent.append(appending)
            else:
                output_tokens_tensor_aligned_sent.append(0)
            _i_input += 1
            # padded_reached_ind is to make sure we 're not facing problem in the output
            _i_output += (1 - n_to_1_token - padded_reached_ind)

            if _i_input == len(_input_alignement_with_raw):
                not_the_end_of_input = False

        if _1_to_n_token:
            break
        printing("TO FILL output {} index {}", var=[output_tokens_tensor_aligned_sent, ind_sent], verbose=verbose,
                 verbose_level=3)
        output_tokens_tensor_aligned[ind_sent] = torch.Tensor(output_tokens_tensor_aligned_sent)

    if input_tokens_tensor.is_cuda:
        output_tokens_tensor_aligned = output_tokens_tensor_aligned.cuda()
    return output_tokens_tensor_aligned, _1_to_n_token


def realigne(ls_sent_str, input_alignement_with_raw, null_str,
             remove_null_str=True, remove_extra_predicted_token=False):
    """
    ** remove_extra_predicted_token used iif pred mode **
    - detokenization of ls_sent_str based on input_alignement_with_raw index
    - we remove paddding and end detokenization at symbol [SEP] that we take as the end of sentence signal
    """

    assert len(ls_sent_str) == len(input_alignement_with_raw), \
        "ls_sent_str {} input_alignement_with_raw {} ".format(len(ls_sent_str), len(input_alignement_with_raw))
    new_sent_ls = []
    for sent, index_ls in zip(ls_sent_str, input_alignement_with_raw):
        assert len(sent) == len(index_ls)
        former_index = -1
        new_sent = []
        former_token = ""
        for _i, (token, index) in enumerate(zip(sent, index_ls)):
            trigger_end_sent = False
            if remove_extra_predicted_token:
                if index == 1000:
                    # we reach the end according to gold data
                    # (this means we just stop looking at the prediciton of the model (we can do that because we assumed word alignement))
                    trigger_end_sent = True
            if token == null_str:
                token = NULL_STR_TO_SHOW if not remove_null_str else ""
            if index == former_index:
                if token.startswith("##"):
                    former_token += token[2:]
                else:
                    former_token += token
            if index != former_index or _i + 1 == len(index_ls):
                new_sent.append(former_token)
                former_token = token
                if trigger_end_sent:
                    break
            # if not pred mode : always not trigger_end_sent : True (required for the model to not stop too early if predict SEP too soon)
            if (former_token == TOKEN_BPE_BERT_SEP or _i+1 == len(index_ls) and not remove_extra_predicted_token) or \
                ((remove_extra_predicted_token and (former_token == TOKEN_BPE_BERT_SEP and trigger_end_sent) or  _i+1 == len(index_ls))):
                new_sent.append(token)
                break
            former_index = index
        new_sent_ls.append(new_sent[1:])
    return new_sent_ls
