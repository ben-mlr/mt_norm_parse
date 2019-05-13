from env.importing import *
from io_.info_print import printing
from io_.dat.constants import NULL_STR_TO_SHOW, TOKEN_BPE_BERT_START, TOKEN_BPE_BERT_SEP, PAD_POS, PAD_ID_BERT


def aligned_output(input_tokens_tensor, output_tokens_tensor,
                   input_alignement_with_raw, output_alignement_with_raw,
                   null_token_index, mask_token_index,
                   input_mask, use_gpu, verbose=1):
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
    input_tokens_tensor_aligned = torch.empty_like(input_tokens_tensor)
    output_tokens_tensor_aligned_sent_ls = []
    input_tokens_tensor_aligned_sent_ls = []
    new_alignement_with_input_ls = []
    new_input_mask_ls = []

    assert input_mask.size(0) == input_tokens_tensor.size(0) and input_mask.size(1) and input_tokens_tensor.size(1)

    # TODO : realign MASK also here
    add_mask = False
    for ind_sent, (_input_alignement_with_raw, _output_alignement_with_raw) in enumerate(zip(input_alignement_with_raw,
                                                                                             output_alignement_with_raw)):
        _i_input = 0
        _i_output = 0
        _1_to_n_token_counter = 0
        _1_to_n_token = False
        not_the_end_of_input = True
        output_tokens_tensor_aligned_sent = []
        input_tokens_tensor_aligned_sent = []
        new_input_mask = []
        new_alignement_with_input = []
        padded_reached_ind = 0

        while not_the_end_of_input:
            # did we reach padding on the src side ?
            padded_reach = _input_alignement_with_raw[_i_input] == 1000

            if not (padded_reach and len(_output_alignement_with_raw) ==_i_output):
                # usual case
                # n to 1 : the index of the output is faster than on the input side :
                #  one (at least) extra btoken on the src side
                try:
                    n_to_1_token = _input_alignement_with_raw[_i_input] < _output_alignement_with_raw[_i_output]
                except Exception as e:
                    print(e)
                    print("ERROR {} index , len(input) {} , index {} len(output) {} "
                          .format(_i_input, len(_input_alignement_with_raw),
                                  _i_output, len(_output_alignement_with_raw)))
                    n_to_1_token = _input_alignement_with_raw[_i_input] < _output_alignement_with_raw[_i_output]
                # n to 1 : the index of the output is slower than on the input side :
                #  one (at least) extra btoken on the src side
                _1_to_n_token = _input_alignement_with_raw[_i_input] > _output_alignement_with_raw[_i_output]
                end_output_with_padded_reach = 0
            else:
                # we reach padding on input and the end on the output
                end_output_with_padded_reach = 1
                n_to_1_token, _1_to_n_token = 0, 0
            # if the output token don't change we have to shift the input of one
            if _1_to_n_token:
                printing("WARNING : _1_to_n_token --> next batch ",
                         verbose=verbose, verbose_level=2)
                _1_to_n_token_counter+=1
                #break
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
                input_tokens_tensor_aligned_sent.append(input_tokens_tensor[ind_sent, _i_input])
                new_input_mask.append(input_mask[ind_sent, _i_input].item())
                # index alignement
                new_alignement_with_input.append(_input_alignement_with_raw[_i_input])
            # --
            elif _1_to_n_token:
                output_tokens_tensor_aligned_sent.append(output_tokens_tensor[ind_sent, _i_output])
                add_mask = True
                input_tokens_tensor_aligned_sent.append(mask_token_index)
                # mask update
                new_input_mask.append(1)
                new_alignement_with_input.append(_input_alignement_with_raw[_i_input-_1_to_n_token])
            # --
            elif not end_output_with_padded_reach:
                appending = output_tokens_tensor[ind_sent, _i_output]
                output_tokens_tensor_aligned_sent.append(appending)
                input_tokens_tensor_aligned_sent.append(input_tokens_tensor[ind_sent, _i_input])
                new_input_mask.append(input_mask[ind_sent, _i_input].item())
                new_alignement_with_input.append(_input_alignement_with_raw[_i_input])
            else:
                output_tokens_tensor_aligned_sent.append(0)
            _i_input += 1-_1_to_n_token
            # padded_reached_ind is to make sure we 're not facing problem in the output
            _i_output += (1 - n_to_1_token - padded_reached_ind)

            if _i_input == len(_input_alignement_with_raw):
                not_the_end_of_input = False
        if _1_to_n_token:
            #break
            pass
        printing("TO FILL output {} index {}", var=[output_tokens_tensor_aligned_sent, ind_sent], verbose=verbose,
                 verbose_level=3)

        #output_tokens_tensor_aligned[ind_sent] = torch.Tensor(output_tokens_tensor_aligned_sent)
        #input_tokens_tensor_aligned[ind_sent] = torch.Tensor(input_tokens_tensor_aligned_sent)
        #output_tokens_tensor_aligned_sent_ls.append(torch.Tensor(output_tokens_tensor_aligned_sent))
        output_tokens_tensor_aligned_sent_ls.append(output_tokens_tensor_aligned_sent)
        #input_tokens_tensor_aligned_sent_ls.append(torch.Tensor(input_tokens_tensor_aligned_sent))
        #new_input_mask_ls.append(torch.Tensor(new_input_mask))
        new_input_mask_ls.append(new_input_mask)
        #new_alignement_with_input_ls.append(torch.Tensor(new_alignement_with_input))
        new_alignement_with_input_ls.append(new_alignement_with_input)
        input_tokens_tensor_aligned_sent_ls.append(input_tokens_tensor_aligned_sent)

    assert len(output_tokens_tensor_aligned_sent_ls) == len(input_tokens_tensor_aligned_sent_ls)

    max_token = max(max([len(out) for out in output_tokens_tensor_aligned_sent_ls]),
                    max([len(inp) for inp in input_tokens_tensor_aligned_sent_ls]))
    output_tokens_tensor_aligned = torch.empty((len(output_tokens_tensor_aligned_sent_ls), max_token)).long()
    input_tokens_tensor_aligned = torch.empty((len(output_tokens_tensor_aligned_sent_ls), max_token)).long()
    input_tokens_tensor_aligned_sent_ls_tensor = torch.empty((len(output_tokens_tensor_aligned_sent_ls), max_token)).long()
    input_mask_aligned = torch.empty((len(output_tokens_tensor_aligned_sent_ls), max_token)).long()
    for ind_sent, (out, inp, mask, index_aligned) in enumerate(zip(output_tokens_tensor_aligned_sent_ls,
                                                                   input_tokens_tensor_aligned_sent_ls,
                                                                   new_input_mask_ls,
                                                                   new_alignement_with_input_ls)):

        output_tokens_tensor_aligned[ind_sent] = torch.Tensor(out+[PAD_ID_BERT for _ in range(max_token-len(out))])
        input_tokens_tensor_aligned[ind_sent] = torch.Tensor(inp + [PAD_ID_BERT for _ in range(max_token-len(inp))])

        input_mask_aligned[ind_sent] = torch.Tensor(mask+[0 for _ in range(max_token-len(inp))])
        input_tokens_tensor_aligned_sent_ls_tensor[ind_sent] = torch.Tensor(index_aligned+[1000 for _ in range(max_token-len(inp))])
    # we take care of padding here
    input_mask_aligned[input_tokens_tensor_aligned == PAD_ID_BERT] = 0
    if use_gpu:
        input_tokens_tensor_aligned_sent_ls_tensor = input_tokens_tensor_aligned_sent_ls_tensor.cuda()
        # input_tokens_tensor_aligned = input_tokens_tensor_aligned.cuda() # !! WE REPLACED input_tokens_tensor_aligned with tokens_tensor_aligned_sent_ls_tensor
        output_tokens_tensor_aligned = output_tokens_tensor_aligned.cuda()
        input_mask_aligned = input_mask_aligned.cuda()
        input_tokens_tensor_aligned = input_tokens_tensor_aligned.cuda()

    return output_tokens_tensor_aligned, input_tokens_tensor_aligned, input_tokens_tensor_aligned_sent_ls_tensor, input_mask_aligned, _1_to_n_token_counter


def realigne(ls_sent_str, input_alignement_with_raw, null_str, mask_str, tasks,
             remove_null_str=True, remove_mask_str=False, remove_extra_predicted_token=False):
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
            if token == mask_str:
                token = "X" if not remove_mask_str else ""
            if "normalize" in tasks:
                if index == former_index:
                    if token.startswith("##"):
                        former_token += token[2:]
                    else:
                        former_token += token
            elif "pos" in tasks:
                # we just if ignore token
                if index == former_index:
                    pass
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
            if "pos" == tasks[0] and False:
                if token == PAD_POS:
                    pass
                if index != former_index or _i + 1 == len(index_ls):
                    new_sent.append(former_token)
                    former_token = token
                    if trigger_end_sent:
                        break
                else:
                    new_sent.append(token)
        new_sent_ls.append(new_sent[1:])
    return new_sent_ls
