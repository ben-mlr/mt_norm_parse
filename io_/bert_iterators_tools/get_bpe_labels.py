
from env.importing import pdb, OrderedDict, np, torch, time
from io_.dat.constants import  PAD_ID_BERT, PAD_ID_TAG, PAD_ID_HEADS, ROOT_HEADS_INDEX, END_HEADS_INDEX


def get_mask_input(input_tokens_tensor, use_gpu):
    new_input = np.array(input_tokens_tensor.cpu())
    _input_mask = [[0 if new_input[ind_sent][ind_tok] == PAD_ID_BERT else 1 for ind_tok in range(len(new_input[ind_sent]))] for ind_sent in range(len(new_input))]
    input_mask = torch.Tensor(_input_mask).long()
    if use_gpu:
        input_mask = input_mask.cuda()

    return input_mask


# as CLS is appended at the begining of each sentences : we need to adjust the labels for it
CLS_ADJUST = 0


def get_bpe_label_word_level_task(labels, batch, input_tokens_tensor, input_alignement_with_raw, use_gpu, graph_labels=False):
    output_tokens_tensor = np.array(labels.cpu())
    new_input = np.array(input_tokens_tensor.cpu())
    len_max = max([len(sent) for sent in new_input])
    new_input = [[inp for inp in sent] + [PAD_ID_BERT for _ in range(len_max - len(sent))] for sent
                 in new_input]
    # we mask bpe token that have been split (we don't mask the first bpe token of each word)
    padding = PAD_ID_BERT
    _input_mask = [[0 if new_input[ind_sent][ind_tok] == padding or input_alignement_with_raw[ind_sent][ind_tok-1] ==
                         input_alignement_with_raw[ind_sent][ind_tok] else 1 for ind_tok in range(len(new_input[ind_sent]))] for ind_sent in range(len(new_input))]
    if graph_labels:
        # for each sentence : each bpe token : we count the number of multi-bpe token before it
        def get_cumulated_non_first_bpe_counter(sent):
            counter = 0
            new_sent = []
            counter_former = 0
            cumulated = 0
            for ind, token in enumerate(sent):
                if ind+1 < len(sent) and token == sent[ind+1] and token != 1000:
                    counter += 1
                elif token != 1000:
                    new_sent.append(counter_former+cumulated)
                    cumulated += counter_former
                    counter_former = counter
                    counter = 0
            return new_sent
        def test_get_cumulated_non_first_bpe_counter():
            assert [0, 0, 0, 1, 1, 1, 3, 3, 3, 5, 5, 5] == get_cumulated_non_first_bpe_counter([0, 1,2 ,2 ,3, 4, 5, 5, 5, 6, 7, 8, 8, 8, 9, 10 ,11, 1000])
            assert [0, 0, 0, 1, 1, 1, 3, 3, 3, 5, 5, 5] == get_cumulated_non_first_bpe_counter([0, 1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 8, 8, 8, 9, 10, 11])
            #print("TEST passed ")
        test_get_cumulated_non_first_bpe_counter()

        cumulate_shift = [get_cumulated_non_first_bpe_counter(input_alignement_with_raw[ind_sent]) for ind_sent in range(len(input_alignement_with_raw))]
        #cumulate_shift = [[sum(multi_bpe_token[ind_sent][:ind]) for ind, _ in enumerate(multi_bpe_token[ind_sent])] for ind_sent in range(len(multi_bpe_token))]
        #cumulate_shift = [[sum(multi_bpe_token[ind_sent][:ind+1]) for ind, _ in enumerate(multi_bpe_token[ind_sent]) if
        #            ind + 1 < len(multi_bpe_token[ind_sent])
        #            and not (multi_bpe_token[ind_sent][ind] != 1
        #            and multi_bpe_token[ind_sent][ind] != multi_bpe_token[ind_sent][ind + 1])]
        #                  for ind_sent in range(len(multi_bpe_token))]
    output_tokens_tensor_new = []
    for ind_sent in range(len(_input_mask)):
        output_tokens_tensor_new_ls = []
        shift = 0
        for ind_tok in range(len(_input_mask[ind_sent])):
            mask = _input_mask[ind_sent][ind_tok]
            try:
                label = output_tokens_tensor[ind_sent, ind_tok - shift]
                if graph_labels:
                    # as CLS is appended at the begining of each sentences : we need to adjust the labels for it
                    # TODO : !! cumulated is indexed by bpe tokenized sequence : label is indexed by original index : should get cumulated[lnew_index_label]
                    # CLS and SEQ points to the first token indexed by -1 so become 1
                    #pdb.set_trace()
                    if label not in [ROOT_HEADS_INDEX, END_HEADS_INDEX] and cumulate_shift[ind_sent][label] > 0:
                        label += cumulate_shift[ind_sent][label]
                        #pdb.set_trace()
                    label += CLS_ADJUST
            except Exception as e:
                try:
                    assert input_alignement_with_raw[ind_sent][ind_tok] == 1000, "ERROR we should have reached the end of get labels also "
                    label = PAD_ID_TAG if not graph_labels else PAD_ID_HEADS # output_tokens_tensor[ind_sent, output_tokens_tensor.shape[1] - 1]
                except Exception as f:
                    print("ERROR (get_bpe_labels): we reached the end of output labels but input is not done ! ", f)
                    print("ERROR ind_send:{} ind_tok {} shift {} output_tokens_tensor {} alignement {} -  {}".format(ind_sent, ind_tok, shift, output_tokens_tensor,
                                                                                                                     input_alignement_with_raw[ind_sent], e))
                    print("ERROR ind_send ", batch.raw_input, batch.raw_output)
                    pdb.set_trace()
                    #label = output_tokens_tensor[ind_sent, output_tokens_tensor.shape[1] - 1]
                    raise(e)
                #raise(e)

            if mask == 0:
                # 1 for _PAD_POS
                pad = PAD_ID_TAG if not graph_labels else PAD_ID_HEADS
                output_tokens_tensor_new_ls.append(pad)
                shift += 1
            else:
                output_tokens_tensor_new_ls.append(label)
                current_token_shift = 0
                # TODO : next step is pointing other bpe to single other bpe
                #if False and graph_labels and input_alignement_with_raw[ind_sent][ind_tok] != 1000:
                #    # pointing toward first bpe of the given token
                #    current_token_shift += 1
                #    output_tokens_tensor_new_ls.append(input_alignement_with_raw[ind_sent][ind_tok-current_token_shift]+cumulate_shift[ind_sent][ind_tok])
        output_tokens_tensor_new.append(output_tokens_tensor_new_ls)

    def sanity_test_parsing_label(labels, output_tokens_tensor_new, input_alignement_with_raw, cumulate_shift):
        for sent in range(labels.size(0)):
            ind_max = len(cumulate_shift[sent])-1
            for _ in range(5):
                ind = np.random.choice(range(ind_max))
                # the new label must be equal to the old one at the corresponding position + 1 + the number of non-first-bpe-token (original indexing of the label)
                if output_tokens_tensor_new[sent][ind] not in [ROOT_HEADS_INDEX+1, END_HEADS_INDEX, PAD_ID_HEADS]:
                    try:
                        assert output_tokens_tensor_new[sent][ind] == labels[sent, input_alignement_with_raw[sent][ind]]+CLS_ADJUST+cumulate_shift[sent][labels[sent, input_alignement_with_raw[sent][ind]]], \
                        "ERROR sent {} ind word {} " \
                        "new {} and old {} cumulted {} ".format(sent, ind, output_tokens_tensor_new[sent][ind],
                                                            labels[sent, input_alignement_with_raw[sent][ind]], cumulate_shift[sent][ind])
                    except AssertionError as e:
                        print(e)
                        pdb.set_trace()
                    #print("TEST passed for sent {} word {}".format(sent, ind))

    if graph_labels:
        pass
        start_time = time.time()
        sanity_test_parsing_label(labels, output_tokens_tensor_new, input_alignement_with_raw, cumulate_shift)
        #print("TIME TEST", time.time()-start_time)

    output_tokens_tensor = torch.Tensor(output_tokens_tensor_new).long()
    head_mask = torch.Tensor(_input_mask).long()
    input_tokens_tensor = torch.Tensor(new_input).long()
    if use_gpu:
        head_mask = head_mask.cuda()
        input_tokens_tensor = input_tokens_tensor.cuda()
        output_tokens_tensor = output_tokens_tensor.cuda()

    return output_tokens_tensor, head_mask, input_tokens_tensor





def get_label_per_bpe(tasks, batch, input_tokens_tensor, input_alignement_with_raw, use_gpu, tasks_parameters):
    """
    returns input, input masks and output for each tasks
    (in regard to the task type , so far only word level is supported)
    """
    #  TODO : should be done in pytorch + reducancies with get_index

    label_per_task = OrderedDict()
    input_mask, output_tokens_tensor = None, None
    if False and tasks[0] == "pos" and len(tasks) == 1:

        output_tokens_tensor = np.array(batch.pos.cpu())
        new_input = np.array(input_tokens_tensor.cpu())
        len_max = max([len(sent) for sent in new_input])
        new_input = [[inp for inp in sent] + [PAD_ID_BERT for _ in range(len_max - len(sent))] for sent in new_input]
        # we mask bpe token that have been split (we don't mask the first bpe token of each word)
        _input_mask = [[0 if new_input[ind_sent][ind_tok] == PAD_ID_BERT or
                             input_alignement_with_raw[ind_sent][ind_tok - 1] ==
                             input_alignement_with_raw[ind_sent][ind_tok] else 1 for ind_tok in
                        range(len(new_input[ind_sent]))] for ind_sent in range(len(new_input))]
        output_tokens_tensor_new = []

        for ind_sent in range(len(_input_mask)):
            output_tokens_tensor_new_ls = []
            shift = 0
            for ind_tok in range(len(_input_mask[ind_sent])):
                mask = _input_mask[ind_sent][ind_tok]
                try:
                    label = output_tokens_tensor[ind_sent, ind_tok - shift]
                except Exception as e:
                    print(
                        "ERROR ind_send:{} ind_tok {} shift {} output_tokens_tensor {} {}".format(ind_sent, ind_tok, shift, output_tokens_tensor, e))
                    print("ERROR ind_send ", batch.raw_input, batch.raw_output)
                    label = output_tokens_tensor[ind_sent, output_tokens_tensor.shape[1] - 1]

                if mask != 0:
                    output_tokens_tensor_new_ls.append(label)
                else:
                    # 1 for _PAD_POS
                    output_tokens_tensor_new_ls.append(PAD_ID_TAG)
                    shift += 1
            output_tokens_tensor_new.append(output_tokens_tensor_new_ls)

        output_tokens_tensor = torch.Tensor(output_tokens_tensor_new).long()
        input_mask = torch.Tensor(_input_mask).long()
        input_tokens_tensor = torch.Tensor(new_input).long()

        output_tokens_tensor_aligned = output_tokens_tensor[:, : input_tokens_tensor.size(1)]
        output_tokens_tensor_aligned = output_tokens_tensor_aligned.contiguous()
        token_type_ids = torch.zeros_like(input_tokens_tensor)

        if use_gpu:
            output_tokens_tensor_aligned = output_tokens_tensor_aligned.cuda()
            input_tokens_tensor = input_tokens_tensor.cuda()
            for lab in label_per_task:
                label_per_task[lab] = label_per_task[lab].cuda()

    else:
        head_masks = OrderedDict()
        for simul_task in tasks:
            for task in simul_task:
                for task_batch_name in tasks_parameters[task]["label"]:
                    task_batch = eval("batch.{}".format(task_batch_name))
                    # we handle all word level tasks in the same way
                    assert tasks_parameters[task]["prediction_level"] == "word", "ERROR only word level task supported here so far"
                    if tasks_parameters[task]["prediction_level"] == "word":
                        output_tokens_tensor, head_mask, input_tokens_tensor = get_bpe_label_word_level_task(task_batch, batch,
                                                                                                             input_tokens_tensor,
                                                                                                             input_alignement_with_raw,
                                                                                                             use_gpu, graph_labels=bool("parsing_heads" == task_batch_name))
                        head_masks[task] = head_mask
                        output_tokens_tensor_aligned = output_tokens_tensor[:, : input_tokens_tensor.size(1)]
                        output_tokens_tensor_aligned = output_tokens_tensor_aligned.contiguous()
                        if use_gpu:
                            output_tokens_tensor_aligned = output_tokens_tensor_aligned.cuda()
                        # if the task has several label : we just appen the label name to the task in the label dictionary
                        label_name = task_batch_name #task if len(tasks_parameters[task]["label"]) == 1 else task+"_"+task_batch_name
                        label_per_task[label_name] = output_tokens_tensor_aligned
                    else:
                        raise(Exception("ERROR : only word level supported so far "))

    token_type_ids = torch.zeros_like(input_tokens_tensor)

    if use_gpu:
        token_type_ids = token_type_ids.cuda()

    return head_masks, input_tokens_tensor, token_type_ids, label_per_task
