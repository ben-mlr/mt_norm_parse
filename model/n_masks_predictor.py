from env.importing import torch
from io_.dat.constants import NUM_LABELS_N_MASKS


def pred_n_bpe(input):
    output = torch.empty_like(input).long()
    for ind_sent in range(input.size(0)):
        count_1 = 1
        for ind_word in range(input.size(1)):
            if input[ind_sent, ind_word] == 1:
                output[ind_sent, ind_word] = -1
                if count_1 == 1:
                    ind_multi_bpe = ind_word-1
                count_1 += 1
            elif input[ind_sent, ind_word] == 0:
                # reached the end of the multi-bpe
                if ind_word >= 0 and input[ind_sent, ind_word - 1] == 1:
                    output[ind_sent, ind_multi_bpe] = min(count_1, NUM_LABELS_N_MASKS-1)
                    count_1 = 1
                output[ind_sent, ind_word] = 1
            else:
                raise(Exception("input[ind_sent, ind_word] is neither 0 nor 1 but {}".format(input[ind_sent, ind_word])))
    return output
