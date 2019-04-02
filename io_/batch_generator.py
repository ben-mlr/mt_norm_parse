from torch.autograd import Variable
import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from io_.dat.constants import PAD_ID_CHAR 
from io_.info_print import printing
import time
from io_.dat.constants import PAD_ID_CHAR, PAD_ID_WORD, CHAR_START_ID, CHAR_END_ID
from toolbox.sanity_check import get_timing
from collections import OrderedDict


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class MaskBatch(object):
    def __init__(self, input_seq, output_seq,
                 output_word=None, pos=None, input_word=None,
                 output_norm_not_norm=None, pad=0, verbose=0, timing=False, dropout_input=0.):
        # input mask
        #if not output_seq.size(0) >1:
        #    pdb.set_trace()
        #assert output_seq.size(0) >1 , "ERROR  batch_size should be strictly above 1 but is {} ".format(output_seq.size())
        # originnaly batch_size, word len
        self.input_seq = input_seq
        self.pos = pos
        self.input_word = input_word
        if dropout_input>0:
            # we put it jere so that input_seq_mask computed based on droped input_seq # migh cause trouble for input_seq_len
            #pdb.set_trace()
            droping_multiplier_word = torch.zeros_like(input_word).bernoulli_(1-dropout_input)
            # NB : we can use this droping cause we replace by 0 character and word id which corresponds to UNK token
            droping_multiplier_word[0, :] = 1
            droping_multiplier_word[input_word == 3] = 1
            droping_multiplier_word[input_word == PAD_ID_WORD] = 1
            self.input_word = torch.mul(droping_multiplier_word, input_word)
            droping_multiplier_char = torch.zeros_like(input_seq).bernoulli_(1 - dropout_input)
            droping_multiplier_char[0, :, :] = 1 # making sure first tokens are always untouched
            droping_multiplier_char[input_seq == PAD_ID_CHAR] = 1 # making sure padding are always untouched
            droping_multiplier_char[input_seq == CHAR_END_ID] = 1  # making sure padding are always untouched
            self.input_seq = torch.mul(droping_multiplier_char, input_seq)

            # TODO : add different drop out in output_seq_x and output_seq_y
        self.output_norm_not_norm = output_norm_not_norm
        self.output_word = output_word
        # unsqueeze add 1 dim between batch and word len ##- ?   ##- for commenting on context implementaiton
        start = time.time()
        self.input_seq_mask = (input_seq != pad).unsqueeze(-2)
        get_seq_mask, start = get_timing(start)
        _input_seq_mask = self.input_seq_mask
        ##- would be the same
        _input_seq_mask[:, :, :, -1] = 0
        zero_last, start = get_timing(start)
        # Handle long unpadded sequence
        ##- still last dimension : maybe 3
        self.input_seq_len = torch.argmin(_input_seq_mask.cpu(), dim=-1)
        if _input_seq_mask.is_cuda:
            self.input_seq_len = self.input_seq_len.cuda()
        get_len_input, start = get_timing(start)
        printing("BATCH : SOURCE true dim size {} ", var=(self.input_seq.size()), verbose=verbose, verbose_level=3)
        printing("BATCH : SOURCE input_seq_len  {} ", var=(self.input_seq_len), verbose=verbose, verbose_level=5)
        printing("BATCH : SOURCE input_seq_len size {} ", var=(self.input_seq_len.size()), verbose=verbose,
                 verbose_level=5)
        self.output_seq = output_seq
        if output_seq is not None:
            ##- would be last dim also !
            self.output_seq_x = output_seq[:, :, :-1]
            droping_multiplier_char_output_seq = torch.zeros_like(self.output_seq_x).bernoulli_(1 - dropout_input)
            droping_multiplier_char_output_seq[0, :, :] = 1  # making sure first tokens are always untouched
            droping_multiplier_char_output_seq[self.output_seq_x == PAD_ID_CHAR] = 1  # making sure padding are always untouched
            droping_multiplier_char_output_seq[self.output_seq_x == CHAR_END_ID] = 1  # making sure padding are always untouched
            self.output_seq_x = torch.mul(droping_multiplier_char_output_seq, self.output_seq_x)
            zero_last_output, start = get_timing(start)
            ##- ? what unsequeeze
            _output_mask_x = (self.output_seq_x != pad).unsqueeze(-2)
            get_mask_output, start = get_timing(start)
            # Handle long unpadded sequence
            # we force the last token to be masked so that we ensure the argmin computation we'll be correct
            _output_mask_x[:, :, :, -1] = 0
            zero_mask_output, start = get_timing(start)
            self.output_seq_y = output_seq[:, :, 1:]
            ##- last dim also
            self.output_seq_len = torch.argmin(_output_mask_x.cpu(), dim=-1)
            if _output_mask_x.is_cuda:
                self.output_seq_len = self.output_seq_len.cuda()
            get_len_output, start = get_timing(start)
            printing("BATCH : OUTPUT self.output_seq_x,  subsequent {} {} ", var=(self.output_seq_x.size(), self.output_seq_x), verbose= verbose, verbose_level=5)
            printing("BATCH : OUTPUT self.output_seq_len,  {} {} ", var=(self.output_seq_len.size(), self.output_seq_len), verbose=verbose, verbose_level=5)
            self.ntokens = (self.output_seq_y != pad).data.sum()
            get_n_token, start = get_timing(start)
            # dealing with bach_size == 1
            if self.output_seq_len.size(0) > 1 :
                output_y_shape = self.output_seq_y.size()
                self.output_seq_y = self.output_seq_y.view(self.output_seq_y.size(0)*self.output_seq_y.size(1), self.output_seq_y.size(2))
                output_seq_len = self.output_seq_len.view(self.output_seq_len.size(0)*self.output_seq_len.size(1))
                # self.output_seq_len = output_seq_len
                reshape_output_seq_and_len, start = get_timing(start)
                output_seq_len, perm_idx = output_seq_len.squeeze().sort(0, descending=True)
                inverse_perm_idx = torch.from_numpy(np.argsort(perm_idx.cpu().numpy()))
                self.output_seq_y = self.output_seq_y[perm_idx, :]
                reorder_output, start = get_timing(start)

            else:
                output_y_shape = self.output_seq_y.size()
                self.output_seq_y = self.output_seq_y.view(self.output_seq_y.size(0) * self.output_seq_y.size(1),self.output_seq_y.size(2))
                output_seq_len = self.output_seq_len.view(self.output_seq_len.size(0) * self.output_seq_len.size(1))
                # self.output_seq_len = output_seq_len
                reshape_output_seq_and_len, start = get_timing(start)
                output_seq_len, perm_idx = output_seq_len.squeeze().sort(0, descending=True)
                inverse_perm_idx = torch.from_numpy(np.argsort(perm_idx.cpu().numpy()))
                self.output_seq_y = self.output_seq_y[perm_idx, :]
                reorder_output, start = get_timing(start)
                print(Exception("self.output_seq_len.size(0) <=1 not suppoerted {}".format(self.output_seq_len.size(0))))
            printing("BATCH : TARGET before packed true size {} ", var=(self.output_seq_y.size()),verbose=verbose,
                     verbose_level=4)
            printing("BATCH : TARGET before packed true {} ", var=(self.output_seq_y),verbose=verbose, verbose_level=5)
            printing("BATCH : output seq len {} ", var=(output_seq_len), verbose=verbose, verbose_level=5)
            printing("BATCH : output seq len packed {} ", var=(output_seq_len.size()),verbose= verbose, verbose_level=4)
            output_seq_len[output_seq_len == 0] = 1
            zero_last_output_len, start = get_timing(start)
            self.output_seq_y = pack_padded_sequence(self.output_seq_y, output_seq_len.squeeze().cpu().numpy(),
                                                     batch_first=True)
            pack_output_y, start = get_timing(start)
            #pdb.set_trace()
            self.output_seq_y, lenghts = pad_packed_sequence(self.output_seq_y, batch_first=True, padding_value=PAD_ID_CHAR)
            pad_output_y, start = get_timing(start)
            #useless but bug raised of not packeding (would like to remove packing which I think is useless ?)

            self.output_seq_y = self.output_seq_y[inverse_perm_idx]
            reorder_output_y, start = get_timing(start)
            #print("Warning confirm shape of")
            # we reshape so that it fits tthe generated sequence
            self.output_seq_y = self.output_seq_y.view(output_y_shape[0], -1, torch.max(lenghts))
            reshape_output_seq_y, start = get_timing(start)
            printing("self.output_seq_y 1 {} ", var=(self.output_seq_y), verbose=verbose, verbose_level=6)
            printing("BATCH : TARGET true dim {} ", var=(self.output_seq_y.size()), verbose=verbose, verbose_level=3)
            printing("BATCH : TARGET after packed true {} ", var=(self.output_seq_y), verbose=verbose, verbose_level=5)
            if timing:
                print("Batch TIMING {}".format(OrderedDict([("reshape_output_seq_y",reshape_output_seq_y), ("reorder_output_y",reorder_output_y), ("pad_output_y",pad_output_y), ("zero_last_output_len",zero_last_output_len),
                                                            ("reorder_output", reorder_output), ("reshape_output_seq_and_len", reshape_output_seq_and_len), ("get_n_token", get_n_token),
                                                            ("get_len_output", get_len_output), ("zero_mask_output", zero_mask_output), ("get_mask_output", get_mask_output),
                                                            ("zero_last_output", zero_last_output),  ("get_len_input", get_len_input), ("zero_last", zero_last),
                                                            ("get_seq_mask", get_seq_mask)])))

    @staticmethod
    def make_mask(output_seq, padding):
        "create a mask to hide paddding and future work"
        mask = (output_seq != padding).unsqueeze(-2)
        mask = mask & Variable(subsequent_mask(output_seq.size(-1)).type_as(mask.data))
        return mask


# test
if __name__=="__main__":
    plt.figure(figsize=(5, 5))
    plt.imshow(subsequent_mask(20)[0])
    #plt.show()
    data_out = torch.cat((torch.empty(1, 4, dtype=torch.long).random_(2,5), torch.ones(1,4,dtype=torch.long)),  dim=1)
    data_out = torch.cat((data_out, data_out), dim=0)
    data_in = torch.cat((torch.empty(1, 4, dtype=torch.long).random_(2,4), torch.zeros(1,3,dtype=torch.long)), dim=1)
    data_in[:,0] = 2
    data_out[:,0] = 2
    #data_in = torch.cat((data_in, data_in), dim=0)
    #data_out = data_out.unsqueeze(0)
    #data_in = data_in.unsqueeze(0)

    print("DATA IN {} {} ".format(data_in,data_in.size()))
    print("DATA OUT {} {} ".format(data_out,data_out.size()))
    batch = MaskBatch(data_in, data_out, pad=1, verbose=5)
    print("INPUT MASK {} , output mask {} ".format(batch.input_seq_mask, batch.output_mask))

    # NB : sequence cannot be padded on the middle (we'll cut to the first padded sequence )