from torch.autograd import Variable
import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from io_.info_print import printing


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class MaskBatch(object):
    def __init__(self, input_seq, output_seq, pad=0, verbose=0):
        # input mask
        assert output_seq.size(0)>1, "ERROR  batch_size should be strictly above 1 "
        self.input_seq = input_seq
        self.input_seq_mask = (input_seq != pad).unsqueeze(-2)

        _input_seq_mask = self.input_seq_mask.clone()
        _input_seq_mask[:, :, -1] = 0
        # Handle long unpadded sequence
        self.input_seq_len = torch.argmin(_input_seq_mask, dim=2)
        printing("BATCH : SOURCE true dim {} ".format(self.input_seq.size()),verbose, verbose_level=3)
        printing("BATCH : SOURCE input_seq_len  {} ".format(self.input_seq_len),verbose, verbose_level=5)


        self.output_seq = output_seq
        if output_seq is not None:

            self.output_seq_x = output_seq[:, :-1]
            _output_mask_x = (self.output_seq_x != pad).unsqueeze(-2)
            # Handle long unpadded sequence
            # we force the last token to be masked so that we ensure the argmin computation we'll be correct
            _output_mask_x[:, :,-1] = 0
            self.output_seq_len = torch.argmin(_output_mask_x, dim=2) #if not bool(_output_mask_x.sum().data == _output_mask_x.size(0)*_output_mask_x.size(2)) else
            self.output_seq_y = output_seq[:, 1:]
            self.output_mask = self.make_mask(self.output_seq_x, pad)
            printing("BATCH : OUTPUT self.output_mask  subsequent {} {} ".format(self.output_mask.size(),  self.output_mask),  verbose, verbose_level=5)
            printing("BATCH : OUTPUT self.output_seq_x,  subsequent {} {} ".format(self.output_seq_x.size(), self.output_seq_x),  verbose, verbose_level=5)

            self.ntokens = (self.output_seq_y != pad).data.sum()
            # dealing with bach_size == 1
            if self.output_seq_len.size(0) > 1:
                output_seq_len, perm_idx = self.output_seq_len.squeeze().sort(0, descending=True)
                self.output_seq_y = self.output_seq_y[perm_idx, :]
            else:
                # TODO should be able to handle batch_size == 1 but is not
                output_seq_len, perm_idx  = self.output_seq_len, torch.zeros([1],dtype=torch.long)

            printing("BATCH : TARGET before packed true {} ".format(self.output_seq_y),verbose, verbose_level=5)
            self.output_seq_y = pack_padded_sequence(self.output_seq_y, output_seq_len.squeeze().cpu().numpy(),
                                                     batch_first=True)

            self.output_seq_y, lenghts = pad_packed_sequence(self.output_seq_y, batch_first=True)
            printing("BATCH : TARGET true dim {} ".format(self.output_seq_y.size()),verbose, verbose_level=3)
            printing("BATCH : TARGET after packed true {} ".format(self.output_seq_y),verbose, verbose_level=5)


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