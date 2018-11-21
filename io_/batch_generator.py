from torch.autograd import Variable
import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class MaskBatch(object):
    def __init__(self, input_seq, output_seq, pad=0):
        # input mask
        self.input_seq = input_seq
        self.input_seq_mask = (input_seq != pad).unsqueeze(-2)
        #pdb.set_trace()
        # output mask
        self.output_seq = output_seq
        #pdb.set_trace()
        if output_seq is not None:
            self.output_seq_x = output_seq[:, :-1]
            self.output_seq_y = output_seq[:, 1:]
            self.output_mask = self.make_mask(self.output_seq_x, pad)
            self.ntokens = (self.output_seq_y != pad).data.sum()

    @staticmethod
    def make_mask(output_seq, padding):
        "create a mask to hide paddding and future work"
        mask = (output_seq != padding).unsqueeze(-2)
        #pdb.set_trace()
        #pdb.set_trace()
        mask = mask & Variable(subsequent_mask(output_seq.size(-1)).type_as(mask.data))
        #pdb.set_trace()
        return mask


# test
if __name__=="__main__":
    plt.figure(figsize=(5, 5))
    plt.imshow(subsequent_mask(20)[0])
    #plt.show()
    data_out = torch.cat((torch.empty(1, 4, dtype=torch.long).random_(1,5), torch.zeros(1,4,dtype=torch.long)), dim=1)
    data_in = torch.cat((torch.empty(1, 4, dtype=torch.long).random_(1,4),torch.zeros(1,3,dtype=torch.long)),dim=1)
    print("DATA IN {} ".format(data_in))
    print("DATA OUT {} ".format(data_out))
    batch = MaskBatch(data_in, data_out, pad=0)
    print("INPUT MASK {} , output mask {} ".format(batch.input_seq_mask, batch.output_mask))