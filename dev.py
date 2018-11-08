from model.seq2seq import LexNormalizer
import torch.nn as nn
import torch
from torch.autograd import Variable

import numpy as np
import pdb
import

pdb.set_trace = lambda: 1

data_in = torch.LongTensor([[1., 2., 4., 0]])
data_out = torch.LongTensor([[1, 0, 0, 0]])

model = LexNormalizer()
forward = model.forward(input_seq=data_in, output_seq=data_out)
pdb.set_trace()

print(forward)
loss = model.loss(input_seq=data_in, output_seq=data_out)
pdb.set_trace()
print(loss.size(), loss)


class MaskBatch(object):
    def __init__(self, input_seq, output_seq):
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.output_seq_x = output_seq[:, :-1]
        self.output_seq_y = output_seq[:, 1:]
        self.output_mask = self.make_mask(self.output_seq)
    @staticmethod
    def make_mask(output_seq, padding):
        "create a mask to hide paddding and future work"
        mask = (output_seq!=padding).unsqueeze(-2)
        pdb.set_trace()
        mask = mask & Variable()

        return mask


def train():
    model.zero_grad()
    loss = 0
    pass
    #for char_i in range(input_lenght):



# TODO
##confirm dimensions output
##TODO 2 Design test masking :
##-- : for test : you can try with padding sequences (should I pad with zeros : yes) : print the mask along forward pass to see (with various batch , seq len ...)
##-- : both input and output
## come up with a test to make sure the seq output never see the target !
## + a test on the softmax score !

