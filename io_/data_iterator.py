from torch.autograd import Variable
import torch
import numpy as np
import pdb
from io_.batch_generator import MaskBatch


def data_gen(V, batch, nbatches,seq_len=10):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, seq_len)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield MaskBatch(src, tgt, 0)

run = 0
if run:
    iter = data_gen(V=5, batch=1, nbatches=2)
    for ind, batch in enumerate(iter):
        print("BATCH NUMBER {} ".format(ind))
        print("SRC : ", batch.input_seq)
        print("SRC MASK : ", batch.input_seq_mask)
        print("TARGET : ", batch.output_seq)
        print("TARGET MASK : ", batch.output_mask)