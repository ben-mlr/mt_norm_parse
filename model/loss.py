from torch.autograd import Variable
import torch.nn as nn
import torch
from model.seq2seq import Generator
import matplotlib.pyplot as plt
import numpy as np
from io_.info_print import printing
import pdb

class LossCompute:
    def __init__(self, generator, opt=None, pad=1, verbose=0):
        self.generator = generator
        self.loss_distance = nn.CrossEntropyLoss(reduce=True, ignore_index=pad)
        self.opt = opt
        self.verbose = verbose

    def __call__(self, x, y):

        printing("LOSS decoding states {} ".format(x.size()), self.verbose, verbose_level=3)
        if True:
            x = self.generator(x)
        #else:
        #    for di in range(y.size(1)):

        printing("LOSS input x candidate scores size {} ".format(x.size()), self.verbose, verbose_level=3)
        printing("LOSS input y observations size {} ".format(y.size()), self.verbose, verbose_level=3)
        printing("LOSS input x candidate scores   {} ".format(x), self.verbose,verbose_level=5)
        printing("LOSS input x candidate scores  reshaped {} ".format(x.view(-1, x.size(-1))),
                 self.verbose,verbose_level=5)
        printing("LOSS input y observations {} reshaped {} ".format(y, y.contiguous().view(-1)),
                 self.verbose, verbose_level=5)
        pdb.set_trace()
        loss = self.loss_distance(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))

        printing("LOSS loss size {}".format(loss.size()), verbose=self.verbose, verbose_level=3)
        # define loss_distance as --> Cross-entropy
        loss.backward()
        if self.opt is not None:
            printing("Optimizing", self.verbose, verbose_level=3)
            self.opt.step()
            self.opt.zero_grad()
        return loss


# TODO add test for the loss
loss_display = False
loss_compute_test = False
if loss_compute_test:
    gene = Generator(hidden_size_decoder=10, voc_size=5)
    loss = LossCompute(generator=gene)
    input = torch.randn(2, 4, 10, requires_grad=True)
    print("input -> ", input.size())
    target = torch.empty(2, 4, dtype=torch.long).random_(5)
    print(target)
    print("target --> ", target.size())
    loss(input, target)
#scrit = LabelSmoothing(5, 0, 0.1)


def loss(x):
    d = x + 3 * 1
    loss = LossCompute(generator=gene)
    dist = loss.loss_distance
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],])
    #print(predict)
    return dist(Variable(predict.log()),
                 Variable(torch.LongTensor([1])))
if loss_display:
    plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
    plt.show()