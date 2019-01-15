from torch.autograd import Variable
import torch.nn as nn
import torch
from model.generator import Generator
import matplotlib.pyplot as plt
import numpy as np
from io_.info_print import printing
import pdb
import time
from toolbox.sanity_check import get_timing
from collections import OrderedDict

class LossCompute:
    def __init__(self, generator, opt=None, pad=1, use_gpu=False, timing=False, verbose=0):
        self.generator = generator
        self.loss_distance = nn.CrossEntropyLoss(reduce=True, ignore_index=pad)
        if use_gpu:
            printing("Setting loss_distance to GPU mode", verbose=verbose, verbose_level=3)
            self.loss_distance = self.loss_distance.cuda()
        self.opt = opt
        self.use_gpu=use_gpu
        self.verbose = verbose
        self.timing = timing

    def __call__(self, x, y):

        printing("LOSS decoding states {} ", var=(x.size()), verbose=self.verbose, verbose_level=3)
        start = time.time() if self.timing else None
        x = self.generator(x)
        generate_time, start = get_timing(start)
        if self.use_gpu:
            printing("use gpu is True", self.verbose, verbose_level=3)
        
        printing("LOSS input x candidate scores size {} ", var=(x.size()),verbose= self.verbose, verbose_level=3)
        printing("LOSS input y observations size {} ", var=(y.size()), verbose=self.verbose, verbose_level=3)
        printing("LOSS input x candidate scores   {} ", var=(x), verbose=self.verbose,verbose_level=5)
        printing("LOSS input x candidate scores  reshaped {} ", var=(x.view(-1, x.size(-1))),
                 verbose=self.verbose,verbose_level=5)
        printing("LOSS input y observations {} reshaped {} ", var=(y, y.contiguous().view(-1)),
                 verbose=self.verbose, verbose_level=5)
        y = y[:,:x.size(1),:]
        printing("TYPE  y {} is cuda ", var=(y.is_cuda), verbose=0, verbose_level=5)
        reshaping, start = get_timing(start)
        loss = self.loss_distance(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
        loss_distance_time, start = get_timing(start)
        printing("LOSS loss size {} ", var=(str(loss.size())), verbose=self.verbose, verbose_level=3)
        printing("TYPE  loss {} is cuda ", var=(loss.is_cuda), verbose=0, verbose_level=5)

        # define loss_distance as --> Cross-entropy

        if self.opt is not None:
            loss.backward()
            loss_backwrd_time, start = get_timing(start)
            printing("Optimizing", self.verbose, verbose_level=3)
            self.opt.step()
            step_opt_time, start = get_timing(start)
            self.opt.zero_grad()
            # TODO : should it be before ?
            zerp_gradtime, start = get_timing(start)
        else:
            printing("WARNING no optimization : is backward required here ? (loss.py) ", verbose=self.verbose, verbose_level=3)
        if self.timing:
            print("run loss timing : {} ".format(OrderedDict([("loss_distance_time", loss_distance_time),
                                                             ("reshaping",reshaping), ("generate_time", generate_time),
                                                              ("loss_backwrd_time", loss_backwrd_time),
                                                               ("step_opt_time",step_opt_time), ("zerp_gradtime", zerp_gradtime)])))
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