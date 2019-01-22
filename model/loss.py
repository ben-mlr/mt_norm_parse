from torch.autograd import Variable
import torch.nn as nn
import torch
from model.generator import Generator
import matplotlib.pyplot as plt
import numpy as np
from io_.info_print import printing
from io_.dat.constants import PAD_ID_NORM_NOT_NORM
import pdb
from env.project_variables import LOSS_DETAIL_TEMPLATE
import time
from toolbox.sanity_check import get_timing
from collections import OrderedDict

class LossCompute:

    def __init__(self, generator,
                 auxilliary_task_norm_not_norm=False,
                 weight_binary_loss=1,
                 opt=None, pad=1, use_gpu=False, timing=False, verbose=0):
        self.generator = generator
        self.loss_distance = nn.CrossEntropyLoss(reduce=True, ignore_index=pad)
        printing("LOSS : weight_binary_loss is set to {}", var=(weight_binary_loss), verbose=verbose, verbose_level=2)
        self.loss_binary = nn.CrossEntropyLoss(reduce=True, ignore_index=PAD_ID_NORM_NOT_NORM) if auxilliary_task_norm_not_norm else None
        self.weight_binary_loss = weight_binary_loss if self.loss_binary is not None else None
        if use_gpu:
            printing("Setting loss_distance to GPU mode", verbose=verbose, verbose_level=3)
            self.loss_distance = self.loss_distance.cuda()
            if self.loss_binary is not None:
                self.loss_binary = self.loss_binary.cuda()
        self.opt = opt

        self.loss_details_template = LOSS_DETAIL_TEMPLATE.copy()
        if auxilliary_task_norm_not_norm:
            self.loss_details_template["loss_binary"] = 0

        self.use_gpu = use_gpu
        self.verbose = verbose
        self.timing = timing

    def __call__(self, x, y, x_norm_not_norm=None, y_norm_not_norm=None):

        loss_details = self.loss_details_template.copy()
        if self.loss_binary is not None:
            assert x_norm_not_norm is not None and y_norm_not_norm is not None, \
                "ERROR : auxilliary_task_norm_not_norm was set to True but x_norm_not_norm or" \
                " x_norm_not_norm was not y_norm_not_norm "
        printing("LOSS decoding states {} ", var=(x.size()), verbose=self.verbose, verbose_level=3)
        start = time.time() if self.timing else None
        x = self.generator(x)
        generate_time, start = get_timing(start)
        if self.use_gpu:
            printing("LOSS : use gpu is True", self.verbose, verbose_level=3)
        printing("LOSS input x candidate scores size {} ", var=[x.size()],verbose= self.verbose, verbose_level=4)
        printing("LOSS input y observations size {} ", var=[y.size()], verbose=self.verbose, verbose_level=4)
        printing("LOSS input x candidate scores   {} ", var=(x), verbose=self.verbose,verbose_level=5)
        printing("LOSS input x candidate scores  reshaped {} ", var=(x.view(-1, x.size(-1))),
                 verbose=self.verbose,verbose_level=5)
        printing("LOSS input y observations {} reshaped {} ", var=(y, y.contiguous().view(-1)),
                 verbose=self.verbose, verbose_level=5)
        # we remove empty words in the gold
        y = y[:, :x.size(1), :]
        y_norm_not_norm = y_norm_not_norm[:, :x_norm_not_norm.size(1)] if y_norm_not_norm is not None else None
        printing("TYPE  y {} is cuda ", var=(y.is_cuda), verbose=0, verbose_level=5)
        reshaping, start = get_timing(start)
        loss = self.loss_distance(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
        loss_distance_time, start = get_timing(start)
        loss_binary = self.loss_binary(x_norm_not_norm.contiguous().view(-1, x_norm_not_norm.size(-1)),
                                       y_norm_not_norm.contiguous().view(-1)) if self.loss_binary is not None else None

        multi_task_loss = loss+self.weight_binary_loss*loss_binary if self.loss_binary is not None else loss
        loss_details["overall_loss"] = multi_task_loss
        loss_details["loss_seq_prediction"] = loss
        if self.loss_binary:
            loss_details["loss_binary"] = loss_binary*self.weight_binary_loss
            loss_details["other"]["ponderation_binary"] = self.weight_binary_loss

        if loss_binary is not None:
            printing("LOSS BINARY loss size {} ", var=(str(loss_binary.size())), verbose=self.verbose, verbose_level=3)
            printing("TYPE  loss_binary {} is cuda ", var=(loss_binary.is_cuda), verbose=0, verbose_level=5)
        printing("LOSS loss size {} ", var=(str(loss.size())), verbose=self.verbose, verbose_level=3)
        printing("TYPE  loss {} is cuda ", var=(loss.is_cuda), verbose=0, verbose_level=5)
        # define loss_distance as --> Cross-entropy
        if self.opt is not None:
            multi_task_loss.backward()
            loss_backwrd_time, start = get_timing(start)
            printing("Optimizing", self.verbose, verbose_level=3)
            self.opt.step()
            step_opt_time, start = get_timing(start)
            self.opt.zero_grad()
            # TODO : should it be before ?
            zero_gradtime, start = get_timing(start)
        else:
            printing("WARNING no optimization : is backward required here ? (loss.py) ", verbose=self.verbose, verbose_level=3)
        if self.timing:
            print("run loss timing : {} ".format(OrderedDict([("loss_distance_time", loss_distance_time),
                                                             ("reshaping",reshaping), ("generate_time", generate_time),
                                                              ("loss_backwrd_time", loss_backwrd_time),
                                                               ("step_opt_time",step_opt_time), ("zerp_gradtime", zero_gradtime)])))
        return multi_task_loss, loss_details


# TODO add test for the loss
loss_display = False
loss_compute_test = False
if loss_compute_test:
    gene = Generator(hidden_size_decoder=10, voc_size=5)
    loss, details = LossCompute(generator=gene)
    input = torch.randn(2, 4, 10, requires_grad=True)
    print("input -> ", input.size())
    target = torch.empty(2, 4, dtype=torch.long).random_(5)
    print(target)
    print("target --> ", target.size())
    loss(input, target)
#scrit = LabelSmoothing(5, 0, 0.1)


def loss(x):
    d = x + 3 * 1
    loss, details = LossCompute(generator=gene)
    dist = loss.loss_distance
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],])
    #print(predict)
    return dist(Variable(predict.log()),
                 Variable(torch.LongTensor([1])))
if loss_display:
    plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
    plt.show()