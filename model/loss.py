from torch.autograd import Variable
import torch.nn as nn
import torch
from model.generator import Generator
import matplotlib.pyplot as plt
import numpy as np
from io_.info_print import printing
from io_.dat.constants import PAD_ID_NORM_NOT_NORM, PAD_ID_WORD, PAD_ID_TAG
import pdb
from env.project_variables import LOSS_DETAIL_TEMPLATE
import time
from toolbox.sanity_check import get_timing
from toolbox.norm_not_norm import schedule_training
from collections import OrderedDict


class LossCompute:

    def __init__(self, generator,
                 auxilliary_task_norm_not_norm=False,
                 word_decoding=False,
                 char_decoding=True,
                 pos_pred=False,
                 tasks=None,
                 opt=None, pad=1, use_gpu=False, timing=False,
                 multi_task_loss_ponderation ="all", writer=None, model=None,
                 use="", verbose=0):

        #assert (word_decoding or char_decoding) and not (word_decoding and char_decoding), \
        #    "ERROR : strictly one of the two (word,char) decoding should be True "
        if tasks is None:
            tasks = []
        self.generator = generator
        assert (char_decoding and generator is not None) or (not char_decoding and generator is None), "ERROR : char_decoding {} generator {}".format(char_decoding, generator)

        self.multi_task_loss_ponderation = multi_task_loss_ponderation
        self.writer = writer
        self.loss_distance = nn.CrossEntropyLoss(reduce=True, ignore_index=pad) if char_decoding else None
        self.loss_binary = nn.CrossEntropyLoss(reduce=True, ignore_index=PAD_ID_NORM_NOT_NORM) if auxilliary_task_norm_not_norm else None
        self.loss_edit = nn.MSELoss(reduce=True) if "edit_prediction" in tasks else None
        self.loss_distance_word_level = nn.CrossEntropyLoss(reduce=True, ignore_index=PAD_ID_WORD) if word_decoding else None
        self.loss_distance_pos = nn.CrossEntropyLoss(reduce=True, ignore_index=PAD_ID_TAG) if pos_pred else None
        if use_gpu:
            printing("Setting loss_distance to GPU mode", verbose=verbose, verbose_level=3)
            if self.loss_distance is not None:
                self.loss_distance = self.loss_distance.cuda()
            if self.loss_binary is not None:
                self.loss_binary = self.loss_binary.cuda()
            if self.loss_distance_word_level is not None:
                self.loss_distance_word_level = self.loss_distance_word_level.cuda()
        self.opt = opt
        self.loss_details_template = LOSS_DETAIL_TEMPLATE.copy()
        if auxilliary_task_norm_not_norm:
            self.loss_details_template["loss_binary"] = 0
        self.model = model
        self.use_gpu = use_gpu
        self.use = use
        self.verbose = verbose
        self.timing = timing

    def __call__(self, x, y, x_norm_not_norm=None,
                 y_norm_not_norm=None,
                 y_pos=None, x_pos=None,
                 y_word=None, x_word_pred=None,
                 y_edit=None, pred_edit=None,
                 pos_batch=False,
                 weight_binary_loss=1,
                 weight_pos_loss=0,
                 ponderation_normalize_loss=0,
                 clipping=None, step=None):

        if clipping is not None:
            assert self.model is not None, "Using clipping requires passing the model in the loss"
        loss_details = self.loss_details_template.copy()
        if self.loss_binary is not None:
            assert x_norm_not_norm is not None , \
                "ERROR : auxilliary_task_norm_not_norm was set to True but x_norm_not_norm {} ".format(x_norm_not_norm)
        printing("LOSS decoding states {} ", var=[x.size()] if x is not None else None, verbose=self.verbose, verbose_level=3)
        start = time.time() if self.timing else None
        x = self.generator(x) if x is not None else None
        generate_time, start = get_timing(start)
        if self.use_gpu:
            printing("LOSS : use gpu is True", self.verbose, verbose_level=3)
        if x is not None:
            printing("LOSS input x candidate scores size {} ", var=[x.size()],verbose= self.verbose, verbose_level=4)
            printing("LOSS input y observations size {} ", var=[y.size()], verbose=self.verbose, verbose_level=4)
            printing("LOSS input x candidate scores   {} ", var=(x), verbose=self.verbose,verbose_level=5)
            printing("LOSS input x candidate scores  reshaped {} ", var=(x.view(-1, x.size(-1))),
                     verbose=self.verbose, verbose_level=5)
            printing("LOSS input y observations {} reshaped {} ", var=(y, y.contiguous().view(-1)),
                     verbose=self.verbose, verbose_level=5)
            # we remove empty words in the gold
        y = y[:, :x.size(1), :] if x is not None else None
        y_edit = y_edit[:, :pred_edit.size(1)] if y_edit is not None else None
        y_norm_not_norm = y_norm_not_norm[:, :x_norm_not_norm.size(1)] if y_norm_not_norm is not None else None
        y_word = y_word[:, :x_word_pred.size(1)] if y_word is not None and x_word_pred is not None else None
        # WE PUT THAT HERE IN CASE LATER : WE WANT To ADAPT BATCH-WISE PONDERATION
        loss_scheduler = schedule_training(multi_task_loss_ponderation=self.multi_task_loss_ponderation)
        scheduling_normalize = loss_scheduler["normalize"]
        schedule_pos = loss_scheduler["pos"]
        scheduling_norm_not_norm = loss_scheduler["norm_not_norm"]
        scheduling_edit = loss_scheduler["edit_prediction"]
        #print("sanitity", scheduling_normalize,  schedule_pos, scheduling_norm_not_norm)
        if y is not None:
            printing("TYPE  y {} is cuda ", var=(y.is_cuda), verbose=0, verbose_level=5)
        reshaping, start = get_timing(start)
        if self.loss_distance is not None:
            loss = self.loss_distance(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
        elif self.loss_distance_word_level is not None:
            loss = self.loss_distance_word_level(x_word_pred.contiguous().view(-1, x_word_pred.size(-1)), y_word.contiguous().view(-1))
            assert ponderation_normalize_loss is not None
            assert scheduling_normalize is not None
        else:
            loss = 0
        #else:
        #    print("no loss were set up for normalization")
        #    raise(Exception)
        loss_distance_time, start = get_timing(start)
        loss_binary = 0
        if self.loss_binary is not None:
            # PROBLEMS IN THE LABELS !!
            if y_norm_not_norm is not None:
                loss_binary = self.loss_binary(x_norm_not_norm.contiguous().view(-1, x_norm_not_norm.size(-1)),y_norm_not_norm.contiguous().view(-1))
            #except:
            #    pdb.set_trace()
            #    loss_binary = self.loss_binary(x_norm_not_norm.contiguous().view(-1, x_norm_not_norm.size(-1)), y_norm_not_norm.contiguous().view(-1))
            assert weight_binary_loss is not None
            assert scheduling_norm_not_norm is not None
        loss_edit = 0
        if self.loss_edit is not None and y_edit is not None:
            # self.loss_edit tells us it model has ability to predict edit, y_edit if we provided labels (could use pred_edit also in a way)
            assert pred_edit is not None, "ERROR pred_edit was given as None while model has a loss_edit and we got label"
            loss_edit = self.loss_edit(pred_edit.contiguous().view(-1), y_edit.contiguous().view(-1))

        if pos_batch and self.loss_distance_pos is not None:
            assert x_pos is not None and y_pos is not None, "ERROR x_pos and y_pos should be define "
            y_pos = y_pos[:, :x_pos.size(1)]
            loss_pos = self.loss_distance_pos(x_pos.contiguous().view(-1, x_pos.size(-1)), y_pos.contiguous().view(-1))
        else:
            loss_pos = 0

        if loss_binary is not None or (pos_batch and self.loss_distance_pos is not None) :
            multi_task_loss = ponderation_normalize_loss*scheduling_normalize*loss+\
                              weight_binary_loss*loss_binary*scheduling_norm_not_norm+\
                              schedule_pos*loss_pos*weight_pos_loss+loss_edit*scheduling_edit
        else:
            multi_task_loss = loss

        loss_details["overall_loss"] = multi_task_loss.item()
        if not isinstance(loss, int):
            loss_details["loss_seq_prediction"] = loss.item()
        else:
            loss_details["loss_seq_prediction"] = 0
        if not isinstance(loss_binary,int):
            printing("LOSS BINARY loss size {} ", var=(str(loss_binary.size())), verbose=self.verbose, verbose_level=3)
            printing("TYPE  loss_binary {} is cuda ", var=(loss_binary.is_cuda), verbose=0, verbose_level=5)

        if self.writer is not None:
            self.writer.add_scalars("loss-"+self.use,
                                    {"loss-{}-seq_pred".format(self.use): loss.clone().cpu().data.numpy() if not isinstance(loss, int) else 0,
                                     "loss-{}-seq_pred-ponderation_normalize_loss".format(self.use): loss.clone().cpu().data.numpy()*ponderation_normalize_loss if not isinstance(loss, int) else 0,
                                     "loss-{}-multitask".format(self.use): multi_task_loss.clone().cpu().data.numpy(),
                                     "loss-{}-loss_binary".format(self.use): loss_binary.clone().cpu().data.numpy() if not isinstance(loss_binary, int) else 0,
                                     "loss-{}-loss_pos-schedule_pos".format(self.use): loss_pos.clone().cpu().data.numpy()*schedule_pos*weight_pos_loss if not isinstance(loss_pos, int) else 0,
                                     "loss-{}-loss_pos".format(self.use): loss_pos.clone().cpu().data.numpy() if not isinstance(loss_pos, int) else 0,
                                     "loss-{}-loss_edit".format(self.use): loss_edit.clone().cpu().data.numpy() if not isinstance(loss_edit, int) else 0,
                                     "loss-{}-loss_edit-schedule_edit".format(self.use): scheduling_edit*loss_edit.clone().cpu().data.numpy() if not isinstance(loss_edit,
                                                                                                           int) else 0,
                                     },
                                    step)

        if self.opt is not None:
            self.opt.zero_grad()
            multi_task_loss.backward()
            loss_backwrd_time, start = get_timing(start)
            if clipping is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipping)
            gradient_clipping, start = get_timing(start)
            printing("Optimizing", self.verbose, verbose_level=3)
            self.opt.step()
            step_opt_time, start = get_timing(start)
            # TODO : should it be before ?
            zero_gradtime, start = get_timing(start)
        else:
            printing("WARNING no optimization : is backward required here ? (loss.py) ", verbose=self.verbose, verbose_level=3)
        if self.timing:
            print("run loss timing : {} ".format(OrderedDict([("loss_distance_time", loss_distance_time),
                                                              ("reshaping", reshaping), ("generate_time", generate_time),
                                                              ("loss_backwrd_time", loss_backwrd_time),
                                                              ("gradient_clipping", gradient_clipping),
                                                              ("step_opt_time", step_opt_time),
                                                              ("zerp_gradtime", zero_gradtime)])))
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