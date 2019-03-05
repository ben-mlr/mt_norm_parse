import time
import torch
import numpy as np
from env.project_variables import AVAILABLE_TASKS
from io_.info_print import printing, VERBOSE_1_LOG_EVERY_x_BATCH
import pdb
from toolbox.sanity_check import get_timing
import time
from collections import OrderedDict


def update_loss_details_dic(total_loss_details_dic, loss_details_current_dic):
    "update loss details with current batch loss to compute epochs loss average"
    for loss_key in total_loss_details_dic.keys():
        if loss_key != "other":
            total_loss_details_dic[loss_key] += loss_details_current_dic[loss_key]
    return total_loss_details_dic


def divide_loss_details_n_tokens(total_loss_details_dic, n_tokens):
    "divide loss by n_tokens"
    for loss_key, val in total_loss_details_dic.items():
        if loss_key != "other":
            total_loss_details_dic[loss_key] = float(val)/int(n_tokens)
    return total_loss_details_dic


def run_epoch(data_iter, model, loss_compute,
              weight_binary_loss,
              weight_pos_loss,
              ponderation_normalize_loss,
              verbose=0, i_epoch=None,
              n_epochs=None, n_batches=None, empty_run=False, timing=False,
              multi_task_mode="all", clipping=None,
              step=0, proportion_pred_train=None,
              pos_batch=False,
              # should be added in policy

              log_every_x_batch=VERBOSE_1_LOG_EVERY_x_BATCH):
    "Standard Training and Logging Function"

    assert multi_task_mode in AVAILABLE_TASKS
    _start = time.time()
    total_tokens = 0
    total_loss = 0
    total_loss_details = loss_compute.loss_details_template.copy()
    tokens = 0
    i_epoch = -1 if i_epoch is None else i_epoch
    n_epochs = -1 if n_epochs is None else n_epochs
    batch_time_start = time.time()
    for i, (batch,_) in enumerate(data_iter):
        batch_time_, batch_time_start = get_timing(batch_time_start)
        printing("Starting {} batch out of {} batches", var=(i+1, n_batches), verbose= verbose, verbose_level=2)
        if not empty_run:
            start = time.time() if timing else None
            out, out_word, pos_pred_state, norm_not_norm_hidden, attention = model.forward(input_seq=batch.input_seq,
                                                                                           output_seq=batch.output_seq_x,
                                                                                           input_word_len=batch.input_seq_len,
                                                                                           output_word_len=batch.output_seq_len,
                                                                                           proportion_pred_train=proportion_pred_train,
                                                                                           word_embed_input=batch.input_word)
            forward_time, start = get_timing(start)
        else:
            out = 0, _
            printing("DATA : \n input Sequence {} \n Target sequence {} ", var=(batch.input_seq, batch.output_seq),
                     verbose=verbose, verbose_level=1)
        if not empty_run:
            loss, loss_details_current = loss_compute(x=out, y=batch.output_seq_y,
                                                      x_norm_not_norm=norm_not_norm_hidden, y_norm_not_norm=batch.output_norm_not_norm,
                                                      y_word=batch.output_word, x_word_pred=out_word,
                                                      y_pos=batch.pos,  x_pos=pos_pred_state, pos_batch=pos_batch,
                                                      clipping=clipping,
                                                      weight_binary_loss=weight_binary_loss,
                                                      weight_pos_loss=weight_pos_loss,
                                                      ponderation_normalize_loss=ponderation_normalize_loss,
                                                      step=i+step)#, batch.ntokens)

            loss_time, start = get_timing(start)
            total_loss += loss.item()
            total_loss_details = update_loss_details_dic(total_loss_details, loss_details_current)
            total_tokens += batch.ntokens.type(torch.FloatTensor)
            tokens += batch.ntokens.type(torch.FloatTensor)
            elapsed = torch.from_numpy(np.array(time.time() - _start)).float()
            _start = time.time() if verbose >= 2 else _start
            _loss = loss / float(batch.ntokens)
            printing("Epoch {} Step: {}  Loss: {}  Tokens per Sec: {} , total tokens {}  ", var=(
                i_epoch+1, i, _loss, tokens / elapsed, tokens), verbose=verbose, verbose_level=2)
            tokens = 0 if verbose >= 2 else tokens
            if i % log_every_x_batch == 1 and verbose == 1:
                print("Epoch {} Step: {}  Loss: {}  Tokens per Sec: {} , total tokens {}".format(i_epoch, i, loss / float(batch.ntokens), tokens / elapsed, tokens))
                _start = time.time()
                tokens = 0
        else:
            total_loss, total_tokens = 0, 1
        batch_time_start = time.time()
        if timing:
            print("run epoch timing : {}".format(OrderedDict([("forward_time", forward_time), ("loss_time", loss_time),
                                                             ("batch_time_start", batch_time_)])))
    if not empty_run:
        printing("INFO : epoch {} done ", var=(n_epochs), verbose=verbose, verbose_level=1)
        printing("Loss epoch {} is  {} total out of {} tokens ", var=(i_epoch, float(total_loss)/int(total_tokens), total_tokens), verbose=verbose, verbose_level=1)

    total_loss_details = divide_loss_details_n_tokens(total_loss_details, total_tokens)
    #training_report = {"n_epochs":n_epochs, "batch_size": batch.input_seq.size(0), "time_training": None, "total_tokens" : total_tokens, "loss": total_loss / total_tokens}
    step = step+i

    return float(total_loss) / int(total_tokens), total_loss_details, step