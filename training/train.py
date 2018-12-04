import time
import torch
import numpy as np
from io_.info_print import printing, VERBOSE_1_LOG_EVERY_x_BATCH


def run_epoch(data_iter, model, loss_compute, verbose=0, i_epoch=None,
              n_epochs=None, n_batches=None, empty_run=False,
              log_every_x_batch=VERBOSE_1_LOG_EVERY_x_BATCH):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    printing("Starting {} epoch out of {} ".format(i_epoch+1, n_epochs), verbose, verbose_level=1)

    for i, batch in enumerate(data_iter):
        printing("Starting {} batch out of {} batches".format(i+1, n_batches), verbose, verbose_level=2)

        # model forward path (produce decoder state for each input, output pairs with relevant masking)
        #out = model.forward(batch.input_seq, batch.output_seq_x,
        #                    batch.input_seq_mask, batch.output_seq_y, input_word_len=batch.input_seq_len)
        #nput_seq, output_seq, input_mask, input_word_len, output_mask
        printing("DATA : \n input Sequence {} \n Target sequence {} ".format(batch.input_seq, batch.output_seq), verbose, verbose_level=5)
        if not empty_run:
            if True:
                out = model.forward(input_seq=batch.input_seq,
                                output_seq=batch.output_seq_x,
                                input_mask=batch.input_seq_mask,
                                input_word_len= batch.input_seq_len,
                                output_mask=batch.output_mask,
                                output_word_len=batch.output_seq_len)
            
            # compute loss , (compute score over decoding states then softmax and Cross entropy )
        else:
            out = 0
            printing("DATA : \n input Sequence {} \n Target sequence {} ".format(batch.input_seq, batch.output_seq), verbose, verbose_level=1)
        if not empty_run:
            loss = loss_compute(out, batch.output_seq_y)#, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens.type(torch.FloatTensor)
            tokens += batch.ntokens.type(torch.FloatTensor)
            elapsed = torch.from_numpy(np.array(time.time() - start)).float()
            start = time.time() if verbose>=2 else start
            printing("Epoch {} Step: {}  Loss: {}  Tokens per Sec: {}  ".format(i_epoch+1, i, loss / batch.ntokens.type(torch.FloatTensor), tokens / elapsed), verbose=verbose, verbose_level=2)
            tokens = 0 if verbose >= 2 else tokens
            if i % log_every_x_batch == 1 and verbose == 1:
                print("Epoch {} Step: {}  Loss: {}  Tokens per Sec: {}  ".format(i_epoch, i, loss / batch.ntokens.type(torch.FloatTensor), tokens / elapsed))
                start = time.time()
                tokens = 0
        else:
            total_loss, total_tokens = 0, 1
    if verbose >= 1 and not empty_run:
        printing("INFO : {} epoch done ".format(n_epochs), verbose, verbose_level=1)
        printing("Loss epoch {} is  {} total out of {} tokens ".format(i_epoch, total_loss/total_tokens, total_tokens), verbose, verbose_level=1)

    #training_report = {"n_epochs":n_epochs, "batch_size": batch.input_seq.size(0), "time_training": None, "total_tokens" : total_tokens, "loss": total_loss / total_tokens}

    return total_loss / total_tokens