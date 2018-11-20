import time
import torch
import numpy as np


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        # model forward path (produce decoder state for each input, output pairs with relevant masking)
        out = model.forward(batch.input_seq, batch.output_seq_x,
                            batch.input_seq_mask, batch.output_seq_y)
        # compute loss , (compute score over decoding states then softmax and Cross entropy )
        #print(" --OK ", out, batch.output_seq_y, batch.ntokens)
        loss = loss_compute(out, batch.output_seq_y)#, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens.type(torch.FloatTensor)
        tokens += batch.ntokens.type(torch.FloatTensor)
        if i % 50 == 1:
            #elapsed = torch.from_numpy(np.array(time.time() - start))
            print("Epoch Step: %d Loss: %f Tokens per Sec: " % (i, loss / batch.ntokens.type(torch.FloatTensor)))
            #, tokens / elapsed))
            start = time.time()
            tokens = 0

    print("Total loss {} {} type total tokens {} {} type".format(total_loss,total_loss.dtype, total_tokens,total_tokens.dtype))
    total_tokens = total_tokens.type(torch.FloatTensor)
    print("Total loss {} {} type total tokens {} {} type".format(total_loss, total_loss.dtype, total_tokens,total_tokens.dtype))

    return total_loss / total_tokens