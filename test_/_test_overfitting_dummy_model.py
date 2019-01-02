from model.seq2seq import LexNormalizer
from model.generator import Generator
import torch.nn as nn
import os
import torch
from training.epoch_train import run_epoch
from io_.data_iterator import data_gen_dummy
from model.loss import LossCompute
import matplotlib.pyplot as plt
from tracking.plot_loss import simple_plot
from tqdm import tqdm
from io_.info_print import disable_tqdm_level
import pdb
# hyperparameters
V = 5
lr = 0.001
# optimizer


verbose = 1
model = LexNormalizer(generator=Generator, char_embedding_dim=5, hidden_size_encoder=12, voc_size=9, output_dim=50,
                      hidden_size_sent_encoder=10,
                      hidden_size_decoder=11, verbose=verbose)
adam = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
# reporting
training_loss = []
nbatches = 1
EPOCHS = 20
seq_len = 10
generalize_extra = 5


def _test_overfit_dummy():
    pdb.set_trace = lambda: 1
    for epoch in tqdm(range(EPOCHS), disable_tqdm_level(verbose=verbose, verbose_level=0)):
        model.train()
        run_epoch(data_gen_dummy(V=V, batch=2, nbatches=nbatches, sent_len=seq_len, verbose=verbose),
                  model, LossCompute(model.generator, opt=adam, verbose=verbose), verbose=verbose, i_epoch=epoch,
                  n_epochs=EPOCHS, n_batches=nbatches)
        model.eval()
        loss = run_epoch(data_gen_dummy(V, batch=2, nbatches=10, sent_len=seq_len + generalize_extra), model,
                         LossCompute(model.generator), i_epoch=epoch, n_epochs=EPOCHS)
        training_loss.append(loss)
        if verbose >= 1:
            print("Final Loss {} ".format(loss))

    simple_plot(final_loss=loss, loss_ls=training_loss, save=True, show=True, epochs=EPOCHS, seq_len=seq_len, V=V,
                verbose=1,
                lr=lr, prefix="**test-dummy-fake_data.png")

    print("TEST LOSS {} should be very small and plot should be decreasing  ".format(loss))

    while True:
        anwser = input("Action to take : check loss and curve if fine press 1 otherwise 0 : ")
        if anwser=="0":
            raise Exception("Test did not pass")
            break
        elif anwser=="1":
            print("_test_overfit_dummy pass ")
            break
        else:
            print("answer is in { 0 , 1 } ")
            continue


if __name__ == "__main__":
    _test_overfit_dummy()