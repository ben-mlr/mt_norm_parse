from model.seq2seq import LexNormalizer, Generator
import torch.nn as nn
import os
import torch
from training.train import run_epoch
from io_.data_iterator import data_gen
from model.loss import LossCompute
import matplotlib.pyplot as plt


# hyperparameters
V = 5
lr = 0.001
model = LexNormalizer(generator=Generator, char_embedding_dim=5, hidden_size_encoder=11, hidden_size_decoder=11, verbose=0)
# optimizer
adam = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)


# reporting
training_loss = []

verbose = 1
EPOCHS = 100
seq_len = 10
generalize_extra = 5
if __name__ == "__main__":

    for epoch in range(EPOCHS):
        model.train()
        run_epoch(data_gen(V=V, batch=100, nbatches=20, seq_len=seq_len), model, LossCompute(model.generator, opt=adam))
        model.eval()
        loss = run_epoch(data_gen(V, batch=10, nbatches=10,seq_len=seq_len+generalize_extra), model, LossCompute(model.generator))
        training_loss.append(loss)
        if verbose>=1:
            print("Final Loss {} ".format(loss))

    plt.title("Training Loss with {} lr".format(lr))
    plt.xlabel("epoch")
    plt.plot(training_loss)
    dir_fig = os.path.join(os.path.dirname(os.path.realpath(__file__)),"test_logs","test-{}ep-{}V-{}lr-{}seq-{}gener.png".format(EPOCHS, V, lr, seq_len, generalize_extra))
    plt.savefig(dir_fig )
    print("Loss of the test saved to {} ".format(dir_fig))
    plt.show()
