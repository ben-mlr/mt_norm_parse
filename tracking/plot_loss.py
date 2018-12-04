import os
import matplotlib.pyplot as plt
from io_.info_print import printing


def simple_plot(final_loss, loss_ls, epochs=None, V=None, seq_len=None, lr=None, save=False, show=True, prefix="test", verbose=0, verbose_level=1):

    printing("Final Loss to be plotted {} ".format(final_loss), verbose=0, verbose_level=1)
    plt.title("Training Loss with {} lr".format(lr))
    plt.xlabel("epoch")
    plt.plot(loss_ls)
    dir_fig = os.path.join("/Users/benjaminmuller/Desktop/Work/INRIA/dev/mt_norm_parse/test/test_logs","{}-{}ep-{}V-{}lr-{}seq.png".format(prefix, epochs, V, lr, seq_len))
    if save:
        plt.savefig(dir_fig )
        printing("Loss of the test saved to {} ".format(dir_fig), verbose=verbose, verbose_level=verbose_level)

    if show:
        print("Showing loss")
        plt.show()