import os
from io_.info_print import printing
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def show_attention(prediction_word, input_word, attentions, model_full_name=None, dir_save=None,
                   show=False,
                   save=False,verbose=1):
    # Set up figure with colorbar
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels(['']+prediction_word, rotation=0)
    ax.set_yticklabels([''] + input_word)
    plt.xlabel("Predicted  Normalization word")
    plt.ylabel("Noisy source word")
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #show_plot_visdom()
    if save:
        model_full_name = "no_model" if model_full_name is None else model_full_name
        file_name = "{}_model-{}_pred_word-attention.png".format(model_full_name, "".join(prediction_word))
        dir_save = "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/test_/test_plot_attention" if dir_save is None else dir_save
        dir_save = os.path.join(dir_save, file_name)
        plt.savefig(dir_save)
        printing("Attention saved in {}", var=[dir_save], verbose=verbose, verbose_level=1)
    if show:
        plt.show()
    plt.close()