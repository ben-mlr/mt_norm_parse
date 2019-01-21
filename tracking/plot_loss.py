import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.patches as mpatches
import git
from io_.info_print import printing


def simple_plot(final_loss, loss_ls, loss_2=None, epochs=None, V=None, seq_len=None,
                label="", label_2="",
                dir="/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/test_/test_logs",
                lr=None, save=False, show=True, prefix="test", verbose=0, verbose_level=1):

    if loss_2 is None:
        assert len(label_2) == 0, "Label_2 should be '' as loss_2 is None "
    if len(label_2) > 0:
        assert len(label) > 0, "label should be specified as label_2 is "

    printing("Final Loss to be plotted {} ".format(final_loss), verbose=0, verbose_level=1)
    plt.figure()
    plt.title("Training Loss with after {} epo (lr {}) ".format(epochs, lr))
    plt.xlabel("epoch")
    color_train = "red"
    plt.plot(loss_ls, label="plot1", color=color_train)
    patches = [mpatches.Patch(color=color_train, label=label)]
    if loss_2 is not None:
        color_dev = "blue"
        plt.plot(loss_2, label="plot2",color=color_dev)
        patches.append(mpatches.Patch(color=color_dev, label=label_2))
    plt.legend(handles=patches)
    dir_fig = os.path.join(dir, "{}-{}-plo-seq.png".format(prefix, "last", V, lr, seq_len))
    if save:
        plt.savefig(dir_fig)
        printing("Learning curve saved at {} ", var= ([dir_fig]), verbose=verbose, verbose_level=verbose_level)

    if show:
        print("Not Showing loss")
        #plt.show()
    plt.close()
    return dir_fig


def simple_plot_ls(final_loss, losses_ls, x_axis,
                   epochs=None, V=None, seq_len=None,
                   labels="", n_tokens_info="", filter_by_label="",
                   dir="/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/test_/test_logs",
                   lr=None, save=False, show=True,
                   label_color_0="", label_color_1="",
                   prefix="simple_plot_ls", verbose=1, verbose_level=1):

    printing("Final Loss to be plotted {} ".format(final_loss), verbose=0, verbose_level=0)
    plt.figure(figsize=(20, 10))
    plt.title("Score after {} epo (lr {}) only {} ".format(epochs, lr, filter_by_label))
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    colors_palette_1 = ["red", "tomato", "pink"]
    colors_palette_0 = ["turquoise", "blue", "lightblue"]
    color_counters = [-1, -1]
    patches = []
    assert len(losses_ls) == len(labels), "ERROR {}and {}".format(losses_ls, labels)
    for loss_ls,  label in zip(losses_ls, labels):
        if filter_by_label in label:
            color_counters[0] = color_counters[0] + 1 if label_color_0 in label else color_counters[0]
            color_counters[1] = color_counters[1] + 1 if label_color_1 in label else color_counters[1]
            _color = colors_palette_0[color_counters[0]] if label_color_0 in label else colors_palette_1[color_counters[1]]
            assert color_counters[0]<3, "ERROR "
            plt.plot(x_axis, loss_ls, label="plot1", color=_color)#colors_palette[color_counters[0]])
            patches.append(mpatches.Patch(color=_color,#colors_palette[color_counters[0]],
             label=label))
    plt.legend(handles=patches, fontsize='medium')
    #dir_fig = os.path.join(dir, "{}-{}-filtered_by-{}-{}lr-plot-seq.png".format(prefix, "last", V, lr, filter_by_label, seq_len))
    dir_fig = os.path.join(dir, "{}-filtered_by-{}-lr-plot-seq.png".format(prefix, filter_by_label))
    if save:
        plt.savefig(dir_fig)
        printing("Scoring curves saved at {} ", var=([dir_fig]), verbose=verbose, verbose_level=verbose_level)
    plt.close()
    return dir_fig

if __name__=="__main__":
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    print(sha)