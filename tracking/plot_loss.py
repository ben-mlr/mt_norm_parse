import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.patches as mpatches
import git
from io_.info_print import printing


def simple_plot(final_loss, loss_ls, loss_2=None, epochs=None, V=None, seq_len=None,
                label="", label_2="",
                dir="/Users/benjaminmuller/Desktop/Work/INRIA/dev/mt_norm_parse/test/test_logs",
                lr=None, save=False, show=True, prefix="test", verbose=0, verbose_level=1):

    if loss_2 is None:
        assert len(label_2) == 0, "Label_2 should be '' as loss_2 is None "
    if len(label_2) > 0:
        assert len(label) > 0, "label should be specified as label_2 is "

    printing("Final Loss to be plotted {} ".format(final_loss), verbose=0, verbose_level=1)
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
        plt.savefig(dir_fig )
        printing("Loss of the test saved to {} ".format(dir_fig), verbose=verbose, verbose_level=verbose_level)

    if show:
        print("Not Showing loss")
        #plt.show()
    return dir_fig

if __name__=="__main__":
    #simple_plot(0, [0, 1], [0, 2])
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    print(sha)