
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import NullFormatter
from sklearn.manifold import TSNE
import numpy as np


def plot_sentence_embedding(gold_embedding, noisy_embedding,perplexity=30):
    #
    # pdb.set_trace()
    id2lab = {0:"normed",
              1:"ugc"}
    X = np.concatenate((gold_embedding ,noisy_embedding), axis=0)
    y = np.array([0 for _ in range(gold_embedding.shape[0])] +
                 [1 for _ in range(noisy_embedding.shape[0])], dtype=np.int)

    embeddings = TSNE(n_components=2, init='pca', verbose=2, perplexity=perplexity, n_iter=500).fit_transform(X)
    xx = embeddings[:, 0]
    yy = embeddings[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    num_classes =2
    colors = cm.Spectral(np.linspace(0, 1, 5))
    labels = np.arange(num_classes)

    for i in range(num_classes):
        ax.scatter(xx[y == i], yy[y == i], color=colors[i], label=id2lab[i], s=12)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(loc='best', scatterpoints=1, fontsize=10)

    # plt.savefig(output_folder + str(layer_id) +".pdf", format='pdf', dpi=600)
    plt.show()
