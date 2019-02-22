import numpy as np
import torch

UNK_ID = 0


def construct_word_embedding_table(word_dim, word_dictionary, word_embed_init_toke2vec, warmup=False, verbose=1):
    scale = np.sqrt(3.0 / word_dim)
    # +1 required for default value
    table = np.zeros([len(word_dictionary) + 1, word_dim], dtype=np.float32)
    # WARNING: it means that unfilled commodities will get 0 which is the defulat index !!
    if verbose >= 1:
        print("Initializing table with shape {} based onword_dictionary and word_dim  ".format(table.shape))
    table[UNK_ID, :] = np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)

    oov = 0
    inv = 0
    for word, index in word_dictionary.items():

        if word in word_embed_init_toke2vec:
            embedding = word_embed_init_toke2vec[word]
            inv += 1
        elif word.lower() in word_embed_init_toke2vec:
            embedding = word_embed_init_toke2vec[word.lower()]
            inv += 1
        else:
            embedding = np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)

            oov += 1

        table[index, :] = embedding
    print('word OOV: %d/%d (%f rate (percent)) in %d' % (
    oov, len(word_dictionary) + 1, 100 * float(oov / (len(word_dictionary) + 1)), inv))
    return torch.from_numpy(table)