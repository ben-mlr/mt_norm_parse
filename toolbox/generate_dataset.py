import random
import numpy as np
from env.project_variables import DIR_TWEET_W2V
from toolbox.load_w2v import load_emb

CHAR_VOCAB = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*,.?!@#")


def generate_word(len=None, word_level=False, word_vocab=None):
    if not word_level:
        return "".join([random.choice(CHAR_VOCAB) for _ in range(len)])
    else:
        return random.choice(word_vocab)


def generate_data(dir,n_sent, max_n_words_per_sent, max_words_len=None, word_level=False, word_embed_dir=DIR_TWEET_W2V, n_vocab=None,add_n_words=None):
    if word_level:
        assert word_embed_dir is not None and n_vocab is not None
        print("Loading", word_embed_dir)
        embed_ugc = load_emb(word_embed_dir)
        embed_ugc_vocab = list(embed_ugc.keys())[:n_vocab]
        if add_n_words:
            stard_new= np.random.randint(len(list(embed_ugc.keys()))-add_n_words-1)
            embed_ugc_vocab+=list(embed_ugc.keys())[stard_new:stard_new+add_n_words]
    else:
        embed_ugc_vocab = None
    with open(dir,"w") as f:
        for ind, raw_row in enumerate(range(n_sent)):
            if ind>0:
                f.write("\n")
            f.write('# text = "id": {} \n'.format(ind))
            n_words = np.random.randint(2, max_n_words_per_sent)
            for ids_word in range(1, n_words+1):
                if not word_level:
                    len_word = np.random.randint(1, max_words_len)
                else:
                    len_word = None
                word = generate_word(len=len_word, word_level=word_level, word_vocab=embed_ugc_vocab)
                #print("word",word)
                f.write("{}\t{}\t_\t_\t_\t_\t1\t_\t_\tNorm={}|\n".format(ids_word,word, word ))
    print("DATA GENERATED at {} ".format(dir))


if __name__=="__main__":
    char_level_data = False
    word_level_data = False
    if char_level_data:
        generate_data("../data/copy_paste_train.conll", n_sent=100000, max_n_words_per_sent=30, max_words_len=15)
        generate_data("../data/copy_paste_dev.conll", n_sent=10000, max_n_words_per_sent=30, max_words_len=15)
        generate_data("../data/copy_paste_test.conll", n_sent=20000, max_n_words_per_sent=30, max_words_len=15)
    if word_level_data:
        generate_data("../data/copy_paste_real_word_train.conll", n_sent=100000, max_n_words_per_sent=30, max_words_len=15, word_level=True, word_embed_dir=DIR_TWEET_W2V, n_vocab=7000)
        # we add 500 extra words to dev vocab
        generate_data("../data/copy_paste_real_word_dev.conll", n_sent=10000, max_n_words_per_sent=30,
                      max_words_len=15, word_level=True, word_embed_dir=DIR_TWEET_W2V, n_vocab=7000, add_n_words=500)
        generate_data("../data/copy_paste_real_word_train.conll", n_sent=20000, max_n_words_per_sent=30,
                      max_words_len=15, word_level=True, word_embed_dir=DIR_TWEET_W2V, n_vocab=7000, add_n_words=1000)
