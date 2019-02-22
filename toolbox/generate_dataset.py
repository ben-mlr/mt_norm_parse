
import random
import numpy as np


CHAR_VOCAB = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*,.?!@#")


def generate_word(len):
    return "".join([random.choice(CHAR_VOCAB) for _ in range(len)])


def generate_data(dir,n_sent,max_n_words_per_sent, max_words_len):

    with open(dir,"w") as f:
        for ind, raw_row in enumerate(range(n_sent)):
            if ind>0:
                f.write("\n")
            f.write('# text = "id": {} \n'.format(ind))
            n_words = np.random.randint(2, max_n_words_per_sent)
            for ids_word in range(1, n_words+1):
                len_word = np.random.randint(1, max_words_len)
                word = generate_word(len_word)
                #print("word",word)
                f.write("{}\t{}\t_\t_\t_\t_\t1\t_\t_\tNorm={}|\n".format(ids_word,word, word ))
    print("DATA GENERATED at {} ".format(dir))


if __name__=="__main__":
    generate_data("../data/copy_paste_train.conll", n_sent=100000, max_n_words_per_sent=30, max_words_len=15)
    generate_data("../data/copy_paste_dev.conll", n_sent=10000, max_n_words_per_sent=30, max_words_len=15)
    generate_data("../data/copy_paste_test.conll", n_sent=20000, max_n_words_per_sent=30, max_words_len=15)