
VERBOSE_1_LOG_EVERY_x_BATCH = 25


def printing(message, verbose, verbose_level):
    if verbose >= verbose_level:
        print(message)


def print_char_seq(active=False, nbatch=None, sent_len=None, word_len=None, char_array=None, dic=None):
    if active:
        to_print = [" ".join([dic.get_instance(char_array[batch, word, char_i]) for char_i in range(word_len)]) for batch in range(nbatch) for word in range(sent_len)]
        for e in to_print:
            print("raw text : ", e)


def disable_tqdm_level(verbose,verbose_level):
    return False if verbose >= verbose_level else True