from env.importing import *

VERBOSE_1_LOG_EVERY_x_BATCH = 25
DEBUG = False

LOGGING_SPECIFIC_INFO_AVAILABLE = ["cuda"]


def printing(message, verbose, verbose_level, var=None):
    """
    # if verbose is string then has to be in LOGGING_SPECIFIC_INFO_AVAILABLE
    :param message:
    :param verbose:
    :param verbose_level:
    :param var:
    :param carasteristic:
    :return:
    """
    if isinstance(verbose_level, str):
        assert verbose_level in LOGGING_SPECIFIC_INFO_AVAILABLE, "ERROR unavailble verbosity {} not in {}".format(verbose_level, LOGGING_SPECIFIC_INFO_AVAILABLE)

    verbose_level = 0 if DEBUG else verbose_level
    if isinstance(verbose, int) and isinstance(verbose_level, int):
        if verbose >= verbose_level:
            if var is not None:
                if isinstance(var, Iterable):
                    print(message.format(*var))
                else:
                    print(message.format(var))
            else:
                print(message)
    elif isinstance(verbose, str) and isinstance(verbose_level, str):
        if verbose == "cuda":
            print(message.format(*var))
    sys.stdout.flush()


def print_char_seq(active=False, nbatch=None, sent_len=None, word_len=None, char_array=None, dic=None):
    if active:
        to_print = [" ".join([dic.get_instance(char_array[batch, word, char_i]) for char_i in range(word_len)])
                    for batch in range(nbatch) for word in range(sent_len)]
        for e in to_print:
            print("raw text : ", e)


def disable_tqdm_level(verbose, verbose_level):
    return False if isinstance(verbose,int) and verbose >= verbose_level else True