from torch.autograd import Variable
import torch
import numpy as np
import pdb
from io_.batch_generator import MaskBatch
import sys
from tqdm import tqdm
#sys.path.insert(0, "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/ELMoLex_sosweet/")
from io_.dat import conllu_data
from io_.info_print import printing, print_char_seq, disable_tqdm_level
import time
from toolbox.sanity_check import get_timing
NORM2NOISY=False


def data_gen_conllu(data, word_dictionary, char_dictionary,
                    batch_size,
                    get_batch_mode=True,
                    padding=1, print_raw=False, normalization=False, pos_dictionary=None,
                    extend_n_batch=1,
                    timing=False,
                    verbose=0):

    n_sents = data[-1]
    if extend_n_batch != 1:
        assert get_batch_mode, "ERROR extending nbatch only makes sense in get_batch True (random iteration) "

    nbatch = n_sents//batch_size*extend_n_batch  # approximated lower approximation 1.9//2 == 0

    if nbatch == 0:
        printing("INFO : n_sents < batch_size so nbatch set to 1 ", verbose=verbose, verbose_level=0)
    print("Running {} batches of {} dim (nsent : {}) (if 0 will be set to 1) ".format(nbatch, batch_size, n_sents))
    nbatch = 1 if nbatch == 0 else nbatch
    # deterministic run over all the dataset (for evaluation)
    if not normalization:
        printing("WARNING : Normalisation is False : model is a autoencoder (BOTH iteration and get cases) --> {} ",
             verbose=verbose, verbose_level=0)
    if not get_batch_mode:
        for batch in tqdm(conllu_data.iterate_batch_variable(data, batch_size=batch_size,
                                                             normalization=normalization),
                          disable=disable_tqdm_level(verbose, verbose_level=2)):

            words, word_norm, chars, chars_norm, word_norm_not_norm, pos, xpos, heads, types, masks, lengths, order_ids, raw_word_inputs, raw_lines = batch
            if not normalization:
                chars_norm = chars.clone()
            if not NORM2NOISY:
                yield MaskBatch(chars, chars_norm,  output_norm_not_norm=word_norm_not_norm, pad=padding, timing=timing,
                                output_word=word_norm, pos=pos, input_word=words,
                                verbose=verbose), order_ids
            else:
                yield MaskBatch(chars_norm, chars,  output_norm_not_norm=word_norm_not_norm, pad=padding, timing=timing,
                output_word=word_norm, pos=pos, input_word=words,
                verbose=verbose), order_ids


    # get_batch randomly (for training purpose)
    elif get_batch_mode:
        for ibatch in tqdm(range(1, nbatch+1), disable=disable_tqdm_level(verbose, verbose_level=2)):
            # word, char, pos, xpos, heads, types, masks, lengths, morph
            printing("Data : getting {} out of {} batches", var=(ibatch, nbatch+1), verbose= verbose, verbose_level=2)
            word, word_norm, char, chars_norm, word_norm_not_norm, pos, _, _, _, _, lenght, order_ids = \
                conllu_data.get_batch_variable(data,
                                               batch_size=batch_size,
                                               normalization=normalization,
                                               unk_replace=0)

            if char.size(0) <= 1:
                print("WARNING : Skip character ", char)
                continue 
            printing("TYPE {} word, char {} , chars_norm {} length {} ", var=(word.is_cuda, char.is_cuda,
                                                                              #chars_norm.is_cuda, lenght.is_cuda
                                                                              ),
                     verbose=verbose, verbose_level=5)
            assert min(lenght.data) > 0, "ERROR : min(lenght.data) is {} ".format(min(lenght.data))
            # TODO : you want to correct that : you're missing word !!

            __word_ind = 0
            word_len = char.size(2)
            if normalization:
                if word_norm_not_norm is not None:
                    printing("norm not norm {} ", var=(word_norm_not_norm[:, __word_ind]), verbose=verbose,
                             verbose_level=5)
                printing("Normalized sequence {} ", var=(chars_norm[:, __word_ind, :]), verbose=verbose, verbose_level=5)
            printing("Char {} word ind : word : {}  ", var=(__word_ind, char[:, __word_ind, :]), verbose=verbose,
                     verbose_level=5)
            _verbose = 5 if print_raw else verbose

            if _verbose >= 5:
                character_display = [" ".join([char_dictionary.get_instance(char[sent, word_ind, char_i]) for char_i in range(word_len)]) + " | NORM : {} |SENT {} WORD {}| ".format(word_norm_not_norm[sent,word_ind],sent, word_ind) for ind_sent,sent in enumerate(range(char.size(0)))
                                     for ind_w, word_ind in enumerate(range(char.size(1)))]

                word_display = [word_dictionary.get_instance(word[batch, word_ind]) + " " for batch in range(char.size(0)) for word_ind in range(char.size(1))]

                if pos_dictionary is not None:
                    pos_display = [pos_dictionary.get_instance(pos[batch, __word_ind]) + " " for batch in
                                   range(char.size(0))]
                else:
                    pos_display = None

            else:
                word_display = []
                character_display = []
                pos_display = []
            if not normalization:
                chars_norm = char.clone()

            if _verbose >= 5:
                character_norm_display = [" ".join([char_dictionary.get_instance(chars_norm[sent, word_ind, char_i])
                                               for char_i in range(chars_norm.size(2))]) +
                                          "|  NORM : {} |SENT {} WORD {}| \n ".format(word_norm_not_norm[sent,word_ind],sent, word_ind)
                                     for ind_sent, sent in enumerate(range(chars_norm.size(0)))
                                     for ind_w, word_ind in enumerate(range(chars_norm.size(1)))]
            else:
                character_norm_display = []
            printing("Feeding source characters {} \n ------ Target characters {}  "
                     "(NB : the character vocabulary is the same at input and output)", var=(character_display, character_norm_display),
                     verbose=_verbose, verbose_level=5)
            printing("Feeding source words {} ", var=[word_display], verbose=_verbose, verbose_level=5)
            printing("Feeding source pos {} ", var=[pos_display], verbose=_verbose, verbose_level=5)
            printing("TYPE {} char before batch chars_norm {} ", var=(char.is_cuda, chars_norm.is_cuda),
                     verbose=verbose, verbose_level=5)

            if NORM2NOISY:
                print("WARNING !! NORM2NOISY ON ")
                yield MaskBatch(chars_norm, char, output_word=word_norm,
                                output_norm_not_norm=word_norm_not_norm,
                                pos=pos, pad=padding, timing=timing, input_word=word, verbose=verbose), order_ids
            else:
                yield MaskBatch(char, chars_norm, output_word=word_norm,
                                output_norm_not_norm=word_norm_not_norm,
                                pos=pos, pad=padding, timing=timing, input_word=word, verbose=verbose), order_ids


def data_gen_dummy(V, batch, nbatches, sent_len=9, word_len=5, verbose=0, seed=None):
    "Generate random data for a src-tgt copy task."
    if seed is not None:
        np.random.seed(seed)
    for i in tqdm(range(nbatches), disable=disable_tqdm_level(verbose, verbose_level=2)):
        data = torch.from_numpy(np.random.randint(low=2, high=V, size=(batch, sent_len, word_len)))
        data[:, :,0] = 2
        # we force padding in the dummy model
        data[:, :, -1] = 1
        data[:, :, -2] = 1
        printing("DATA dummy {} ", var=(data), verbose=verbose, verbose_level=5)
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield MaskBatch(src, tgt, pad=1)


def data_gen(V, batch, nbatches,seq_len=10):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(2, V, size=(batch, seq_len)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield MaskBatch(src, tgt, pad=1)


if __name__=="__main__":
    dummy, conll = False, True
    if dummy:
        iter = data_gen_dummy(V=5, batch=2, nbatches=1)

        for ind, batch in enumerate(iter):
            print("BATCH NUMBER {} ".format(ind))
            print("SRC : ", batch.input_seq)
            print("SRC MASK : ", batch.input_seq_mask)
            print("TARGET : ", batch.output_seq)
            #print("TARGET MASK : ", batch.output_mask)
    elif conll:
        dict_path = "../dictionaries/"
        test_path = "/Users/bemuller/Documents/Work/INRIA/dev/parsing/normpar/data/lexnorm.integrated.demo2"
        verbose = 2
        batch_size = 10
        nbatch = 50
        add_start_char = 1
        add_end_char = 1
        normalization = True
        word_dictionary,word_dictionary_norm , char_dictionary, pos_dictionary,\
        xpos_dictionary, type_dictionary = conllu_data.create_dict(dict_path=dict_path,
                                                                   train_path=test_path,
                                                                   dev_path=test_path,
                                                                   test_path=None,
                                                                   word_embed_dict={},
                                                                   dry_run=False,
                                                                   vocab_trim=True, add_start_char=add_start_char)
        batchIter = data_gen_conllu(test_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                                    type_dictionary,
                                    print_raw=False, normalization=normalization,
                                    verbose=verbose)
        for i, batch in enumerate(batchIter):
            print("Batch {} ".format(i))
