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


def data_gen_conllu(data_path, word_dictionary, char_dictionary, pos_dictionary,
                    xpos_dictionary, type_dictionary, batch_size, nbatch,
                    add_start_char=1,
                    add_end_char=1, padding=1, print_raw=False, verbose=0):

    data = conllu_data.read_data_to_variable(data_path, word_dictionary, char_dictionary, pos_dictionary,
                                             xpos_dictionary, type_dictionary,
                                             use_gpu=0, symbolic_root=False, dry_run=0, lattice=False,verbose=verbose,
                                             add_start_char=add_start_char,add_end_char=add_end_char)
    for ibatch in tqdm(range(1, nbatch+1), disable=disable_tqdm_level(verbose, verbose_level=2)):
        # word, char, pos, xpos, heads, types, masks, lengths, morph
        printing("Data : getting {} out of {} batches".format(ibatch, nbatch+1), verbose, verbose_level=2)
        _, char, _, _, _, _, _, lenght, _ = conllu_data.get_batch_variable(data, batch_size=batch_size, unk_replace=0)
        assert min(lenght.data) > 0, "ERROR : min(lenght.data) is {} ".format(min(lenght.data))
        #print_char_seq(active=print_raw, char_array=char, sent_len=min(lenght.data), word_len=char.size(2),
        #               nbatch=batch_size, dic=char_dictionary)
        for word_ind in range(min(lenght.data)):
            # we don't pass empty words ! to the word level seq2seq
            ## TODO : replace that !!
            #char[:, word_ind, -2] = 1
            #char[:, word_ind, -1] = 1
            #try:
            sent_len = min(lenght.data)
            word_len = char.size(2)
            printing("Char {} word ind : word : {}  ".format(word_ind, char[:, word_ind, :]), verbose=verbose,
                     verbose_level=5)
            to_print = [" ".join([char_dictionary.get_instance(char[batch, word_ind, char_i]) for char_i in range(word_len)]) + " / " for batch in range(char.size(0))]

            _verbose = 5 if print_raw else verbose
            printing("Feeding {} ".format(to_print), verbose=_verbose, verbose_level=5)
            print()
            yield MaskBatch(char[:, word_ind, :], char[:, word_ind, :], pad=padding, verbose=verbose)
            #except Exception as e:
            #    print("ERROR : {} happened on char {} ".format(e, char))
            #    #pdb.set_trace()
            #    continue



def data_gen_dummy(V, batch, nbatches,seq_len=10,
                   verbose=0, seed=None):
    "Generate random data for a src-tgt copy task."
    if seed is not None:
        np.random.seed(seed)
    for i in tqdm(range(nbatches), disable=disable_tqdm_level(verbose, verbose_level=2)):
        data = torch.from_numpy(np.random.randint(low=2,high=V, size=(batch, seq_len)))
        data[:, 0] = 2
        # we force padding in the dummy model
        data[:, -1] = 1
        data[:, -2] = 1
        printing("DATA dummy {} ".format(data), verbose=verbose, verbose_level=5)
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
    dummy , conll = False,True
    if dummy:
        iter = data_gen_dummy(V=5, batch=2, nbatches=1)

        for ind, batch in enumerate(iter):
            print("BATCH NUMBER {} ".format(ind))
            print("SRC : ", batch.input_seq)
            print("SRC MASK : ", batch.input_seq_mask)
            print("TARGET : ", batch.output_seq)
            print("TARGET MASK : ", batch.output_mask)
    elif conll:
        dict_path = "../dictionaries/"

        test_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/lexnorm.integrated.demo2"

        verbose = 5
        batch_size = 2
        nbatch = 1
        add_start_char = 1
        word_dictionary, char_dictionary, pos_dictionary,\
        xpos_dictionary, type_dictionary = conllu_data.create_dict(dict_path=dict_path,
                                                                   train_path=test_path,
                                                                   dev_path=test_path,
                                                                   test_path=None,
                                                                   word_embed_dict={},
                                                                   dry_run=False,
                                                                   vocab_trim=True, add_start_char=add_start_char)
        batchIter = data_gen_conllu(test_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                                    type_dictionary, add_start_char=add_start_char, batch_size=batch_size,print_raw=True,
                                    nbatch=nbatch, verbose=verbose)
        for i, batch in enumerate(batchIter):
            print(batch)



#TODO : unknown replace ?


# # RECALL :
# # the data_gen_conllu originally generates batch of sentences :
            #  the batch_size provided corresponds of batch of sentences
# # if the batch_size of sentences if greater than the number of sentences available in the data
            #  --> it will fall on the length og the current bucket : for instance :
            #  2 if only two sentences in the bucket (5 word in the sentnce)