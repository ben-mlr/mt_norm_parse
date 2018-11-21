from torch.autograd import Variable
import torch
import numpy as np
import pdb
from io_.batch_generator import MaskBatch
import sys
from tqdm import tqdm
sys.path.insert(0,"/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/ELMoLex_sosweet/")
from dat import conllu_data


def data_gen_conllu(data_path, word_dictionary, char_dictionary, pos_dictionary,
                    xpos_dictionary, type_dictionary, batch_size, nbatch):

    data = conllu_data.read_data_to_variable(data_path, word_dictionary, char_dictionary, pos_dictionary,
                                               xpos_dictionary, type_dictionary,
                                               use_gpu=0, symbolic_root=True, dry_run=0, lattice=False)

    for _ in tqdm(range(1, nbatch)):
        _, char, _, _, _, _, _, _, _ = conllu_data.get_batch_variable(data,batch_size=batch_size, unk_replace=0) # word, char, pos, xpos, heads, types, masks, lengths, morph
        for word_ind in range(char.size(1)):
            yield MaskBatch(char[:, word_ind, :], char[:, word_ind, :])


def data_gen_dummy(V, batch, nbatches,seq_len=10):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(2, V, size=(batch, seq_len)))
        data[:, 0] = 1
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
    iter = data_gen_dummy(V=5, batch=2, nbatches=1)
    for ind, batch in enumerate(iter):
        print("BATCH NUMBER {} ".format(ind))
        print("SRC : ", batch.input_seq)
        print("SRC MASK : ", batch.input_seq_mask)
        print("TARGET : ", batch.output_seq)
        print("TARGET MASK : ", batch.output_mask)