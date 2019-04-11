from env.importing import *
from io_.info_print import disable_tqdm_level, printing


def load_emb(extern_emb_dir, verbose=0):
    loaded_dic = {}

    external_embedding_fp = open(extern_emb_dir, 'r', encoding='utf-8', errors='ignore')
    printing("W2V INFO : Starting loading of {} ", var=[extern_emb_dir], verbose_level=1, verbose=verbose)
    for ind, line in enumerate(external_embedding_fp):
    #for ind, line in enumerate(external_embedding_fp):
        line = line.strip().split()
        if len(line)<=5:
            printing("W2V : skipping line {} because tiny", var=ind, verbose=verbose, verbose_level=1)
            continue
        loaded_dic[line[0]] = [float(f) for f in line[1:]]
    dim = len([float(f) for f in line[1:]])
    external_embedding_fp.close()
    printing("W2V INFO  : Word Embedding loaded form {} with  {} words of dim {} ", var=[extern_emb_dir, len(loaded_dic), dim], verbose=verbose, verbose_level=1)
    return loaded_dic