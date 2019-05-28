
from io_.dat.normalized_writer import write_conll
from env.project_variables import PROJECT_PATH
from env.importing import os, pdb


def line_to_conll(dir_src, dir_target):
    ind = 0
    with open(dir_src, "r") as f:
        for line in f:
            ind += 1
            line = line.strip().split(" ")
            #for index, word in enumerate(line):
                #row_to_write = "{}\t{}\t_\t_\t_\t_\t_\t_\t_\t_\tNorm={}|\n".format(index, word, word)
            write_conll(format="conll", dir_normalized=dir_target+".conll",
                        dir_original=dir_target+"-src.conll",
                        src_text_ls=[line],
                        text_decoded_ls=None, tasks=["normalize"],
                        src_text_pos=None, pred_pos_ls=None, verbose=1,
                        new_file=False, cp_paste=True,
                        permuting_mode=None,
                        ind_batch=ind)
            if ind > 200000:
                break
    print("WRITITNG DONE {} ".format(dir_target))


if __name__ == "__main__":

    line_to_conll(os.path.join(PROJECT_PATH, "data", "pan_ben_conll.tok"), os.path.join(PROJECT_PATH, "data", "pan_tweets-200k"))
