
from io_.dat.normalized_writer import write_conll
from env.project_variables import PROJECT_PATH
from env.importing import os, pdb, unidecode


def line_to_conll(dir_src, dir_target, starting_index=0, cut_sent=False, n_sents=1000000):
    ind = 0
    with open(dir_src, "r") as f:
        max_len = 0
        dir_target = dir_target + ".conll"
        for line in f:
            ind += 1
            if starting_index > 0:
                if ind <= starting_index:
                    continue

            if "  " in unidecode.unidecode(line):
                line = line.strip()
                line = " ".join(line.split())
                assert "  " not in unidecode.unidecode(line)

            line = line.strip().split(" ")


            #for index, word in enumerate(line):
                #row_to_write = "{}\t{}\t_\t_\t_\t_\t_\t_\t_\t_\tNorm={}|\n".format(index, word, word)
            max_len_word = write_conll(format="conll", dir_normalized=dir_target,
                                       dir_original=dir_target+"-src.conll",
                                       src_text_ls=[line],
                                       text_decoded_ls=None, tasks=["normalize"],
                                       src_text_pos=None, pred_pos_ls=None, verbose=1,
                                       new_file=False, cp_paste=True,
                                       permuting_mode=None, cut_sent=cut_sent,
                                       ind_batch=ind)
            #print("MAX INT", max_len_word)
            max_len = max(max_len_word, max_len)
            if ind > (n_sents+starting_index):
                print("break")
                break
    print("WRITITNG DONE {} ".format(dir_target))
    print("MAX_LEN", max_len)


if __name__ == "__main__":
    #line_to_conll(os.path.join(PROJECT_PATH, "data", "pan_ben_conll.tok"), os.path.join(PROJECT_PATH, "data","pan_tweets-dev"), starting_index=1000000, n_sents=30000)

    file_name = "code-test-1k.conll"
    starting_index = 160000
    n_sents = 1000
    line_to_conll(os.path.join(PROJECT_PATH, "data", "code_mixed", "code-mixed_code-mixed1.txt.txt"),
                  os.path.join(PROJECT_PATH, "data", "code_mixed", file_name),
                  cut_sent=True, starting_index=starting_index, n_sents=n_sents)

    with open(os.path.join(PROJECT_PATH, "data", "code_mixed", "README-data.txt"), "a") as f:
        f.write("line_to_conll(os.path.join(PROJECT_PATH, data, code_mixed, code-mixed_code-mixed1.txt.txt), os.path.join(PROJECT_PATH,data, code_mixed, {}),cut_sent=True, starting_index={}, n_sents={}))".format(file_name, starting_index, n_sents))
        f.write("\n")
        print("Loged to ", os.path.join(PROJECT_PATH, "data", "code_mixed", "README-data.txt"))
