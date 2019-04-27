from io_.info_print import printing
from io_.dat.constants import SPECIAL_TOKEN_LS
from env.importing import *


def write_conll(format, dir_normalized, dir_original, src_text_ls, text_decoded_ls,
                src_text_pos, pred_pos_ls, tasks,
                ind_batch=0, new_file=False, verbose=0):
    assert format in ["conll"]
    assert len(tasks) == 1, "ERROR : only supported so far 1 task at a time"

    if tasks[0] == "normalize":
        src_ls = src_text_ls
        pred_ls = text_decoded_ls
    elif tasks[0] == "pos":
        src_ls = src_text_pos
        pred_ls = pred_pos_ls
    if format == "conll":
        mode_write = "w" if new_file else "a"
        with open(dir_normalized, mode_write) as norm_file:
            with open(dir_original, mode_write) as original:
                print("write_output tasks", tasks, src_ls, pred_ls)
                for ind_sent, (original_sent, normalized_sent) in enumerate(zip(src_ls, pred_ls)):
                    try:
                        assert len(original_sent) == len(normalized_sent), "ERROR : original_sent len {} {} \n  " \
                                                                     "normalized_sent len {} {} " \
                                                                     "".format(len(original_sent), original_sent, len(normalized_sent),normalized_sent)
                    except AssertionError as e:
                        print(e)
                    norm_file.write("#\n")
                    original.write("#\n")
                    norm_file.write("#sent_id = {} \n".format(ind_sent+ind_batch+1))
                    original.write("#sent_id = {} \n".format(ind_sent+ind_batch+1))
                    ind_adjust = 0
                    for ind, (original_token, normalized_token) in enumerate(zip(original_sent,
                                                                                 normalized_sent)):
                        # WE REMOVE SPECIAL TOKENS ONLY IF THEY APPEAR AT THE BEGINING OR AT THE END
                        # on the source token !! (it tells us when we stop) (we nevern want to use gold information)
                        if original_token in SPECIAL_TOKEN_LS  and (ind+1 == len(original_sent) or ind == 0):
                            ind_adjust = 1
                            continue

                        # TODO : when want simultanuous training : assert src_pos src_norm same
                        #   --> assert pred_pos and pred_norm are same lengh (number of words) ans write
                        if tasks[0] == "normalize":
                            norm_file.write("{}\t{}\t_\t_\t_\t_\t{}\t_\t_\tNorm={}|\n".format(ind + 1 - ind_adjust,
                                                                                              original_token,
                                                                                              ind - ind_adjust if ind - ind_adjust > 0 else 0,
                                                                                              normalized_token))
                        if tasks[0] == "pos":
                            norm_file.write("{}\t{}\t_\t{}\t_\t_\t{}\t_\t_\tNorm=()|\n".format(ind + 1 - ind_adjust,
                                                                                               original_token,
                                                                                               normalized_token,
                                                                                               ind-ind_adjust if ind - ind_adjust > 0 else 0
                                                                                               ))
                        original.write("{}\t{}\t_\t_\t_\t_\t_\t_\t{}\t_\n".format(ind+1,
                                                                                  original_token,
                                                                                  ind - ind_adjust if ind - ind_adjust > 0 else 0))
                    norm_file.write("\n")
                    original.write("\n")
            printing("WRITING predicted batch of {} original and {} normalized", var=[dir_original, dir_normalized], verbose=verbose, verbose_level=1)
