from io_.info_print import printing
from io_.dat.constants import SPECIAL_TOKEN_LS
from env.importing import *

APPLY_PERMUTE_THRESHOLD_SENT = 0.8
APPLY_PERMUTE_WORD = 0.8


def write_conll(format, dir_normalized, dir_original, src_text_ls, text_decoded_ls,
                src_text_pos, pred_pos_ls, tasks, inverse=False,permuting_mode=None,
                ind_batch=0, new_file=False, verbose=0):
    assert format in ["conll"]
    #assert len(tasks) == 1, "ERROR : only supported so far 1 task at a time"

    if tasks[0] == "normalize":
        src_ls = src_text_ls
        pred_ls = text_decoded_ls
        if text_decoded_ls is None:
            assert permuting_mode is not None
            pred_ls = src_text_ls
    elif tasks[0] == "pos":
        src_ls = src_text_pos
        pred_ls = pred_pos_ls
    if format == "conll":
        mode_write = "w" if new_file else "a"
        if new_file:
            print("CREATING NEW FILE", dir_normalized)
        with open(dir_normalized, mode_write) as norm_file:
            with open(dir_original, mode_write) as original:
                for ind_sent, (original_sent, normalized_sent) in enumerate(zip(src_ls, pred_ls)):
                    try:
                        assert len(original_sent) == len(normalized_sent), "ERROR (writer) original_sent len {} {} \n  " \
                                                                     "normalized_sent len {} {} " \
                                                                     "".format(len(original_sent), original_sent, len(normalized_sent),normalized_sent)
                    except AssertionError as e:
                        print(e)
                    norm_file.write("#\n")
                    original.write("#\n")
                    norm_file.write("#sent_id = {} \n".format(ind_sent+ind_batch+1))
                    original.write("#sent_id = {} \n".format(ind_sent+ind_batch+1))
                    ind_adjust = 0

                    if permuting_mode == "sample_mode":
                        noise_level_sentence = np.random.random(1)[0]

                    for ind, (original_token, normalized_token) in enumerate(zip(original_sent,
                                                                                 normalized_sent)):
                        # WE REMOVE SPECIAL TOKENS ONLY IF THEY APPEAR AT THE BEGINING OR AT THE END
                        # on the source token !! (it tells us when we stop) (we nevern want to use gold information)

                        if original_token in SPECIAL_TOKEN_LS and (ind+1 == len(original_sent) or ind == 0):
                            ind_adjust = 1
                            continue

                        if permuting_mode == "sample_mode":
                            # 20% of sentences we apply a 80 noise level n 80% of cases only a 20% noise level
                            rand_word = np.random.random(1)[0]
                            threshold_word = 0.8 if noise_level_sentence < 0.2 else 0.2
                            if rand_word < threshold_word:
                                permuting_mode = np.random.choice(["permute", "double", "random_replace",
                                                                   "multiply_last", "double_last","remove",
                                                                   "remove_last", "z_replace_s"])
                        else:
                            rand_word = None

                        # TODO : when want simultanuous training : assert src_pos src_norm same
                        #   --> assert pred_pos and pred_norm are same lengh (number of words) ans write
                        print("PERMUTATION is ", permuting_mode,  rand_word, APPLY_PERMUTE_WORD, noise_level_sentence)
                        if tasks[0] == "normalize":
                            if inverse:
                                _original_token = normalized_token
                                _normalized_token = original_token
                            else:
                                _original_token = original_token
                                _normalized_token = normalized_token
                                if permuting_mode is not None:
                                    # rule one
                                    print("ORIGINAL TOKEN", original_token)
                                    if ( _original_token == _normalized_token or _original_token.lower() == _normalized_token.lower())\
                                        and not (original_token.startswith("#") or original_token.startswith("@")):
                                        # rule 1
                                        if permuting_mode == "z_replace_s" and len(original_token) > 1:
                                            if original_token.endswith("s"):
                                                _original_token = original_token[:-1] + "z"
                                            else:
                                                permuting_mode = np.random.choice(["permute", "double",
                                                                                   "random_replace",
                                                                                   "remove", "remove_last",
                                                                                   "multiply_last","double_last",
                                                                                    "z_replace_s"])

                                        if permuting_mode == "permute" and len(original_token) > 1:
                                            start_index = 0 if not (original_token.startswith("#") or original_token.startswith("@")) else 1
                                            to_permute = np.random.randint(start_index, len(original_token)-1)
                                            second_letter = original_token[to_permute+1]
                                            first_letter = original_token[to_permute]
                                            list_original_token = list(original_token)
                                            #pdb.set_trace()
                                            list_original_token[to_permute] = second_letter
                                            list_original_token[to_permute+1] = first_letter
                                            _original_token = "".join(list_original_token)
                                        # rule 2
                                        if (permuting_mode == "double" or permuting_mode == "remove") and len(original_token) > 1:
                                            start_index = 0
                                            to_double = np.random.randint(start_index, len(original_token)-1)
                                            first_letter = original_token[to_double]
                                            list_original_token = list(original_token)
                                            #pdb.set_trace()
                                            if permuting_mode == "double":
                                                list_original_token = list_original_token[:to_double] + [first_letter] + list_original_token[to_double:]
                                            else:
                                                list_original_token = list_original_token[:to_double] + list_original_token[to_double:]

                                            _original_token = "".join(list_original_token)

                                        if permuting_mode == "remove_last" and len(original_token) > 1:
                                            _original_token = _original_token[:-1]
                                        if permuting_mode == "double_last" and len(original_token) > 1:
                                            _original_token = _original_token+_original_token[-1]
                                        if permuting_mode == "random_replace" and len(original_token) > 1:
                                            start_index = 0
                                            to_replace = np.random.randint(start_index, len(original_token) - 1)
                                            random_letter = np.random.choice(list("abcdefghijklmnopqrstuvwxyz"))
                                            first_letter = original_token[to_replace]
                                            list_original_token = list(original_token)
                                            # pdb.set_trace()

                                            list_original_token[to_replace] = random_letter

                                            _original_token = "".join(list_original_token)



                                        print("NEW TOKEN", permuting_mode, _original_token)

                                    #pdb.set_trace()

                            norm_file.write("{}\t{}\t_\t_\t_\t_\t{}\t_\t_\tNorm={}|\n".format(ind + 1 - ind_adjust,
                                                                                              _original_token,
                                                                                              ind - ind_adjust if ind - ind_adjust > 0 else 0,
                                                                                              _normalized_token))
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
            printing("WRITING predicted batch of {} original and {} normalized", var=[dir_original, dir_normalized], verbose=verbose, verbose_level="raw_data")
