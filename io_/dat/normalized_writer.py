from io_.info_print import printing


def write_normalization(format, dir_normalized, dir_original, src_text_ls, text_decoded_ls, verbose=0):
    assert format in ["conll"]
    if format == "conll":
        with open(dir_normalized ,"a") as norm_file:
            with open(dir_original, "a") as original:
                for ind_sent, (original_sent, normalized_sent) in enumerate(zip(src_text_ls ,text_decoded_ls)):
                    norm_file.write("#sent_id = {} \n".format(ind_sent +1))
                    original.write("#sent_id = {} \n".format(ind_sent +1))
                    for ind, (original_token, normalized_token) in enumerate(zip(original_sent,
                                                                                 normalized_sent)):
                        norm_file.write("{}\t{}\t_\t_\t_\t_\t_\t_\t_\tNorm={}|\n".format(ind+1,
                                                                                   original_token, normalized_token))
                        original.write("{}\t{}\t_\t_\t_\t_\t_\t_\t_\t_\n".format(ind+1,
                                                                                 original_token))
                    norm_file.write("\n")
                    original.write("\n")
            printing("WRITING predicted batch of {} original and {} normalized", var=[dir_original, dir_normalized],
                     verbose=verbose, verbose_level=1)