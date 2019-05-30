from env.importing import pdb

from env.project_variables import LIU_TRAIN, LEX_TRAIN_SPLIT, LIU_TRAIN_OWOPUTI, LIU_DEV, LEX_DEV_SPLIT, EN_LINES_EWT_TRAIN, TWEETS_GANESH, TWEETS_GANESH_1M
from io_.data_iterator import readers_load, conllu_data, data_gen_multi_task_sampling_batch
from io_.dat.normalized_writer import write_conll

if __name__ == "__main__":

    tasks = ["normalize"]
    train_path = [TWEETS_GANESH]
    run_mode = "train"
    case = None
    word_normalization = False
    random_iterator_train = False
    batch_size = 1
    print(train_path)
    word_dictionary, word_norm_dictionary, char_dictionary, pos_dictionary, \
    xpos_dictionary, type_dictionary = \
        conllu_data.load_dict(dict_path="../dictionaries",
                              train_path=train_path if run_mode == "train" else None,
                              dev_path=train_path if run_mode == "train" else None,
                              test_path=None,
                              word_embed_dict={},
                              dry_run=False,
                              expand_vocab=False,
                              word_normalization=word_normalization,
                              force_new_dic=True if run_mode == "train" else False,
                              tasks=tasks,
                              pos_specific_data_set=train_path[1] if len(tasks) > 1 and "pos" in tasks else None,
                              case=case,
                              add_start_char=1 if run_mode == "train" else None,
                              verbose=1)

    readers_train = readers_load(datasets=train_path, tasks=tasks, word_dictionary=word_dictionary,
                                 word_dictionary_norm=word_norm_dictionary, char_dictionary=char_dictionary,
                                 pos_dictionary=pos_dictionary, xpos_dictionary=xpos_dictionary,
                                 type_dictionary=type_dictionary, use_gpu=False,
                                 norm_not_norm=word_normalization, word_decoder=word_normalization,
                                 add_start_char=1, add_end_char=1, symbolic_end=1,
                                 symbolic_root=1, bucket=False, max_char_len=20,
                                 must_get_norm=word_normalization,
                                 verbose=1)

    batchIter = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers_train, batch_size=batch_size,
                                                   word_dictionary=word_dictionary,
                                                   char_dictionary=char_dictionary,
                                                   pos_dictionary=pos_dictionary,
                                                   word_dictionary_norm=word_norm_dictionary,
                                                   get_batch_mode=random_iterator_train,
                                                   extend_n_batch=1,
                                                   print_raw=False,
                                                   dropout_input=0.0,
                                                   verbose=1)

    skiped = 0

    not_skiped = 0
    label = "dev"
    extra = "norm+permute"
    ind = 0
    write = True
    new_file = False
    file_name = "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/.././data/wnut-2015-ressources/pan_tweets-200k"#train_path[0]
    while True:
        try:
            batch = batchIter.__next__()
            ind += 1
            print(batch.raw_output, batch.raw_input)

            def check_if_space(output):
                for sent in output:
                    for word in sent:
                        if " " in word:
                            return 1
                return 0

            if check_if_space(batch.raw_output):
                skiped += 1
                continue
            else:
                not_skiped += 1
            if write:
                write_conll(format="conll", dir_normalized=file_name+"-{}.conll".format(extra),
                            dir_original=file_name+"-src_token_only-{}.conll".format(extra),
                            src_text_ls=batch.raw_input,
                            text_decoded_ls=None, tasks=tasks,
                            src_text_pos=None, pred_pos_ls=None, verbose=1,
                            new_file=new_file, cp_paste=False,
                            permuting_mode="sample_mode",
                            #"2_following_letters",
                            ind_batch=ind)
            new_file = False
        except StopIteration:
            break

    print("SKIPED {} - {}".format(skiped, not_skiped))
