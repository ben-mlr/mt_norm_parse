from env.importing import pdb

from env.project_variables import LIU_TRAIN, LEX_TRAIN_SPLIT, LIU_TRAIN_OWOPUTI, LIU_DEV, LEX_DEV_SPLIT
from io_.data_iterator import readers_load, conllu_data, data_gen_multi_task_sampling_batch
from io_.dat.normalized_writer import write_conll

if __name__ == "__main__":

    tasks = ["normalize"]
    train_path = [LEX_DEV_SPLIT]
    run_mode = "train"
    case = None
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
                              word_normalization=True,
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
                                 norm_not_norm=True, word_decoder=True,
                                 add_start_char=1, add_end_char=1, symbolic_end=1,
                                 symbolic_root=1, bucket=True, max_char_len=20,
                                 must_get_norm=True,
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
    ind = 0
    write = False
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
                write_conll(format="conll", dir_normalized="../data/lex_{}_split_liu_{}_owuputi.conll".format(label, label),
                        dir_original="../data/lex_{}_split_liu_or-{}_owuputi.conll".format(label, label),
                        src_text_ls=batch.raw_input,
                        text_decoded_ls=batch.raw_output, tasks=tasks,
                        src_text_pos=None, pred_pos_ls=None, verbose="raw_data",
                        ind_batch=ind)

        except StopIteration:
            break

    print("SKIPED {} - {}".format(skiped, not_skiped))
