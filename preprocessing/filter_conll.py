

from env.project_variables import LIU_TRAIN
from io_.data_iterator import readers_load, conllu_data, data_gen_multi_task_sampling_batch

if __name__=="__main__":
    tasks = ["normalize"]
    train_path = LIU_TRAIN
    run_mode = "train"
    case = "lower"
    random_iterator_train=False
    batch_size = 1


    word_dictionary, word_norm_dictionary, char_dictionary, pos_dictionary, \
    xpos_dictionary, type_dictionary = \
        conllu_data.load_dict(dict_path="./dictionaries",
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

    batchIter_train = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers_train, batch_size=batch_size,
                                                         word_dictionary=word_dictionary,
                                                         char_dictionary=char_dictionary,
                                                         pos_dictionary=pos_dictionary,
                                                         word_dictionary_norm=word_norm_dictionary,
                                                         get_batch_mode=random_iterator_train,
                                                         extend_n_batch=1,
                                                         print_raw=False,
                                                         dropout_input=0.0,
                                                         verbose=1)
