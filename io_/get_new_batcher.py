
from io_.info_print import printing
from io_.data_iterator import readers_load, conllu_data, data_gen_multi_task_sampling_batch


def get_new_shard(shard_path, n_shards, random=True, verbose=1):
    # pick a new file randomly
    path = ""

    printing("INFO : picking shard {} ", var=[path], verbose=verbose, verbose_level=1)
    return path


def load_batcher_shard_data(args, word_dictionary, tokenizer, word_norm_dictionary, char_dictionary,
                            pos_dictionary, xpos_dictionary, type_dictionary, use_gpu,
                            norm_not_norm, word_decoder, add_start_char, add_end_char, symbolic_end,
                            symbolic_root, bucket, max_char_len, must_get_norm, bucketing_level,
                            use_gpu_hardcoded_readers, auxilliary_task_norm_not_norm, random_iterator_train,
                            verbose, shard_path):

    path = get_new_shard(shard_path)

    readers = readers_load(datasets=path ,
                           tasks=args.tasks,
                           args=args,
                           word_dictionary=word_dictionary,
                           bert_tokenizer=tokenizer,
                           word_dictionary_norm=word_norm_dictionary, char_dictionary=char_dictionary,
                           pos_dictionary=pos_dictionary, xpos_dictionary=xpos_dictionary,
                           type_dictionary=type_dictionary, use_gpu=use_gpu_hardcoded_readers,
                           norm_not_norm=auxilliary_task_norm_not_norm,
                           word_decoder=True,
                           add_start_char=1, add_end_char=1, symbolic_end=1,
                           symbolic_root=1, bucket=True, max_char_len=20,
                           must_get_norm=True, bucketing_level=bucketing_level,
                           verbose=verbose)

    batchIter = data_gen_multi_task_sampling_batch(tasks=args.tasks, readers=readers,
                                                   batch_size=args.batch_size,
                                                   word_dictionary=word_dictionary,
                                                   char_dictionary=char_dictionary,
                                                   pos_dictionary=pos_dictionary,
                                                   word_dictionary_norm=word_norm_dictionary,
                                                   get_batch_mode=random_iterator_train,
                                                   extend_n_batch=1,
                                                   print_raw=False,
                                                   dropout_input=0.0,
                                                   verbose=verbose)

    return batchIter
