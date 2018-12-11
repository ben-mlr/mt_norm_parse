







 add_start_char = 1
    verbose = 5
    nbatches = 10


    word_dictionary, char_dictionary, pos_dictionary, \
    xpos_dictionary, type_dictionary = conllu_data.create_dict(dict_path=dict_path,
                                                               train_path=test_path,
                                                               dev_path=test_path,
                                                               test_path=test_path,
                                                               word_embed_dict={},
                                                               dry_run=False,
                                                               add_start_char=add_start_char,
                                                               vocab_trim=True)

    V = len(char_dictionary.instance2index)+1
    print("Character vocabulary is {} length".format(V))

    model = LexNormalizer(generator=Generator, char_embedding_dim=5, voc_size=V,
                          hidden_size_encoder=11, output_dim=10,
                          hidden_size_decoder=11, verbose=verbose)

    batchIter = data_gen_conllu(test_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary,
                                normalization=True,
                                add_start_char=add_start_char, add_end_char=0,
                                batch_size=2, nbatch=nbatches, print_raw=True, verbose=verbose)

    printing("Starting training", verbose=verbose, verbose_level=0)
    loss = run_epoch(batchIter, model, LossCompute(model.generator, verbose=verbose),
                     n_epochs=1, i_epoch=1, n_batches=None, empty_run=False, verbose=verbose)
    printing("END training loss is {} ".format(loss), verbose=verbose, verbose_level=0)