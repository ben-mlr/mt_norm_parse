from env.importing import *
from env.project_variables import *
from io_.info_print import printing
from toolbox.gpu_related import use_gpu_
from toolbox import git_related as gr
import toolbox.deep_learning_toolbox as dptx
from toolbox.directories_handling import setup_repoting_location
from toolbox.optim.freezing_policy import apply_fine_tuning_strategy
from tracking.reporting_google_sheet import append_reporting_sheet, update_status
from io_.data_iterator import readers_load, conllu_data, data_gen_multi_task_sampling_batch
#import model.bert_tools_from_core_code.tokenization as bert_tok
from model.bert_tools_from_core_code.tokenization import BertTokenizer
from training.epoch_run_fine_tuning_bert import epoch_run
from toolbox.report_tools import write_args
from toolbox.pred_tools.heuristics import get_letter_indexes


def run(tasks, train_path, dev_path, n_iter_max_per_epoch, args,
        voc_tokenizer, auxilliary_task_norm_not_norm, model,
        null_token_index, null_str, initialize_bpe_layer=None,
        run_mode="train", test_path_ls=None, dict_path=None, end_predictions=None,
        report=True, model_suffix="", description="",
        saving_every_epoch=10, lr=0.0001, fine_tuning_strategy="standart", model_location=None, model_id=None,
        freeze_parameters=None, freeze_layer_prefix_ls=None,
        # those two have been factorized out in fine_tuning_strategy
        dropout_input_bpe=0,
        report_full_path_shared=None, shared_id=None, bert_model=None, skip_1_t_n=False,
        heuristic_ls=None, gold_error_detection=False, heuristic_test_ls=None,
        portion_mask=None, masking_strategy=None,
        norm_2_noise_eval=False, norm_2_noise_training=None,
        remove_mask_str_prediction=False, inverse_writing=False,
        extra_label_for_prediction="",
        random_iterator_train=True, bucket_test=True, must_get_norm_test=True,
        aggregating_bert_layer_mode=None,
        early_stoppin_metric=None, subsample_early_stoping_metric_val=None,
        compute_intersection_score_test=True,
        slang_dic_test=None, list_reference_heuristic_test=None,
        bert_module=None,
        case=None, threshold_edit=3, tokenize_and_bpe=False,
        layer_wise_attention=None, multi_task_loss_ponderation=None,
        append_n_mask=False,
        debug=False,  batch_size=2, n_epoch=1, verbose=1):
    """
    2 modes : train (will train using train and dev iterators with test at the end on test_path)
              test : only test at the end : requires all directoris to be created
    :return:
    """
    assert run_mode in ["train", "test"], "ERROR run mode {} corrupted ".format(run_mode)
    printing("MODEL : RUNNING IN {} mode", var=[run_mode], verbose=verbose, verbose_level=1)
    printing("WARNING : casing was set to {} (this should be consistent at train and test)", var=[case], verbose=verbose, verbose_level=1)
    use_gpu_hardcoded_readers = False
    printing("WARNING use_gpu_hardcoded_readers hardcoded for readers set to {}", var=[use_gpu_hardcoded_readers],
             verbose=verbose, verbose_level=1)

    if early_stoppin_metric is None:
        if "pos" in tasks:
            early_stoppin_metric = "accuracy-exact-pos"
            subsample_early_stoping_metric_val = "all"
        elif "normalize" in tasks:
            early_stoppin_metric = "f1-normalize"
            subsample_early_stoping_metric_val = "rates"
        printing("INFO : setting early_stoppin_metric to {}", var=[early_stoppin_metric], verbose=verbose, verbose_level=1)
    else:
        if early_stoppin_metric == "accuracy-exact-normalize":
            subsample_early_stoping_metric_val = "all"
        printing("INFO : early_stoppin_metric passed is {}", var=[early_stoppin_metric], verbose=verbose, verbose_level=1)
    assert len(tasks) == len(train_path), "ERROR tasks is {} but train path are {}".format(tasks, train_path)
    assert len(dev_path) == len(train_path)
    #assert len(test_path_ls) == len(tasks), "{} tasks test_path_ls {}".format(test_path_ls, tasks)
    if run_mode == "test":
        assert test_path_ls is not None and isinstance(test_path_ls, list)
    if test_path_ls is not None:
        assert isinstance(test_path_ls, list) and isinstance(test_path_ls[0], list), \
            "ERROR test_path_ls should be a list"

    if run_mode == "train":
        printing("CHECKPOINTING info : saving model every {}", var=saving_every_epoch, verbose=verbose, verbose_level=1)
    use_gpu = use_gpu_(use_gpu=None, verbose=verbose)
    train_data_label = "|".join([REPO_DATASET[_train_path] for _train_path in train_path])
    dev_data_label = "|".join([REPO_DATASET[_dev_path] for _dev_path in dev_path]) if dev_path is not None else None
    if use_gpu:
        model.to("cuda")

    if not debug:
        pdb.set_trace = lambda: None

    iter_train = 0
    iter_dev = 0
    row = None
    writer = None
    if run_mode == "train":
        assert model_location is None and model_id is None, "ERROR we are creating a new one "
        model_id, model_location, dict_path, tensorboard_log, end_predictions = \
            setup_repoting_location(model_suffix=model_suffix, root_dir_checkpoints=CHECKPOINT_BERT_DIR, shared_id=shared_id,
                                    verbose=verbose)

        hyperparameters = OrderedDict([("bert_model", bert_model), ("lr", lr),
                                       ("n_epochs", n_epoch),
                                       ("initialize_bpe_layer", initialize_bpe_layer),
                                       ("fine_tuning_strategy", fine_tuning_strategy),
                                       ("dropout_input_bpe", dropout_input_bpe),
                                       ("heuristic_ls", heuristic_ls),
                                       ("gold_error_detection", gold_error_detection),
                                       ("dropout_classifier", args.dropout_classifier if args is not None else "UNK"),
                                       ("dropout_bert", args.dropout_bert if args is not None else "UNK"),
                                       ("tasks", tasks),
                                       ("masking_strategy", masking_strategy), ("portion_mask", portion_mask),
                                       ("checkpoint_dir", args.checkpoint_dir if args is not None else None),
                                       ("norm_2_noise_training", norm_2_noise_training),
                                       ("random_iterator_train", random_iterator_train),
                                       ("aggregating_bert_layer_mode", aggregating_bert_layer_mode),
                                       ("tokenize_and_bpe", tokenize_and_bpe),
                                       ("SEED", SEED_TORCH), ("case", case), ("bert_module", bert_module),
                                       ("freeze_layer_prefix_ls", freeze_layer_prefix_ls),
                                       ("layer_wise_attention", layer_wise_attention),
                                       ("append_n_mask", append_n_mask),
                                       ("multi_task_loss_ponderation", multi_task_loss_ponderation)
                                       ])
        printing("HYPERPARAMETERS {} ", var=[hyperparameters], verbose=verbose, verbose_level=1)
        args_dir = write_args(model_location, model_id=model_id, hyperparameters=hyperparameters, verbose=verbose)
        if report:
            if report_full_path_shared is not None:
                tensorboard_log = os.path.join(report_full_path_shared, "tensorboard")
            printing("tensorboard --logdir={} --host=localhost --port=1234 ", var=[tensorboard_log], verbose_level=1,verbose=verbose)
            writer = SummaryWriter(log_dir=tensorboard_log)
            if writer is not None:
                writer.add_text("INFO-ARGUMENT-MODEL-{}".format(model_id), str(hyperparameters), 0)
    else:
        assert dict_path is not None
        assert end_predictions is not None
        assert model_location is not None and model_id is not None
        args_dir = os.path.join(model_location, "{}-args.json".format(model_id))

        printing("CHECKPOINTING : starting writing log \ntensorboard --logdir={} --host=localhost --port=1234 ",
                 var=[os.path.join(model_id, "tensorboard")], verbose_level=1,
                 verbose=verbose)

    # build or make dictionaries
    _dev_path = dev_path if dev_path is not None else train_path
    word_dictionary, word_norm_dictionary, char_dictionary, pos_dictionary, \
    xpos_dictionary, type_dictionary = \
        conllu_data.load_dict(dict_path=dict_path,
                              train_path=train_path if run_mode == "train" else None,
                              dev_path=_dev_path if run_mode == "train" else None,
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

    inv_word_dic = word_dictionary.instance2index
    # load , mask, bucket and index data
    tokenizer = BertTokenizer.from_pretrained(voc_tokenizer)
    assert tokenizer is not None, "ERROR : tokenizer is None , voc_tokenizer failed to be loaded {}".format(voc_tokenizer)
    if run_mode == "train":
        readers_train = readers_load(datasets=train_path, tasks=tasks, word_dictionary=word_dictionary,
                                     word_dictionary_norm=word_norm_dictionary, char_dictionary=char_dictionary,
                                     pos_dictionary=pos_dictionary, xpos_dictionary=xpos_dictionary,
                                     type_dictionary=type_dictionary, use_gpu=use_gpu_hardcoded_readers ,
                                     norm_not_norm=auxilliary_task_norm_not_norm, word_decoder=True,
                                     add_start_char=1, add_end_char=1, symbolic_end=1,
                                     symbolic_root=1, bucket=True, max_char_len=20,
                                     must_get_norm=True,
                                     verbose=verbose)

        readers_dev = readers_load(datasets=dev_path, tasks=tasks, word_dictionary=word_dictionary,
                                   word_dictionary_norm=word_norm_dictionary, char_dictionary=char_dictionary,
                                   pos_dictionary=pos_dictionary, xpos_dictionary=xpos_dictionary,
                                   type_dictionary=type_dictionary, use_gpu=use_gpu_hardcoded_readers,
                                   norm_not_norm=auxilliary_task_norm_not_norm, word_decoder=True,
                                   add_start_char=1, add_end_char=1,
                                   symbolic_end=1, symbolic_root=1, bucket=use_gpu_hardcoded_readers, max_char_len=20,
                                   must_get_norm=True,
                                   verbose=verbose) if dev_path is not None else None
        # Load tokenizer
        early_stoping_val_former = 1000
        try:
            for epoch in range(n_epoch):

                checkpointing_model_data = (epoch % saving_every_epoch == 0 or epoch == (n_epoch - 1))
                # build iterator on the loaded data
                batchIter_train = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers_train, batch_size=batch_size,
                                                                     word_dictionary=word_dictionary,
                                                                     char_dictionary=char_dictionary,
                                                                     pos_dictionary=pos_dictionary,
                                                                     word_dictionary_norm=word_norm_dictionary,
                                                                     get_batch_mode=random_iterator_train,
                                                                     extend_n_batch=1,
                                                                     print_raw=False,
                                                                     dropout_input=0.0,
                                                                     verbose=verbose)
                # -|-|-
                batchIter_dev = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers_dev, batch_size=batch_size,
                                                                   word_dictionary=word_dictionary,
                                                                   char_dictionary=char_dictionary,
                                                                   pos_dictionary=pos_dictionary,
                                                                   word_dictionary_norm=word_norm_dictionary,
                                                                   get_batch_mode=False,
                                                                   extend_n_batch=1,print_raw=False,
                                                                   dropout_input=0.0,
                                                                   verbose=verbose) if dev_path is not None else None
                # TODO add optimizer (if not : dev loss)
                model.train()

                model, optimizer = apply_fine_tuning_strategy(model=model,
                                                                             fine_tuning_strategy=fine_tuning_strategy,
                                                                             lr_init=lr, betas=(0.9, 0.99),
                                                                             epoch=epoch, verbose=verbose)
                print("RUNNING TRAIN on GET_BATCH_MODE ")
                loss_train, iter_train, perf_report_train, _ = epoch_run(batchIter_train, tokenizer,
                                                                         pos_dictionary=pos_dictionary,
                                                                         n_epoch=n_epoch,
                                                                         data_label=train_data_label,
                                                                         model=model,
                                                                         writer=writer,
                                                                         iter=iter_train, epoch=epoch,
                                                                         tasks=tasks,
                                                                         writing_pred=epoch == (n_epoch - 1),
                                                                         dir_end_pred=end_predictions,
                                                                         optimizer=optimizer, use_gpu=use_gpu,
                                                                         predict_mode=True,
                                                                         skip_1_t_n=skip_1_t_n,
                                                                         model_id=model_id,
                                                                         heuristic_ls=heuristic_ls, gold_error_detection=gold_error_detection,
                                                                         reference_word_dic={"InV": inv_word_dic},
                                                                         dropout_input_bpe=dropout_input_bpe,
                                                                         null_token_index=null_token_index, null_str=null_str,
                                                                         masking_strategy=masking_strategy, portion_mask=portion_mask,
                                                                         norm_2_noise_training=norm_2_noise_training,
                                                                         norm_2_noise_eval=False,
                                                                         aggregating_bert_layer_mode=aggregating_bert_layer_mode,
                                                                         early_stoppin_metric=None,
                                                                         case=case, tokenize_and_bpe=tokenize_and_bpe,
                                                                         n_iter_max=n_iter_max_per_epoch,
                                                                         append_n_mask=append_n_mask,
                                                                         multi_task_loss_ponderation=multi_task_loss_ponderation,
                                                                         verbose=verbose)


                model.eval()

                if dev_path is not None:
                    print("RUNNING DEV on ITERATION MODE")
                    loss_dev, iter_dev, perf_report_dev, early_stoping_val = epoch_run(batchIter_dev, tokenizer,
                                                                                       pos_dictionary=pos_dictionary,
                                                                                       iter=iter_dev, use_gpu=use_gpu,
                                                                                       model=model,
                                                                                       writer=writer,
                                                                                       writing_pred=epoch == (n_epoch - 1),
                                                                                       dir_end_pred=end_predictions,
                                                                                       predict_mode=True, data_label=dev_data_label,
                                                                                       epoch=epoch, tasks=tasks,
                                                                                       null_token_index=null_token_index, null_str=null_str,
                                                                                       model_id=model_id,
                                                                                       skip_1_t_n=skip_1_t_n,
                                                                                       dropout_input_bpe=0,
                                                                                       masking_strategy=masking_strategy,
                                                                                       portion_mask=portion_mask,
                                                                                       heuristic_ls=heuristic_ls,
                                                                                       gold_error_detection=gold_error_detection,
                                                                                       reference_word_dic={"InV": inv_word_dic},
                                                                                       norm_2_noise_training=norm_2_noise_training,# as training otherwise loss dev not more meaning
                                                                                       norm_2_noise_eval=False,
                                                                                       early_stoppin_metric=early_stoppin_metric,
                                                                                       tokenize_and_bpe=tokenize_and_bpe,
                                                                                       subsample_early_stoping_metric_val=subsample_early_stoping_metric_val,
                                                                                       aggregating_bert_layer_mode=aggregating_bert_layer_mode,
                                                                                       case=case,
                                                                                       append_n_mask=append_n_mask,
                                                                                       multi_task_loss_ponderation=multi_task_loss_ponderation,
                                                                                       n_iter_max=n_iter_max_per_epoch, verbose=verbose)
                else:
                    loss_dev, iter_dev, perf_report_dev = None, 0, None

                printing("PERFORMANCE {} TRAIN", var=[epoch, perf_report_train],
                         verbose=verbose, verbose_level=1)
                printing("PERFORMANCE {} DEV", var=[epoch, perf_report_dev], verbose=verbose, verbose_level=1)

                printing("TRAINING : loss train:{} dev:{} for epoch {}  out of {}", var=[loss_train, loss_dev, epoch, n_epoch], verbose=1, verbose_level=1)

                if checkpointing_model_data or early_stoping_val < early_stoping_val_former:
                    if early_stoping_val is not None:
                        _epoch = "best" if early_stoping_val < early_stoping_val_former else epoch
                    else:
                        print('WARNING early_stoping_val is None so saving based on checkpointing_model_data only')
                        _epoch = epoch
                    checkpoint_dir = os.path.join(model_location, "{}-ep{}-checkpoint.pt".format(model_id, _epoch))
                    if _epoch == "best":
                        print("SAVING BEST MODEL {} (epoch:{}) (new loss is {} former was {})".format(checkpoint_dir, epoch, early_stoping_val, early_stoping_val_former))
                        last_checkpoint_dir_best = checkpoint_dir
                        early_stoping_val_former = early_stoping_val
                        best_epoch = epoch
                        best_loss = early_stoping_val
                    else:
                        print("NOT SAVING BEST MODEL : new loss {} did not beat first loss {}".format(early_stoping_val , early_stoping_val_former))
                    last_model = ""
                    if epoch == (n_epoch - 1):
                        last_model = "last"
                    printing("CHECKPOINT : saving {} model {} ", var=[last_model, checkpoint_dir], verbose=verbose,
                             verbose_level=1)
                    torch.save(model.state_dict(), checkpoint_dir)
                    args_dir = write_args(dir=model_location, checkpoint_dir=checkpoint_dir,
                                          model_id=model_id,
                                          info_checkpoint=OrderedDict([("n_epochs", epoch+1), ("batch_size", batch_size),
                                                                       ("train_path", train_data_label),
                                                                       ("dev_path", dev_data_label)]),
                                          verbose=verbose)

            print("PERFORMANCE LAST {} TRAIN".format(epoch), perf_report_train)
            print("PERFORMANCE LAST {} DEV".format(epoch), perf_report_dev)

            if row is not None:
                update_status(row=row, new_status="training-done", verbose=1)

        except Exception as e:
            if row is not None:
                update_status(row=row, new_status="ERROR", verbose=1)
            raise(e)
    if run_mode in ["train", "test"] and test_path_ls is not None:
        report_all = []
        if run_mode == "train":
            if use_gpu:
                model.load_state_dict(torch.load(last_checkpoint_dir_best))
                model = model.cuda()
            else:
                model.load_state_dict(torch.load(last_checkpoint_dir_best, map_location=lambda storage, loc: storage))
            printing("MODEL : RELOADING best model of epoch {} with loss {} based on {}({}) metric (from checkpoint {})",
                     var=[best_epoch, best_loss, early_stoppin_metric,
                          subsample_early_stoping_metric_val,
                          last_checkpoint_dir_best],
                     verbose=verbose, verbose_level=1)

        model.eval()
        list_reference_heuristic_test = list_reference_heuristic_test + word_norm_dictionary.instances
        list_reference_heuristic_test.sort()
        alphabet_index = get_letter_indexes(list_reference_heuristic_test)
        list_candidates = None
        mode_need_norm_heuristic = None
        for test_path in test_path_ls:
            assert len(test_path) == len(tasks), "ERROR test_path {} tasks {}".format(test_path, tasks)
            for test, task_to_eval in zip(test_path, tasks):
                #label_data = "|".join([REPO_DATASET[_test_path] for _test_path in test_path])
                label_data = REPO_DATASET[test]+"-"+task_to_eval
                if len(extra_label_for_prediction) > 0:
                    label_data += "-" + extra_label_for_prediction
                readers_test = readers_load(datasets=[test],
                                            tasks=[task_to_eval],
                                            word_dictionary=word_dictionary,
                                            word_dictionary_norm=word_norm_dictionary,
                                            char_dictionary=char_dictionary,
                                            pos_dictionary=pos_dictionary,
                                            xpos_dictionary=xpos_dictionary,
                                            type_dictionary=type_dictionary,
                                            use_gpu=use_gpu_hardcoded_readers ,
                                            norm_not_norm=auxilliary_task_norm_not_norm,
                                            word_decoder=True,
                                            add_start_char=1, add_end_char=1, symbolic_end=1,
                                            symbolic_root=1, bucket=bucket_test,
                                            max_char_len=20,
                                            must_get_norm=must_get_norm_test,
                                            verbose=verbose)

                heuritics_zip = [None] if task_to_eval == "pos" else [None, ["@", "#"], ["@", "#"], None, None]
                gold_error_or_not_zip = [False] if task_to_eval == "pos" else [False, False, True, True, True, False]
                norm2noise_zip = [False] if task_to_eval == "pos" else [False, False, False, False, True]

                if label_data.startswith("lex_norm2015"):
                    best_heurisitc = ["@", "#", "url", "slang_translate"]
                elif label_data.startswith("lexnorm-normalize"):
                    best_heurisitc = ["edit_check-all-need_normed", "@", "#", "url"]
                else:
                    best_heurisitc = ["@", "#", "url"]
                heuritics_zip = [None] if task_to_eval == "pos" else [None, None, #["edit_check-all-all"],
                                                                      #["edit_check-all-need_normed"],
                                                                      #["edit_check-data-need_normed"],
                                                                      #["edit_check-ref-need_normed"],
                                                                      ["@", "#", "url"],
                                                                      #best_heurisitc,
                                                                      best_heurisitc,
                                                                      ["slang_translate"]]
                gold_error_or_not_zip = [False] if task_to_eval == "pos" else [False, True, False, False, False]
                norm2noise_zip = [False] if task_to_eval == "pos" else [False, False, False, False, False]
                if heuristic_test_ls is not None:
                    assert isinstance(heuristic_test_ls, list)
                    if heuristic_test_ls[0] is not None:
                        assert isinstance(heuristic_test_ls,list)
                    heuritics_zip = heuristic_test_ls
                    printing("PREDICTION : setting heuristics to heuristic_test_ls {}",
                             var=[heuristic_test_ls], verbose_level=1, verbose=verbose)

                if heuristic_test_ls is None:
                    assert len(gold_error_or_not_zip) == len(heuritics_zip) and len(heuritics_zip) == len(norm2noise_zip)
                else:
                    if not (len(gold_error_or_not_zip) == len(heuritics_zip) and len(heuritics_zip) == len(norm2noise_zip)):
                        heuritics_zip = heuritics_zip[:len(heuristic_test_ls)]
                        gold_error_or_not_zip = gold_error_or_not_zip[:len(heuristic_test_ls)]
                        printing("WARNING : SHORTENING gold_error zip and norm2noise "
                                 "for prediction purpose because heuristic_test_ls was set",
                                 verbose_level=1, verbose=verbose)
                if inverse_writing:
                    print("WARNING : prediction : only straight pred ")
                    heuritics_zip = [None]
                    gold_error_or_not_zip = [False]
                    norm2noise_zip = [False]

                batch_size_TEST = 1

                print("WARNING : batch_size for final eval was hardcoded and set to {}".format(batch_size_TEST))
                for (heuristic_test, gold_error, norm_2_noise_eval) in zip(heuritics_zip, gold_error_or_not_zip, norm2noise_zip):

                    if heuristic_test is not None and heuristic_test[0].startswith("edit_check"):
                        #assert len(heuristic_test) == 1,\
                        #    "ERROR if heuristic_test is edit_check {} category then it should be the only heuristic".format(heuristic_test)
                        pattern = "(.*)-(.*)-(.*)"
                        match = re.match(pattern, heuristic_test[0])
                        assert match is not None, "ERROR did not find pattern {} in {}".format(pattern, heuristic_test[0])
                        mode_heuristic = match.group(2)
                        mode_need_norm_heuristic = match.group(3)
                        assert mode_heuristic in ["all", "ref", "data"]
                        assert list_reference_heuristic_test is not None
                        if mode_heuristic == "all":
                            list_candidates = list_reference_heuristic_test + word_norm_dictionary.instances
                        elif mode_heuristic == "data":
                            list_candidates = word_norm_dictionary.instances
                        elif mode_heuristic == "ref":
                            list_candidates = list_reference_heuristic_test
                        else:
                            raise(Exception("mode_heuristics {} not supported".format(mode_heuristic)))
                        assert mode_need_norm_heuristic in ["need_normed", "all"], "ERROR mode_need_norm_heuristic " \
                                                                                   "should be in " \
                                                                                   "[all, need_normed] " \
                                                                                   "but is {}".format(mode_need_norm_heuristic)

                        info_details_heursitics = "means we concatanete train+dev data dictionary with ref"
                        printing("HEURISTICS : {} words used as reference (ref+data) {} as candidates ('{}' mode) : ",
                                 var=[len(list_reference_heuristic_test),
                                      len(list_candidates),
                                      mode_heuristic, info_details_heursitics],
                                 verbose=verbose, verbose_level=1)
                        printing("HEURISTICS : edit threshold set to {}", var=[threshold_edit], verbose=verbose,
                                 verbose_level=1)

                    batchIter_test = data_gen_multi_task_sampling_batch(tasks=[task_to_eval], readers=readers_test,
                                                                        batch_size=batch_size_TEST,
                                                                        word_dictionary=word_dictionary,
                                                                        char_dictionary=char_dictionary,
                                                                        pos_dictionary=pos_dictionary,
                                                                        word_dictionary_norm=word_norm_dictionary,
                                                                        get_batch_mode=False,
                                                                        extend_n_batch=1,
                                                                        dropout_input=0.0,
                                                                        verbose=verbose)
                    try:

                        loss_test, iter_test, perf_report_test, _ = epoch_run(batchIter_test, tokenizer,
                                                                              pos_dictionary=pos_dictionary,
                                                                              iter=iter_dev, use_gpu=use_gpu,
                                                                              model=model,
                                                                              writer=None,
                                                                              writing_pred=True,
                                                                              optimizer=None, tasks=[task_to_eval],
                                                                              args_dir=args_dir, model_id=model_id,
                                                                              dir_end_pred=end_predictions,
                                                                              skip_1_t_n=skip_1_t_n,
                                                                              predict_mode=True, data_label=label_data,
                                                                              epoch="LAST", extra_label_for_prediction=label_data,
                                                                              null_token_index=null_token_index,
                                                                              null_str=null_str,
                                                                              log_perf=False,
                                                                              dropout_input_bpe=0,
                                                                              masking_strategy=masking_strategy,
                                                                              portion_mask=portion_mask,
                                                                              heuristic_ls=heuristic_test,
                                                                              gold_error_detection=gold_error,
                                                                              slang_dic=slang_dic_test,
                                                                              list_reference_heuristic=list_reference_heuristic_test,
                                                                              list_candidates=list_candidates,
                                                                              index_alphabetical_order=alphabet_index,
                                                                              norm_2_noise_training=None,
                                                                              # we decide wether we eval everything in mode
                                                                              # norm2noise or not
                                                                              # --> we could also add a loop and tag in report
                                                                              norm_2_noise_eval=norm_2_noise_eval,
                                                                              compute_intersection_score=compute_intersection_score_test,
                                                                              remove_mask_str_prediction=remove_mask_str_prediction,
                                                                              inverse_writing=inverse_writing,
                                                                              aggregating_bert_layer_mode=aggregating_bert_layer_mode,
                                                                              reference_word_dic={"InV": inv_word_dic},
                                                                              case=case, threshold_edit=threshold_edit,
                                                                              tokenize_and_bpe=tokenize_and_bpe,
                                                                              append_n_mask=append_n_mask,
                                                                              multi_task_loss_ponderation=multi_task_loss_ponderation,
                                                                              edit_module_pred_need_norm_only=mode_need_norm_heuristic == "need_normed",
                                                                              n_iter_max=n_iter_max_per_epoch, verbose=verbose)
                        print("LOSS TEST", loss_test)
                    except Exception as e:
                        #raise(e)
                        print("ERROR {} test_path {} , heuristic {} , gold error {} , norm2noise {} ".format(e,
                                                                                                             test,
                                                                                                             heuristic_test,
                                                                                                             gold_error,
                                                                                                             norm_2_noise_eval
                                                                                                             ))
                        perf_report_test = []
                    print("PERFORMANCE TEST on data  {} is {} ".format(label_data, perf_report_test))
                    print("DATA WRITTEN {}".format(end_predictions))
                    if writer is not None:
                        writer.add_text("Accuracy-{}-{}-{}".format(model_id, label_data, run_mode),
                                        "After {} epochs with {} : performance is \n {} ".format(n_epoch, description,
                                                                                                 str(perf_report_test)), 0)
                    else:
                        printing("WARNING : could not add accuracy to tensorboard cause writer was found None", verbose=verbose,
                                 verbose_level=1)
                    report_all.extend(perf_report_test)
    else:
        printing("ERROR : EVALUATION none cause {} empty or run_mode {} ",
                 var=[test_path_ls, run_mode], verbose_level=1, verbose=verbose)

    if writer is not None:
        writer.close()
        printing("tensorboard --logdir={} --host=localhost --port=1234 ", var=[tensorboard_log], verbose_level=1,verbose=verbose)
    if row is not None:
        update_status(row=row, new_status="done", verbose=1)
        update_status(row=row, new_status=tensorboard_log, verbose=1, col_number=10)

    report_dir = os.path.join(model_location, model_id+"-report.json")
    if report_full_path_shared is not None:
        report_full_dir = os.path.join(report_full_path_shared, shared_id + "-report.json")
        if os.path.isfile(report_full_dir):
            report = json.load(open(report_full_dir, "r"))
        else:
            report = []
            printing("REPORT = creating overall report at {} ", var=[report_dir], verbose=verbose, verbose_level=1)
        report.extend(report_all)
        json.dump(report, open(report_full_dir, "w"))
        printing("{} {} ", var=[REPORT_FLAG_DIR_STR, report_full_dir], verbose=verbose, verbose_level=0)

    json.dump(report_all, open(report_dir, "w"))

    if report_full_path_shared is None:
        printing("{} {} ", var=[REPORT_FLAG_DIR_STR, report_dir], verbose=verbose, verbose_level=0)

    return model
