from env.importing import *
from env.project_variables import *
from io_.info_print import printing
from toolbox.gpu_related import use_gpu_
from toolbox import git_related as gr
import toolbox.deep_learning_toolbox as dptx
from toolbox.directories_handling import setup_repoting_location
from tracking.reporting_google_sheet import append_reporting_sheet, update_status
from io_.data_iterator import readers_load, conllu_data, data_gen_multi_task_sampling_batch
from model.bert_tools_from_core_code.tokenization import BertTokenizer
from training.epoch_run_fine_tuning_bert import epoch_run

from toolbox.report_tools import write_args


def run(tasks, train_path, dev_path, n_iter_max_per_epoch,args,
        voc_tokenizer, auxilliary_task_norm_not_norm, bert_with_classifier,
        null_token_index, null_str, initialize_bpe_layer=None,
        run_mode="train", test_path_ls = None, dict_path=None, end_predictions=None,
        report=True, model_suffix="", description="",
        saving_every_epoch=10, lr=0.0001, fine_tuning_strategy="standart", model_location=None, model_id=None,
        freeze_parameters=None, freeze_layer_prefix_ls=None,
        report_full_path_shared=None, shared_id=None, bert_model=None,
        debug=False,  batch_size=2, n_epoch=1, verbose=1):
    """
    2 modes : train (will train using train and dev iterators with test at the end on test_path)
              test : only test at the end : requires all directoris to be created
    :return:
    """
    assert run_mode in ["train", "test"], "ERROR run mode {} corrupted ".format(run_mode)
    printing("MODEL : RUNNING IN {} mode", var=[run_mode], verbose=verbose, verbose_level=1)
    if run_mode == "test":
        assert test_path_ls is not None and isinstance(test_path_ls, list)
    if test_path_ls is not None:
        assert isinstance(test_path_ls, list) and isinstance(test_path_ls[0], list), "ERROR test_path_ls should be a list"
    if run_mode == "train":
        printing("CHECKPOINTING info : saving model every {}", var=saving_every_epoch, verbose=verbose, verbose_level=1)
    use_gpu = use_gpu_(use_gpu=None, verbose=verbose)
    train_data_label = "|".join([REPO_DATASET[_train_path] for _train_path in train_path])
    dev_data_label = "|".join([REPO_DATASET[_dev_path] for _dev_path in dev_path]) if dev_path is not None else None
    if use_gpu:
        bert_with_classifier.to("cuda")

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
                                       ("initialize_bpe_layer", initialize_bpe_layer),
                                       ("freeze_parameters", freeze_parameters),
                                       ("freeze_layer_prefix_ls", freeze_layer_prefix_ls),
                                       ("dropout_classifier", args.dropout_classifier)])
        args_dir = write_args(model_location, model_id=model_id, hyperparameters=hyperparameters, verbose=verbose)
        if report:
            if report_full_path_shared is not None:
                tensorboard_log = os.path.join(report_full_path_shared, "tensorboard")
                writer = SummaryWriter(log_dir=tensorboard_log)
            writer.add_text("INFO-ARGUMENT-MODEL-{}".format(model_id), str(hyperparameters), 0)
        try:
            if False:
              description += ",data:{}".format(train_data_label)+ " {}".format(" ".join(["{},{}".format(key, value) for key, value in hyperparameters.items()]))
              row, col = append_reporting_sheet(git_id=gr.get_commit_id(), tasks="BERT NORMALIZE",
                                                rioc_job=os.environ.get("OAR_JOB_ID", "local"), description=description,
                                                log_dir=tensorboard_log, target_dir=model_location,
                                                env=os.environ.get("ENV", "local"), status="running",
                                                verbose=1)

        except Exception as e:
            print("REPORTING TO GOOGLE SHEET FAILED")
            print(e)

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
                              word_normalization=True,
                              force_new_dic=True if run_mode == "train" else False,
                              tasks=tasks,
                              add_start_char=1 if run_mode == "train" else None,
                              verbose=1)

    # load , mask, bucket and index data
    tokenizer = BertTokenizer.from_pretrained(voc_tokenizer)

    readers_train = readers_load(datasets=train_path, tasks=tasks, word_dictionary=word_dictionary,
                                 word_dictionary_norm=word_norm_dictionary, char_dictionary=char_dictionary,
                                 pos_dictionary=pos_dictionary, xpos_dictionary=xpos_dictionary,
                                 type_dictionary=type_dictionary, use_gpu=use_gpu,
                                 norm_not_norm=auxilliary_task_norm_not_norm, word_decoder=True,
                                 add_start_char=1, add_end_char=1, symbolic_end=1,
                                 symbolic_root=1, bucket=True, max_char_len=20,
                                 verbose=verbose)

    readers_dev = readers_load(datasets=dev_path, tasks=tasks, word_dictionary=word_dictionary,
                               word_dictionary_norm=word_norm_dictionary, char_dictionary=char_dictionary,
                               pos_dictionary=pos_dictionary, xpos_dictionary=xpos_dictionary,
                               type_dictionary=type_dictionary, use_gpu=use_gpu,
                               norm_not_norm=auxilliary_task_norm_not_norm, word_decoder=True,
                               add_start_char=1, add_end_char=1,
                               symbolic_end=1, symbolic_root=1, bucket=True, max_char_len=20,
                               verbose=verbose) if dev_path is not None else None
    # Load tokenizer
    if run_mode == "train":
        try:
            for epoch in range(n_epoch):

                checkpointing_model_data = (epoch % saving_every_epoch == 0 or epoch == (n_epoch - 1))
                # build iterator on the loaded data
                print("DEBUG:iter_train", tasks, train_path)
                batchIter_train = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers_train, batch_size=batch_size,
                                                                     word_dictionary=word_dictionary,
                                                                     char_dictionary=char_dictionary,
                                                                     pos_dictionary=pos_dictionary,
                                                                     word_dictionary_norm=word_norm_dictionary,
                                                                     get_batch_mode=False,
                                                                     extend_n_batch=1,
                                                                     dropout_input=0.0,
                                                                     verbose=verbose)
                print("DEBUG:iter_dev", tasks, dev_path)

                batchIter_dev = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers_dev, batch_size=batch_size,
                                                                   word_dictionary=word_dictionary,
                                                                   char_dictionary=char_dictionary,
                                                                   pos_dictionary=pos_dictionary,
                                                                   word_dictionary_norm=word_norm_dictionary,
                                                                   get_batch_mode=False,
                                                                   extend_n_batch=1,
                                                                   dropout_input=0.0,
                                                                   verbose=verbose) if dev_path is not None else None
                # TODO add optimizer (if not : devv loss)
                optimizer = dptx.get_optimizer(bert_with_classifier.parameters(), lr=lr)
                bert_with_classifier.train()

                loss_train, iter_train, perf_report_train = epoch_run(batchIter_train, tokenizer,
                                                                      data_label=train_data_label,
                                                                      bert_with_classifier=bert_with_classifier, writer=writer,
                                                                      iter=iter_train, epoch=epoch,
                                                                      writing_pred=checkpointing_model_data, dir_end_pred=end_predictions,
                                                                      optimizer=optimizer, use_gpu=use_gpu, predict_mode=True,
                                                                      model_id=model_id,
                                                                      null_token_index=null_token_index, null_str=null_str,
                                                                      n_iter_max=n_iter_max_per_epoch, verbose=verbose)

                bert_with_classifier.eval()
                loss_dev, iter_dev, perf_report_dev = epoch_run(batchIter_dev, tokenizer,
                                                                iter=iter_dev, use_gpu=use_gpu,
                                                                bert_with_classifier=bert_with_classifier, writer=writer,
                                                                writing_pred=checkpointing_model_data, dir_end_pred=end_predictions,
                                                                predict_mode=True, data_label=dev_data_label, epoch=epoch,
                                                                null_token_index=null_token_index, null_str=null_str,
                                                                model_id=model_id,
                                                                n_iter_max=n_iter_max_per_epoch, verbose=verbose) if dev_path is not None else None, 0, None

                printing("PERFORMANCE {} TRAIN", var=[epoch, perf_report_train],
                         verbose=verbose, verbose_level=1)
                printing("PERFORMANCE {} DEV", var=[epoch, perf_report_dev], verbose=verbose, verbose_level=1)

                printing("TRAINING : loss train:{} dev:{} for epoch {}  out of {}", var=[loss_train, loss_dev, epoch, n_epoch], verbose=1, verbose_level=1)
                checkpoint_dir = os.path.join(model_location, "{}-ep{}-checkpoint.pt".format(model_id, epoch))

                if checkpointing_model_data :
                    last_model = ""
                    if epoch == (n_epoch -1):
                        last_model = "last"
                    printing("CHECKPOINT : saving {} model {} ", var=[last_model, checkpoint_dir], verbose=verbose,
                             verbose_level=1)
                    torch.save(bert_with_classifier.state_dict(), checkpoint_dir)
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
        bert_with_classifier.eval()
        assert len(test_path_ls[0]) == 1, "ERROR 1 task supported so far for bert"
        for test_path in test_path_ls:
            label_data = "|".join([REPO_DATASET[_test_path] for _test_path in test_path])
            readers_test = readers_load(datasets=test_path, tasks=tasks, word_dictionary=word_dictionary,
                                        word_dictionary_norm=word_norm_dictionary, char_dictionary=char_dictionary,
                                        pos_dictionary=pos_dictionary, xpos_dictionary=xpos_dictionary,
                                        type_dictionary=type_dictionary, use_gpu=use_gpu,
                                        norm_not_norm=auxilliary_task_norm_not_norm, word_decoder=True,
                                        add_start_char=1, add_end_char=1, symbolic_end=1,
                                        symbolic_root=1, bucket=True, max_char_len=20,
                                        verbose=verbose)

            batchIter_test = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers_test, batch_size=batch_size,
                                                                word_dictionary=word_dictionary,
                                                                char_dictionary=char_dictionary,
                                                                pos_dictionary=pos_dictionary,
                                                                word_dictionary_norm=word_norm_dictionary,
                                                                get_batch_mode=False,
                                                                extend_n_batch=1,
                                                                dropout_input=0.0,
                                                                verbose=verbose)

            loss_test, iter_test, perf_report_test = epoch_run(batchIter_test, tokenizer,
                                                               iter=iter_dev, use_gpu=use_gpu,
                                                               bert_with_classifier=bert_with_classifier, writer=None,
                                                               writing_pred=True,
                                                               optimizer=None,
                                                               args_dir=args_dir, model_id=model_id,
                                                               dir_end_pred=end_predictions,
                                                               predict_mode=True, data_label=label_data,
                                                               epoch="LAST", extra_label_for_prediction=label_data,
                                                               null_token_index=null_token_index, null_str=null_str,
                                                               n_iter_max=n_iter_max_per_epoch, verbose=verbose)
            print("PERFORMANCE TEST on data {} is {} ".format(label_data, perf_report_test))
            if writer is not None:
                writer.add_text("Accuracy-{}-{}-{}".format(model_id, label_data, run_mode),
                                "After {} epochs with {} : performance is \n {} ".format(n_epoch, description,
                                                                                         str(perf_report_test)), 0)
            else:
                printing("WARNING : could not add accuracy to tensorboard cause writer was found None", verbose=verbose,
                         verbose_level=1)
            report_all.extend(perf_report_test)
        else:
            printing("EVALUATION none cause {} empty", var=[test_path_ls], verbose_level=1, verbose=verbose)

    if writer is not None:
        writer.close()
        printing("tensorboard --logdir={} --host=localhost --port=1234 ", var=[tensorboard_log], verbose_level=1,
                 verbose=verbose)
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


    return bert_with_classifier
