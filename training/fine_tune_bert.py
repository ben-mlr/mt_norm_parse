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


def run(tasks, train_path, dev_path, n_iter_max_per_epoch,
        voc_tokenizer, auxilliary_task_norm_not_norm, bert_with_classifier,
        null_token_index, null_str, initialize_bpe_layer=None,
        run_mode="train", test_path_ls = None, dict_path=None, end_predictions=None,
        report=True, model_suffix="", description="",
        saving_every_epoch=10, lr=0.0001, fine_tuning_strategy="standart", model_location=None, model_id=None,
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
    dev_data_label = "|".join([REPO_DATASET[_dev_path] for _dev_path in dev_path])
    if use_gpu:
        bert_with_classifier.to("cuda")

    if not debug:
        pdb.set_trace = lambda: None

    iter_train = 0
    iter_dev = 0
    row = None
    if run_mode == "train":
        assert model_location is None and model_id is None, "ERROR we are creating a new one "
        model_id, model_location, dict_path, tensorboard_log, end_predictions = \
            setup_repoting_location(model_suffix=model_suffix, root_dir_checkpoints=CHECKPOINT_BERT_DIR, verbose=verbose)
        try:
            row, col = append_reporting_sheet(git_id=gr.get_commit_id(), tasks="BERT NORMALIZE",
                                              rioc_job=os.environ.get("OAR_JOB_ID", "no"), description=description,
                                              log_dir=tensorboard_log, target_dir=model_location,
                                              env=os.environ.get("ENV", "local"), status="running",
                                              verbose=1)



        except Exception as e:
            print("REPORTING TO GOOGLE SHEET FAILED")
            print(e)

        hyperparameters = OrderedDict([("lr", lr),
                                       ("model", "bert+token_classficiation"),
                                       ("initialize_bpe_layer", initialize_bpe_layer)])
        args_dir = write_args(model_location, model_id=model_id, hyperparameters=hyperparameters, verbose=verbose)

    else:
        assert dict_path is not None
        assert end_predictions is not None
        assert model_location is not None and model_id is not None
        args_dir = os.path.join(model_location, "{}-args.json".format(model_id))
    if report and run_mode == "train":
        writer = SummaryWriter(log_dir=tensorboard_log)

        printing("CHECKPOINTING : starting writing log \ntensorboard --logdir={} --host=localhost --port=1234 ",
                 var=[os.path.join(model_id, "tensorboard")], verbose_level=1,
                 verbose=verbose)
    else:
        writer = None

    # build or make dictionaries

    word_dictionary, word_norm_dictionary, char_dictionary, pos_dictionary, \
    xpos_dictionary, type_dictionary = \
        conllu_data.load_dict(dict_path=dict_path,
                              train_path=train_path if run_mode == "train" else None,
                              dev_path=dev_path if run_mode == "train" else None,
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
                               verbose=verbose)
    # Load tokenizer
    if run_mode == "train":
        try:
            for epoch in range(n_epoch):

                checkpointing_model_data = (epoch % saving_every_epoch == 0 or epoch == (n_epoch - 1))
                # build iterator on the loaded data
                batchIter_train = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers_train, batch_size=batch_size,
                                                                     word_dictionary=word_dictionary,
                                                                     char_dictionary=char_dictionary,
                                                                     pos_dictionary=pos_dictionary,
                                                                     word_dictionary_norm=word_norm_dictionary,
                                                                     get_batch_mode=False,
                                                                     extend_n_batch=1,
                                                                     dropout_input=0.0,
                                                                     verbose=verbose)
                batchIter_dev = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers_dev, batch_size=batch_size,
                                                                   word_dictionary=word_dictionary,
                                                                   char_dictionary=char_dictionary,
                                                                   pos_dictionary=pos_dictionary,
                                                                   word_dictionary_norm=word_norm_dictionary,
                                                                   get_batch_mode=False,
                                                                   extend_n_batch=1,
                                                                   dropout_input=0.0,
                                                                   verbose=verbose)
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
                                                                null_token_index=null_token_index, null_str=null_str, model_id=model_id,
                                                                n_iter_max=n_iter_max_per_epoch, verbose=verbose)

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
                                                                       ("training_data", train_data_label),
                                                                       ("dev_data", dev_data_label)]),
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
                                                               args_dir=args_dir, model_id=model_id,
                                                               dir_end_pred=end_predictions,
                                                               predict_mode=True, data_label=label_data,
                                                               epoch="LAST", extra_label_for_prediction=label_data,
                                                               null_token_index=null_token_index, null_str=null_str,
                                                               n_iter_max=n_iter_max_per_epoch, verbose=verbose)
            print("PERFORMANCE TEST on data {} is {} ".format(label_data, perf_report_test))
            if writer is not None:
                writer.add_text("Accuracy-{}".format(label_data),
                            "After {} epochs with {} : performance is \n {} ".format(n_epoch, description, str(perf_report_test)),
                            0)
            else:
                printing("WARNING : could not add accuracy to tensorboard cause writer was found None", verbose=verbose, verbose_level=1)
        else:
            printing("EVALUATION none cause {} empty", var=[test_path_ls], verbose_level=1, verbose=verbose)

    if writer is not None:
        writer.close()
        printing("tensorboard --logdir={} --host=localhost --port=1234 ", var=[os.path.join(model_id, "tensorboard")], verbose_level=1,
                 verbose=verbose)
    if row is not None:
        update_status(row=row, new_status="done", verbose=1)
    return bert_with_classifier
