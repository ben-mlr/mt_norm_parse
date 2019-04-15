from env.importing import *
sys.path.insert(0, "..")
sys.path.insert(0, ".")

from training.train import train
from io_.info_print import printing
from evaluate.evaluate_epoch import evaluate
from env.project_variables import CHECKPOINT_DIR, REPO_DATASET
from training.args_tool import args_train
from toolbox.gpu_related import use_gpu_
from env.project_variables import PROJECT_PATH, TRAINING,LIU_TRAIN, DEMO_SENT, CP_WR_PASTE_TEST_269, \
    LIU_DEV, DEV, DIR_TWEET_W2V, TEST, DIR_TWEET_W2V, CHECKPOINT_DIR, DEMO, DEMO2, CP_PASTE_WR_TRAIN, \
    CP_WR_PASTE_DEV, CP_WR_PASTE_TEST, CP_PASTE_DEV, CP_PASTE_TRAIN, CP_PASTE_TEST, EWT_DEV, EWT_TEST, \
    LIU_DEV_SENT, LIU_TRAIN_SENT, DEV_SENT, TEST_SENT, DEMO_SENT, TRAINING_DEMO, EN_LINES_EWT_TRAIN, EN_LINES_DEV, EN_LINES_EWT_TRAIN, \
    MTNT_TOK_TRAIN, MTNT_TOK_DEV, MTNT_EN_FR_TRAIN, MTNT_EN_FR_DEV, MTNT_EN_FR_TEST, DEFAULT_SCORING_FUNCTION, WARMUP_N_EPOCHS


def get_data_set_label(data_set):
    if isinstance(data_set, str):
        return REPO_DATASET[data_set]
    elif isinstance(data_set, list):
        label = "+".join([REPO_DATASET[data] for data in data_set])
        return label


def train_eval(train_path, dev_path, model_id_pref, pos_specific_path=None,
               expand_vocab_dev_test=False,
               checkpointing_metric="loss-dev-all",
               n_epochs=11, test_path=None, args=None,
               overall_report_dir=CHECKPOINT_DIR, overall_label="DEFAULT",
               get_batch_mode_all=True,
               warmup=False, freq_checkpointing=1, debug=False,compute_scoring_curve=False,
               compute_mean_score_per_sent=False, print_raw=False, freq_scoring=5, bucketing_train=True, freq_writer=None,
               extend_n_batch=1, score_to_compute_ls=None,
               symbolic_end=False, symbolic_root=False,
               gpu=None, use_gpu=None, scoring_func_sequence_pred=DEFAULT_SCORING_FUNCTION,
               max_char_len=None,
               verbose=0):
    if gpu is not None and use_gpu_(use_gpu):
        assert use_gpu or use_gpu is None, "ERROR : use_gpu should be neutral (None) or True as 'gpu' is defined"
        #assert os.environ.get("CUDA_VISIBLE_DEVICES") is not None, "ERROR : no CUDA_VISIBLE_DEVICES env variable (gpu should be None)"
        os.environ["CUDA_VISIBLE_DEVICES"] ="1"# gpu
        printing("ENV : CUDA_VISIBLE_DEVICES set to {}", var=[gpu], verbose=verbose, verbose_level=1)

    else:
        printing("CPU mode cause {} gpu arg or use_gpu detected {} ", var=(gpu, use_gpu_(use_gpu)), verbose_level=1, verbose=verbose)

    hidden_size_encoder = args.get("hidden_size_encoder", 10)
    word_embed = args.get("word_embed", False)
    word_embedding_projected_dim = args.get("word_embedding_projected_dim", None)
    word_embedding_dim  = args.get("word_embedding_dim", 0)
    mode_word_encoding = args.get("mode_word_encoding", "cat")
    char_level_embedding_projection_dim = args.get("char_level_embedding_projection_dim", 0)

    output_dim = args.get("output_dim", 10)
    char_embedding_dim = args.get("char_embedding_dim", 10)
    hidden_size_sent_encoder = args.get("hidden_size_sent_encoder", 10)
    hidden_size_decoder = args.get("hidden_size_decoder", 10)
    batch_size = args.get("batch_size", 2)
    dropout_sent_encoder, dropout_word_encoder_cell, dropout_word_decoder = args.get("dropout_sent_encoder",0), \
    args.get("dropout_word_encoder_cell", 0), args.get("dropout_word_decoder", 0)
    n_layers_word_encoder = args.get("n_layers_word_encoder", 1)
    n_layers_sent_cell = args.get("n_layers_sent_cell", 1)
    dir_sent_encoder = args.get("dir_sent_encoder", 1)

    drop_out_word_encoder_out = args.get("drop_out_word_encoder_out", 0)
    drop_out_sent_encoder_out = args.get("drop_out_sent_encoder_out", 0)
    dropout_bridge = args.get("dropout_bridge", 0)

    word_recurrent_cell_encoder = args.get("word_recurrent_cell_encoder", "GRU")
    word_recurrent_cell_decoder = args.get("word_recurrent_cell_decoder", "GRU")
    dense_dim_auxilliary = args.get("dense_dim_auxilliary", None)
    dense_dim_auxilliary_2 = args.get("dense_dim_auxilliary_2", None)

    drop_out_char_embedding_decoder = args.get("drop_out_char_embedding_decoder", 0)
    unrolling_word= args.get("unrolling_word", False)

    #auxilliary_task_norm_not_norm = args.get("auxilliary_task_norm_not_norm",False)
    char_src_attention = args.get("char_src_attention",False)
    weight_binary_loss = args.get("weight_binary_loss", 1)
    dir_word_encoder = args.get("dir_word_encoder", 1)
    shared_context = args.get("shared_context", "all")

    schedule_training_policy = args.get("policy", None)
    lr = args.get("lr", 0.001)
    gradient_clipping = args.get("gradient_clipping", None)

    teacher_force = args.get("teacher_force", True)
    proportion_pred_train = args.get("proportion_pred_train",None)
    if teacher_force and proportion_pred_train is not None:
        printing("WARNING : inconsistent arguments solved :  proportion_pred_train forced to None while it was {} "
                 "cause teacher_force mode", var=[proportion_pred_train], verbose=verbose, verbose_level=0)
        proportion_pred_train = None

    stable_decoding_state = args.get("stable_decoding_state", False)
    init_context_decoder = args.get("init_context_decoder", True)
    optimizer = args.get("optimizer", "adam")

    word_decoding = args.get("word_decoding", False)
    dense_dim_word_pred = args.get("dense_dim_word_pred", None)
    dense_dim_word_pred_2 = args.get("dense_dim_word_pred_2", None)
    dense_dim_word_pred_3 = args.get("dense_dim_word_pred_3", 0)
    word_embed_init = args.get("word_embed_init", None)

    char_decoding = args.get("char_decoding", True)

    #auxilliary_task_pos = args.get("auxilliary_task_pos", False)
    dense_dim_auxilliary_pos = args.get("dense_dim_auxilliary_pos", None)
    dense_dim_auxilliary_pos_2 = args.get("dense_dim_auxilliary_pos_2", None)

    activation_char_decoder = args.get("activation_char_decoder", None)
    activation_word_decoder = args.get("activation_word_decoder", None)
    tasks = args.get("tasks", ["normalize"])

    attention_tagging = args.get("attention_tagging", False)

    multi_task_loss_ponderation = args.get("multi_task_loss_ponderation", "all")
    dropout_input = args.get("dropout_input", None)

    n_epochs = WARMUP_N_EPOCHS if warmup else n_epochs

    if test_path is not None:
        assert isinstance(test_path, list),"ERROR test_path should be a list with one element per task "
        assert isinstance(test_path[0], list), "ERROR : each element of test_path should be a list of dataset path even of len 1"
    print("WARNING : only dataset that are in test_path will be evlauated (test_path:{}) ".format(test_path))
    if warmup:
        printing("Warm up : running 1 epoch ", verbose=verbose, verbose_level=0)
    printing("GRID : START TRAINING ", verbose_level=0, verbose=verbose)
    printing("SANITY CHECK : TASKS {} ", var=[tasks], verbose=verbose, verbose_level=1)
    normalization = "normalize" in tasks or "norm_not_norm" in tasks
    printing("SANITY CHECK : normalization {} ", var=normalization, verbose=verbose, verbose_level=1)
    model_full_name = train(train_path, dev_path, pos_specific_path=pos_specific_path,
                            checkpointing_metric=checkpointing_metric,
                            expand_vocab_dev_test=expand_vocab_dev_test if word_embed_init is not None else False,
                            dense_dim_auxilliary=dense_dim_auxilliary, dense_dim_auxilliary_2=dense_dim_auxilliary_2,
                            lr=lr, extend_n_batch=extend_n_batch,
                            n_epochs=n_epochs, normalization=normalization,get_batch_mode_all=get_batch_mode_all,
                            batch_size=batch_size, model_specific_dictionary=True, freq_writer=freq_writer,
                            dict_path=None, model_dir=None, add_start_char=1,
                            freq_scoring=freq_scoring,
                            add_end_char=1, use_gpu=use_gpu, dir_sent_encoder=dir_sent_encoder,
                            dropout_sent_encoder_cell=dropout_sent_encoder,
                            dropout_word_encoder_cell=dropout_word_encoder_cell,
                            dropout_word_decoder_cell=dropout_word_decoder,
                            policy=schedule_training_policy,
                            dir_word_encoder=dir_word_encoder, compute_mean_score_per_sent=compute_mean_score_per_sent,
                            overall_label=overall_label, overall_report_dir=overall_report_dir,
                            label_train=get_data_set_label(train_path), label_dev=get_data_set_label(dev_path),
                            word_recurrent_cell_encoder=word_recurrent_cell_encoder, word_recurrent_cell_decoder=word_recurrent_cell_decoder,
                            drop_out_sent_encoder_out=drop_out_sent_encoder_out, drop_out_char_embedding_decoder=drop_out_char_embedding_decoder,
                            word_embedding_dim=word_embedding_dim, word_embed=word_embed, word_embedding_projected_dim=word_embedding_projected_dim,
                            mode_word_encoding=mode_word_encoding, char_level_embedding_projection_dim=char_level_embedding_projection_dim,
                            drop_out_word_encoder_out=drop_out_word_encoder_out, dropout_bridge=dropout_bridge,
                            freq_checkpointing=freq_checkpointing, reload=False, model_id_pref=model_id_pref,
                            score_to_compute_ls=score_to_compute_ls, mode_norm_ls=["all", "NEED_NORM", "NORMED"],
                            hidden_size_encoder=hidden_size_encoder, output_dim=output_dim,
                            char_embedding_dim=char_embedding_dim,extern_emb_dir=word_embed_init,
                            hidden_size_sent_encoder=hidden_size_sent_encoder, hidden_size_decoder=hidden_size_decoder,
                            n_layers_word_encoder=n_layers_word_encoder, n_layers_sent_cell=n_layers_sent_cell,
                            compute_scoring_curve=compute_scoring_curve,
                            unrolling_word=unrolling_word, char_src_attention=char_src_attention,
                            print_raw=print_raw, debug=debug,
                            shared_context=shared_context,
                            bucketing=bucketing_train, weight_binary_loss=weight_binary_loss,
                            teacher_force=teacher_force, proportion_pred_train=proportion_pred_train,
                            clipping=gradient_clipping,
                            tasks=tasks,
                            optimizer=optimizer,
                            #auxilliary_task_pos=auxilliary_task_pos,
                            dense_dim_auxilliary_pos=dense_dim_auxilliary_pos,
                            dense_dim_auxilliary_pos_2=dense_dim_auxilliary_pos_2,
                            word_decoding=word_decoding, dense_dim_word_pred=dense_dim_word_pred,
                            dense_dim_word_pred_2=dense_dim_word_pred_2, dense_dim_word_pred_3=dense_dim_word_pred_3,
                            char_decoding=char_decoding,
                            activation_char_decoder=activation_char_decoder,
                            activation_word_decoder=activation_word_decoder,
                            symbolic_end=symbolic_end, symbolic_root=symbolic_root,
                            attention_tagging=attention_tagging,
                            stable_decoding_state=stable_decoding_state, init_context_decoder=init_context_decoder,
                            multi_task_loss_ponderation=multi_task_loss_ponderation,
                            dropout_input=dropout_input,
                            test_path=test_path[0] if isinstance(test_path, list) else test_path,
                            max_char_len=max_char_len,
                            checkpointing=True, verbose=verbose)

    model_dir = os.path.join(CHECKPOINT_DIR, model_full_name+"-folder")
    if test_path is not None:
      dict_path = os.path.join(CHECKPOINT_DIR, model_full_name+"-folder", "dictionaries")
      printing("GRID : START EVALUATION FINAL ", verbose_level=0, verbose=verbose)
      # you have to specify all data you want to evaluate !!
      eval_data_paths = test_path
      #eval_data_paths = list(set(eval_data_paths))
      start_eval = time.time()
      if len(tasks) > 1:
          assert isinstance(eval_data_paths, list), "ERROR : on element per task"
          assert isinstance(eval_data_paths[0], list), "ERROR : in multitask we want list of list for eval_data_paths {} one sublist per task {} ".format(eval_data_paths, tasks)
      if len(tasks) == 1:
          tasks = [tasks[0] for _ in eval_data_paths]
      for get_batch_mode_evaluate in [False]:
        print("EVALUATING WITH {}".format(scoring_func_sequence_pred))
        for task, eval_data in zip(tasks, eval_data_paths):
            for eval_data_set in eval_data:
                printing("EVALUATING task {} on dataset {}", var=[task, eval_data_set], verbose=verbose, verbose_level=1)
                evaluate(model_full_name=model_full_name, data_path=eval_data_set,
                         dict_path=dict_path, use_gpu=use_gpu,
                         label_report=REPO_DATASET[eval_data_set],
                         overall_label=overall_label+"-last",
                         score_to_compute_ls=score_to_compute_ls, mode_norm_ls=["all", "NEED_NORM", "NORMED"],
                         normalization=normalization, print_raw=print_raw,
                         model_specific_dictionary=True, get_batch_mode_evaluate=get_batch_mode_evaluate, bucket=True,
                         compute_mean_score_per_sent=compute_mean_score_per_sent,
                         batch_size=batch_size, debug=debug,
                         word_decoding=word_decoding, char_decoding=char_decoding,
                         scoring_func_sequence_pred=scoring_func_sequence_pred,
                         evaluated_task=task, tasks=tasks,
                         max_char_len=max_char_len,
                         dir_report=model_dir, verbose=1)
        printing("GRID : END EVAL {:.3f}s ".format(time.time()-start_eval), verbose=verbose, verbose_level=1)
    printing("WARNING : no evaluation ", verbose=verbose, verbose_level=0)

    return model_full_name, model_dir


if __name__ == "__main__":

    # just here to test train_eval workflow for prod like run : train_evaluate_run.py
    args = args_train(mode="script")
    params = vars(args)

    if args.train_path is None:
        args.train_path = DEMO
    if args.dev_path is None:
        args.dev_path = DEMO2
    if args.model_id_pref is None:
        args.model_id_pref = "TEST"
    print(params)
    train_eval(args=params,
               model_id_pref=args.model_id_pref,
               train_path=args.train_path,
               dev_path=args.dev_path,
               expand_vocab_dev_test=args.expand_vocab_dev_test,
               pos_specific_path=args.pos_specific_path,
               overall_label=args.overall_label, debug=args.debug,
               extend_n_batch=2,
               get_batch_mode_all=True, compute_mean_score_per_sent=False, bucketing_train=True, freq_checkpointing=1,
               symbolic_end=True, symbolic_root=True, freq_writer=1, compute_scoring_curve=False,
               verbose=args.verbose, warmup=args.warmup)

    #print(vars(argparse.Namespace()))


