from env.importing import *

from io_.info_print import printing
from training.train_eval import train_eval
from toolbox.grid_tool import grid_param_label_generate, get_experimented_tasks
from env.project_variables import *

from toolbox.git_related import get_commit_id
from tracking.reporting_google_sheet import update_status, append_reporting_sheet
from toolbox.grid_script_generation import script_generation

FINE_TUNE = 0
GRID = 1


def run_grid(parameters, labels, dir_grid, label_grid,
             scoring_func_sequence_pred=DEFAULT_SCORING_FUNCTION,
             print_raw=False,
             epochs=50, test_before_run=False, debug=False, warmup=False):
    i = 0

    for param, model_id_pref in zip(parameters, labels):
        i += 1
        printing("GRID RUN : RUN_ID {} as prefix".format(RUN_ID), verbose=0, verbose_level=0)
        epochs = epochs if not test_before_run else 1
        if warmup:
            if len(parameters[0]["tasks"]) > 1:
                param["train_path"] = [DEMO, DEMO]
                param["dev_path"] = [DEMO, DEMO]
                param["test_path"] = [[DEV], [LIU_DEV]]
                # TODO : should add assertion on test_paths also
            else:
                param["train_path"] = [DEMO]
                param["dev_path"] = [DEMO2]
                param["test_path"] = [[DEMO2]]
            #param["word_embed_init"] = None

        model_id_pref = label_grid + model_id_pref + "-model_" + str(i)

        if warmup:
            print("GRID RUN : MODEL {} with param {} ".format(model_id_pref, param))
            print("GRID_INFO analy vars=    dense_dim_auxilliary_pos_2 dense_dim_auxilliary_pos")
            print("GRID_INFO fixed vars=  word_embed ")
            print("GRID_INFO fixed vals=  word_embed,False ")

        model_full_name, model_dir = train_eval(train_path=param["train_path"], dev_path=param["dev_path"],
                                                model_id_pref=model_id_pref,
                                                expand_vocab_dev_test=True,
                                                test_path=param["test_path"],
                                                overall_report_dir=dir_grid, overall_label=LABEL_GRID,
                                                compute_mean_score_per_sent=True, print_raw=print_raw,
                                                get_batch_mode_all=True, compute_scoring_curve=True,
                                                freq_scoring=1, freq_checkpointing=1,
                                                bucketing_train=True,
                                                symbolic_root=True, symbolic_end=True,
                                                freq_writer=1 if not test_before_run else 1,
                                                extend_n_batch=1,
                                                checkpointing_metric=param["checkpointing_metric"],
                                                scoring_func_sequence_pred=scoring_func_sequence_pred,
                                                score_to_compute_ls=["exact", "norm_not_norm-F1",
                                                                     "norm_not_norm-Precision",
                                                                     "norm_not_norm-Recall",
                                                                     "norm_not_norm-accuracy"],
                                                warmup=warmup, args=param,
                                                use_gpu=None, n_epochs=epochs,
                                                max_char_len=20,
                                                debug=debug,
                                                verbose=1)

        run_dir = os.path.join(dir_grid, RUN_ID + "-run-log")
        open(run_dir, "a").write("model : done " + model_full_name + " in " + model_dir + " \n")
        print("GRID : Log RUN is : {} to see model list ".format(run_dir))
        print("GRID RUN : DONE MODEL {} with param {} ".format(model_id_pref, param))
        if warmup or test_before_run:
            # breaking after testing first modl
            break


if __name__ == "__main__":

      if platform != "darwin":
        printing("RUN : running in rioc or neff", verbose=1, verbose_level=1)
        assert os.environ.get("MODE_RUN") is not None, "Running in rioc, MODE_RUN empty while it should not "
        assert os.environ.get("MODE_RUN") in ["DISTRIBUTED", "SINGLE"]
        run_standart = os.environ.get("MODE_RUN") != "DISTRIBUTED"
      else:
          run_standart = False
          print("LOCAL")

      params = []

      ls_param = ["hidden_size_encoder", "hidden_size_sent_encoder", "hidden_size_decoder", "output_dim", "char_embedding_dim"]

      params_strong = {"hidden_size_encoder": 100, "output_dim": 100, "char_embedding_dim": 50,
                       "dropout_sent_encoder": 0.0, "drop_out_word_encoder": 0.0, "dropout_word_decoder": 0.,
                       "drop_out_word_encoder_out": 0.0, "drop_out_sent_encoder_out": 0.0,
                       "drop_out_char_embedding_decoder": 0., "dropout_bridge": 0.00,
                       "n_layers_word_encoder": 1, "dir_sent_encoder": 2,
                       "word_recurrent_cell_decoder": "LSTM", "word_recurrent_cell_encoder":"LSTM",
                       "hidden_size_sent_encoder": 100, "hidden_size_decoder": 200, "batch_size": 10}

      params_strong_tryal = {"hidden_size_encoder": 12, "output_dim": 20, "char_embedding_dim": 20,
                             "dropout_sent_encoder": 0, "dropout_word_encoder_cell": 0, "dropout_word_decoder": 0.,
                             "drop_out_word_encoder_out": 0., "drop_out_sent_encoder_out": 0.,
                             "drop_out_char_embedding_decoder": 0., "dropout_bridge": 0.0,
                             "n_layers_word_encoder": 1, "dir_sent_encoder": 2, "word_recurrent_cell_decoder": "LSTM",
                             "word_recurrent_cell_encoder": "LSTM",
                             "hidden_size_sent_encoder": 12,
                             "hidden_size_decoder": 12, "batch_size": 10}

      params_dozat = {"hidden_size_encoder": 400, "output_dim": 100, "char_embedding_dim": 100,
                      "dropout_sent_encoder": 0.5, "dropout_word_decoder": 0.0,
                      "drop_out_word_encoder_out": 0.33, "drop_out_sent_encoder_out": 0.33,
                      "drop_out_char_embedding_decoder": 0.33, "dropout_bridge": 0.33,
                      "n_layers_word_encoder": 1, "dir_sent_encoder": 2, "word_recurrent_cell_decoder": "LSTM",
                      "word_recurrent_cell_encoder": "LSTM",
                       "hidden_size_sent_encoder": 400, "hidden_size_decoder": 100, "batch_size": 500}

      grid_label = "B"#"POS-2LSMT-2dense+no_aux_task-sent_only-EWT_DEV-PONDERATION-1pos-0_norm"
      # param["policy"] = policy
      # param["drop_out_sent_encoder_out"] = 0.2#add_dropout_encoder
      # param["drop_out_word_encoder_out"] = 0.2#add_dropout_encoder
      # param["dropout_bridge"] = 0.2 #add_dropout_encoder
      # param["drop_out_char_embedding_decoder"] = add_dropout_encoder
      # param["dense_dim_auxilliary"] = dense_dim_auxilliary
      # param["dense_dim_auxilliary_2"] = dense_dim_auxilliary_2
      # param["dense_dim_word_pred"] = 200 if word_decoding else None
      # param["dense_dim_word_pred_2"] = 200 if word_decoding else None
      # param["dense_dim_word_pred_3"] = 100 if word_decoding else None
      # param["dense_dim_auxilliary_pos"] = None if not auxilliary_task_pos else 200
      # param["dense_dim_auxilliary_pos_2"] = None

      if run_standart:
          # default not used but could be
          #train_path, dev_path = MTNT_EN_FR_TRAIN, MTNT_EN_FR_DEV#MTNT_EN_FR_TRAIN, MTNT_EN_FR_DEV #EN_LINES_EWT_TRAIN, EWT_DEV  # MTNT_TOK_TRAIN, MTNT_TOK_DEV#EN_LINES_EWT_TRAIN, EWT_DEV # MTNT_EN_FR_TRAIN, MTNT_EN_FR_DEV #MTNT_TOK_TRAIN, MTNT_TOK_DEV#EN_LINES_EWT_TRAIN, EWT_DEV#CP_PASTE_WR_TRAIN, CP_WR_PASTE_DEV#TRAINING, EWT_DEV #LIU_TRAIN, LIU_DEV ## EWT_DEV, DEV
          # [TEST_SENT, MTNT_EN_FR_TEST, MTNT_EN_FR_DEV],#[TEST, TEST],#[EWT_TEST, EWT_DEV, EN_LINES_EWT_TRAIN, TEST], # [TEST_SENT, MTNT_EN_FR_TEST, MTNT_EN_FR_DEV],#
          #train_path = [[DEMO,DEMO2]]
          #dev_path = [[DEMO, DEMO2]]
          #test_path = [[[DEMO], [DEMO2]]]
          train_path = [[DEMO2]]
          dev_path = [[DEMO2]]
          test_path = [[[DEMO2]]]
          # TODO : test with normalize and other multi tasks !!
          params, labels, default_all, analysed, fixed = grid_param_label_generate(params_strong_tryal,
                                                                                   train_ls=train_path,
                                                                                   dev_ls=dev_path,
                                                                                   test_ls=test_path,
                                                                                   checkpointing_metric_ls=["loss-dev-all"],
                                                                                   grid_label="0",
                                                                                   word_recurrent_cell_encoder_ls=["LSTM"],
                                                                                   dropout_word_encoder_cell_ls=[0.0],
                                                                                   stable_decoding_state_ls=[0],
                                                                                   word_decoding_ls=[0],
                                                                                   batch_size_ls=[2], word_embed_ls=[0],
                                                                                   dir_sent_encoder_ls=[2], dir_word_encoder_ls=[2],
                                                                                   lr_ls=[0.0005],
                                                                                   word_embed_init_ls=[None],
                                                                                   #, DIR_FASTEXT_WIKI_NEWS_W2V, DIR_TWEET_W2V],
                                                                                   attention_tagging_ls=[1],
                                                                                   char_src_attention_ls=[1],
                                                                                   teacher_force_ls=[True],
                                                                                   proportion_pred_train_ls=[0],
                                                                                   shared_context_ls=["word"],
                                                                                   word_embedding_projected_dim_ls=[0],
                                                                                   char_level_embedding_projection_dim_ls=[30],
                                                                                   tasks_ls=[["normalize"]],
                                                                                   n_layers_sent_cell_ls=[1],
                                                                                   n_layers_word_encoder_ls=[1],
                                                                                   unrolling_word_ls=[1],
                                                                                   scoring_func="exact_match",
                                                                                   mode_word_encoding_ls=["sum"],
                                                                                   dropout_input_ls=[0.0],
                                                                                   multi_task_loss_ponderation_ls=[{"pos": 1, "normalize": 1,
                                                                                                                    "norm_not_norm": 0,
                                                                                                                    "edit_prediction": 1}],
                                                                                   scale_ls=[1])



        # only for cloud run :
      warmup = True
      if platform != "darwin":
          printing("GRID : ENV : running not from os x assuming we are in command shell run", verbose=0, verbose_level=0)
          parser = argparse.ArgumentParser()
          parser.add_argument("--test_before_run", help="test_before_run", action="store_true")
          parser.add_argument("--desc", help="describe run for reporting", default="", required=False, type=str)
          parser.add_argument("--n_gpu", help="describe run for reporting", required=False, default=None, type=int)
          args = parser.parse_args()
          test_before_run = args.test_before_run
          gpu_ls = GPU_AVAILABLE_DEFAULT_LS if args.n_gpu is None else GPU_AVAILABLE_DEFAULT_LS[:args.n_gpu]
          description_comment = args.desc
          print("GRID : test_before_run set to {} ".format(test_before_run))
          warmup = False
          environment = os.environ.get("ENV","NO ENV FOUND")
          OAR = os.environ.get('OAR_JOB_ID', "")
          print("OAR=", OAR)
          log = "{}/logs/{}".format(os.getcwd(), os.environ.get('OAR_JOB_ID')+"-job.stdout")
      else:
          OAR=""
          environment = "local"
          log = "in the fly logs"
          test_before_run = False
          description_comment = "addslij"
          warmup = False
          gpu_ls = GPU_AVAILABLE_DEFAULT_LS

      RUN_ID = str(uuid4())[0:5]
      LABEL_GRID = grid_label if not warmup else "WARMUP-unrolling-False"
      LABEL_GRID = "test_before_run-"+LABEL_GRID if test_before_run else LABEL_GRID
      OAR = RUN_ID if OAR == "" else OAR
      LABEL_GRID = OAR+"-" + LABEL_GRID
      GRID_FOLDER_NAME = LABEL_GRID if len(LABEL_GRID) > 0 else RUN_ID
      GRID_FOLDER_NAME += "-summary"
      dir_grid = os.path.join(CHECKPOINT_DIR, GRID_FOLDER_NAME)
      os.mkdir(dir_grid)
      printing("GRID RUN : Grid directory : dir_grid {} made".format(dir_grid), verbose=0, verbose_level=0)

      if warmup or test_before_run:
          printing("GRID RUN WARMUP will run for 1 epoch only the first model of the grid (warmup {} test_before_run {}) ",
                   var=[warmup, test_before_run], verbose=0, verbose_level=0)

      if run_standart:
          warmup_desc = "warmup" if warmup else ""
          test_before_run_desc = "test_before_run" if test_before_run else ""
          mode_run = "sing"
          description = "{} - {} ({}) : Analysing : {} with regard to {} fixed".format(len(params) if not (warmup or test_before_run) else str(1)+"_WARMUP",
                                                                                       description_comment, mode_run,
                                                                                       analysed, fixed)
          try:
              row, col = append_reporting_sheet(git_id=get_commit_id(), tasks=get_experimented_tasks(params),rioc_job=OAR, description=description, log_dir=log,
                                                target_dir=dir_grid + " | " + os.path.join(CHECKPOINT_DIR, "{}*".format(LABEL_GRID)),
                                                env=environment, status="running {}{}".format(warmup_desc, test_before_run_desc),
                                                verbose=1)
          except:
              row = None
          print("row:{}".format(row))
          run_grid(parameters=params, labels=labels, dir_grid=dir_grid,
                   label_grid=LABEL_GRID,
                   epochs=30,
                   test_before_run=test_before_run,
                   debug=True, print_raw=True,
                   scoring_func_sequence_pred="exact_match",
                   warmup=warmup)
          if row is not None:
              update_status(row=row, new_status="done {}".format(warmup_desc), verbose=1)
      else:
          train_path, dev_path = EN_LINES_EWT_TRAIN, EWT_DEV#MTNT_TOK_TRAIN, MTNT_TOK_DEV#EN_LINES_EWT_TRAIN, EWT_DEV  # MTNT_TOK_TRAIN, MTNT_TOK_DEV#EN_LINES_EWT_TRAIN, EWT_DEV # MTNT_EN_FR_TRAIN, MTNT_EN_FR_DEV #MTNT_TOK_TRAIN, MTNT_TOK_DEV#EN_LINES_EWT_TRAIN, EWT_DEV#CP_PASTE_WR_TRAIN, CP_WR_PASTE_DEV#TRAINING, EWT_DEV #LIU_TRAIN, LIU_DEV ## EWT_DEV, DEV
          NORMALIZE = False
          if NORMALIZE:
              epochs = 150
              #train_path, dev_path = CP_PASTE_WR_TRAIN, CP_WR_PASTE_DEV #MTNT_TOK_TRAIN, MTNT_TOK_DEV#EN_LINES_EWT_TRAIN, EWT_DEV  # MTNT_TOK_TRAIN, MTNT_TOK_DEV#EN_LINES_EWT_TRAIN, EWT_DEV # MTNT_EN_FR_TRAIN, MTNT_EN_FR_DEV #MTNT_TOK_TRAIN, MTNT_TOK_DEV#EN_LINES_EWT_TRAIN, EWT_DEV#CP_PASTE_WR_TRAIN, CP_WR_PASTE_DEV#TRAINING, EWT_DEV #LIU_TRAIN, LIU_DEV ## EWT_DEV, DEV
              n_sents = 10000
              train_path = [[PERMUTATION_TRAIN_DIC[n_sents]]]
              dev_path = [[PERMUTATION_TEST]]
              test_path = [[[PERMUTATION_TRAIN_DIC[n_sents], PERMUTATION_TEST, TEST]]]
              dir_script, row = script_generation(grid_label=LABEL_GRID, 
                                                  init_param=params_strong_tryal,#params_dozat,#params_strong,#params_dozat,
                                                  warmup=test_before_run, test_before_run=test_before_run,
                                                  dir_grid=dir_grid, environment=environment, dir_log=log,
                                                  stable_decoding_state_ls=[0],
                                                  word_decoding_ls=[0],
                                                  epochs=epochs if not (test_before_run or warmup) else WARMUP_N_EPOCHS,
                                                  batch_size_ls=[20],
                                                  word_embed_ls=[0],
                                                  dir_sent_encoder_ls=[2], dir_word_encoder_ls=[2],
                                                  n_layers_sent_cell_ls=[1], n_layers_word_encoder_ls=[1],
                                                  lr_ls=[0.001],
                                                  word_embed_init_ls=[None],
                                                  teacher_force_ls=[1],
                                                  word_recurrent_cell_encoder_ls=["LSTM"],
                                                  dropout_word_encoder_cell_ls=[0.],
                                                  proportion_pred_train_ls=[None],
                                                  shared_context_ls=["word"],
                                                  word_embedding_projected_dim_ls=[0],
                                                  char_level_embedding_projection_dim_ls=[0],
                                                  mode_word_encoding_ls=["cat"],
                                                  tasks_ls=[["normalize"]],
                                                  char_src_attention_ls=[1], attention_tagging_ls=[0],
                                                  unrolling_word_ls=[1],
                                                  scale_ls=[1, 5, 10],
                                                  overall_report_dir=dir_grid, overall_label=LABEL_GRID,
                                                  description_comment=description_comment,
                                                  train_path=train_path, dev_path=dev_path,
                                                  test_paths=test_path,
                                                  gpu_mode="random",
                                                  gpus_ls=gpu_ls,
                                                  scoring_func="exact_match",
                                                  dropout_input_ls=[0.0, 0.2, 0.4],
                                                  multi_task_loss_ponderation_ls=[{"pos": 0, "normalize": 1, "norm_not_norm":0, "edit_prediction":0}],
                                                  write_to_dir=RUN_SCRIPTS_DIR)
          
          MULTI_TASK = False 
          if MULTI_TASK:
              epochs=150
              train_path = [[EN_LINES_EWT_TRAIN], [EN_LINES_EWT_TRAIN, LIU_TRAIN], [EN_LINES_EWT_TRAIN, LIU_TRAIN]]#[DEMO, DEMO]#
              dev_path = [[EWT_DEV],[EWT_DEV, LIU_DEV], [EWT_DEV, LIU_DEV]]#train_path#
              test_paths = [[[EWT_DEV, EWT_TEST, DEV, TEST]], [[EWT_DEV, EWT_TEST, DEV, TEST], [TEST, LIU_DEV]], [[EWT_DEV, EWT_TEST, DEV, TEST], [TEST, LIU_DEV]]]
              #[[LIU_DEV],[EWT_TEST]]#
              dir_script, row = script_generation(init_param=params_dozat,
                                                  train_path=train_path, dev_path=dev_path, test_paths=test_paths,
                                                  grid_label=LABEL_GRID,
                                                  word_recurrent_cell_encoder_ls=["LSTM"],
                                                  dropout_word_encoder_cell_ls=[0.1],
                                                  stable_decoding_state_ls=[0],
                                                  word_decoding_ls=[0],
                                                  batch_size_ls=[80, 120, 200],
                                                  word_embed_ls=[0],
                                                  dir_sent_encoder_ls=[2],dir_word_encoder_ls=[2],
                                                  lr_ls=[0.001],
                                                  word_embed_init_ls=[None],#, DIR_FASTEXT_WIKI_NEWS_W2V, DIR_TWEET_W2V],
                                                  attention_tagging_ls=[0],
                                                  char_src_attention_ls=[0],
                                                  teacher_force_ls=[1],
                                                  proportion_pred_train_ls=[None],
                                                  shared_context_ls=["word"],
                                                  word_embedding_projected_dim_ls=[125],
                                                  char_level_embedding_projection_dim_ls=[125],
                                                  tasks_ls=[["pos"],
                                                            ["pos", "edit_prediction"],
                                                            ["pos", "norm_not_norm"]],
                                                  n_layers_sent_cell_ls=[2],
                                                  n_layers_word_encoder_ls=[1],
                                                  unrolling_word_ls=[1],
                                                  scoring_func="exact_match",
                                                  mode_word_encoding_ls=["sum"],
                                                  dropout_input_ls=[0.4, 0.5,  0.6],
                                                  multi_task_loss_ponderation_ls=[#{"pos": 1, "normalize": 0, "norm_not_norm": 0, "edit_prediction":0},
                                                                                   {"pos": 1, "normalize": 0, "norm_not_norm": 1, "edit_prediction":1},
                                                                                 ],
                                                  scale_ls=[1],
                                                  # arguments that are specific to script generation
                                                  overall_report_dir=dir_grid, overall_label=LABEL_GRID,
                                                  warmup=test_before_run, test_before_run=test_before_run,
                                                  dir_grid=dir_grid, environment=environment, dir_log=log,
                                                  epochs=epochs if not (test_before_run or warmup) else WARMUP_N_EPOCHS,
                                                  gpus_ls=gpu_ls, gpu_mode="random",
                                                  write_to_dir=RUN_SCRIPTS_DIR, description_comment=description_comment)
          POS_ABLATION = False
          if POS_ABLATION:
            epochs = 150
            train_path = [[EN_LINES_EWT_TRAIN]]#,[EN_LINES_EWT_TRAIN]]#[DEMO, DEMO]#
            dev_path = [[EWT_DEV]]#,[EWT_DEV]]#train_path#
            test_paths = [[[EWT_DEV, TEST]]]#[[[EWT_DEV, EWT_TEST, DEV, TEST]], [[EWT_DEV, EWT_TEST, DEV, TEST]]]#,[[EWT_DEV, EWT_TEST, DEV, TEST]]]
            dir_script, row = script_generation(grid_label=LABEL_GRID,
                                                train_path=train_path, dev_path=dev_path,
                                                test_paths=test_paths, checkpointing_metric_ls=["accuracy-pos"],#"loss-dev-all"],
                                                init_param=params_dozat,#params_dozat,#params_strong,#params_dozat,
                                                warmup=test_before_run, test_before_run=test_before_run,
                                                dir_grid=dir_grid, environment=environment, dir_log=log,
                                                stable_decoding_state_ls=[0],
                                                word_decoding_ls=[0],
                                                epochs=epochs if not (test_before_run or warmup) else WARMUP_N_EPOCHS,
                                                batch_size_ls=[150, 300],
                                                word_embed_ls=[1],
                                                dir_sent_encoder_ls=[2], dir_word_encoder_ls=[1],
                                                n_layers_sent_cell_ls=[2], n_layers_word_encoder_ls=[1],
                                                lr_ls=[0.001],
                                                word_embed_init_ls=[None],
                                                teacher_force_ls=[1],
                                                word_recurrent_cell_encoder_ls=["LSTM"],
                                                dropout_word_encoder_cell_ls=[0.33],
                                                proportion_pred_train_ls=[None],
                                                shared_context_ls=["sent"],
                                                word_embedding_projected_dim_ls=[125],
                                                char_level_embedding_projection_dim_ls=[125],
                                                mode_word_encoding_ls=["sum"],
                                                tasks_ls=[["pos"]],
                                                char_src_attention_ls=[0],
                                                unrolling_word_ls=[1],
                                                scale_ls=[1],
                                                attention_tagging_ls=[1],
                                                overall_report_dir=dir_grid, overall_label=LABEL_GRID,
                                                description_comment=description_comment,
                                                gpu_mode="random",
                                                gpus_ls=gpu_ls,
                                                scoring_func="exact_match", dropout_input_ls=[0.33, 0.4],
                                                multi_task_loss_ponderation_ls=[{"pos": 1, "normalize": 0,
                                                                                 "norm_not_norm": 0,
                                                                                 "edit_prediction": 0}],
                                                write_to_dir=RUN_SCRIPTS_DIR)

          BERT_NORMALIZE = True
          if BERT_NORMALIZE:
              epochs = 1
              dir_script, row = script_generation(py_script="train_evaluate_bert_normalizer",
                                                  init_param=None,
                                                  grid_label=LABEL_GRID,
                                                  batch_size_ls=[1],
                                                  #checkpoint_dir_ls=["'"+os.path.join(PROJECT_PATH,
                                                  #                    "checkpoints", "bert", 
                                                  #                    "9318015-B-133b1-9318015-B-model_6/9318015-B-133b1-9318015-B-model_6-ep20-checkpoint.pt")+"'"],
                                                  gpu_mode="random",
                                                  #norm_2_noise_training_ls=[0., 1.],
                                                  lr_ls=[0.00001],
                                                  masking_strategy_ls=[None ],# ["normed", "0.05"],["normed", "0.1"],["normed", "0.2"]],
                                                  #                     ["normed", "0.75"],["normed", "1."]],#[None,,
                                                  #lr_ls=[OrderedDict([("bert", "0.00001"), ("classifier", "0.0001")]),
                                                  #       OrderedDict([("bert", "0.00001"), ("classifier", "0.00001")])],
                                                  tasks_ls=[["normalize"]],
                                                  fine_tuning_strategy_ls=["standart"],
                                                  dropout_classifier_ls=[0.3],
                                                  #dropout_input_bpe_ls=[0., 0.25, 0.5],
                                                  dropout_bert_ls=[0.1],
                                                  #gold_error_detection_ls=[0], heuristic_ls_ls=[["'#'","'@'"]],
                                                  overall_report_dir=dir_grid, overall_label=LABEL_GRID,  
                                                  train_path=[[GENERATED_DIC[100], EWT_DEV]],  dev_path=[[LIU_DEV, DEMO]],
                                                  #test_paths=[[[DEV], [LIU_DEV], [LEX_TEST],[LEX_TRAIN],[LEX_TEST], [LIU_TRAIN], [TEST]]],
                                                  #test_paths=[[[DEV], [LIU_DEV], [LIU_TRAIN], [TEST], [LEX_TEST]]],
                                                  test_paths=[[[DEMO, LIU_DEV], [DEMO, EWT_TEST]]],
                                                  warmup=test_before_run, test_before_run=test_before_run,
                                                  dir_grid=dir_grid, environment=environment, dir_log=log,
                                                  epochs=epochs if not (test_before_run or warmup) else WARMUP_N_EPOCHS,
                                                  gpus_ls=gpu_ls,
                                                  write_to_dir=RUN_SCRIPTS_DIR, description_comment=description_comment,
                                                  bert_model_ls=["cased"], initialize_bpe_layer_ls=[1],
                                                  freeze_parameters_ls=[0], freeze_layer_prefix_ls_ls=[None],
                                                  word_recurrent_cell_encoder_ls=None, dropout_word_encoder_cell_ls=None,
                                                  stable_decoding_state_ls=None,
                                                  word_decoding_ls=None,word_embed_ls=None, dir_sent_encoder_ls=None,
                                                  dir_word_encoder_ls=None,
                                                  word_embed_init_ls=None, attention_tagging_ls=None,
                                                  char_src_attention_ls=None,
                                                  teacher_force_ls=None,
                                                  proportion_pred_train_ls=None, shared_context_ls=None,
                                                  word_embedding_projected_dim_ls=None,
                                                  char_level_embedding_projection_dim_ls=None, n_layers_sent_cell_ls=None,
                                                  n_layers_word_encoder_ls=None,
                                                  unrolling_word_ls=None, scoring_func=None, mode_word_encoding_ls=None,
                                                  dropout_input_ls=None, multi_task_loss_ponderation_ls=None,
                                                  scale_ls=[1])
                                # arguments that are specific to script generation


          print("row:{}".format(row))
          print("dir_script:{}".format(dir_script))


# WARNING : different behavior in warmup and test_before_run between DISTRIBUTED mode and SINGLE mode
## in SINGLE mode only warmup means one mode train and evaluated in the grid search for 1 epoch
    #  test_before models all models are run and evaluation
## in DISTRIBUTED : they are merged and corresponds to test_before_run

# oarsub -q gpu
# -l /core=2, walltime=48:00:00
# -p "host='gpu004'" -O ./logs/%jobid%-job.stdout -E ./logs/%jobid%-job.stderr ./train/train_mt_norm.sh
