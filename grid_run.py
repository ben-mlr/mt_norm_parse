
from io_.info_print import printing
import os
from training.train_eval import train_eval
from toolbox.grid_tool import grid_param_label_generate
from env.project_variables import PROJECT_PATH, TRAINING,LIU_TRAIN, DEMO_SENT, CP_WR_PASTE_TEST_269, \
    LIU_DEV, DEV, DIR_TWEET_W2V, TEST, DIR_TWEET_W2V, CHECKPOINT_DIR, DEMO, DEMO2, CP_PASTE_WR_TRAIN, \
    CP_WR_PASTE_DEV, CP_WR_PASTE_TEST, CP_PASTE_DEV, CP_PASTE_TRAIN, CP_PASTE_TEST, EWT_DEV, EWT_TEST, \
    LIU_DEV_SENT, LIU_TRAIN_SENT, DEV_SENT, TEST_SENT, DEMO_SENT, TRAINING_DEMO, EN_LINES_EWT_TRAIN, EN_LINES_DEV, EN_LINES_EWT_TRAIN, \
    MTNT_TOK_TRAIN, MTNT_TOK_DEV, MTNT_EN_FR_TRAIN, MTNT_EN_FR_DEV, MTNT_EN_FR_TEST
from uuid import uuid4
import argparse
from sys import platform
from tracking.reporting_google_sheet import update_status, append_reporting_sheet

#4538

FINE_TUNE = 0
GRID = 1

if __name__ == "__main__":
      if GRID:
          ##train_path = LIU
          ##dev_path = DEV
          params = []

          ls_param = ["hidden_size_encoder", "hidden_size_sent_encoder","hidden_size_decoder", "output_dim", "char_embedding_dim"]
          param_0 = {"hidden_size_encoder": 5, "output_dim": 10, "char_embedding_dim": 10,
                     "dropout_sent_encoder": 0., "dropout_word_encoder": 0., "dropout_word_decoder": 0.,
                     "n_layers_word_encoder": 1, "dir_sent_encoder": 2,
                     "hidden_size_sent_encoder": 10, "hidden_size_decoder": 5, "batch_size": 10}

          params_baseline = {"hidden_size_encoder": 50, "output_dim": 100, "char_embedding_dim": 50,
                             "dropout_sent_encoder": 0., "drop_out_word_encoder": 0., "dropout_word_decoder": 0.,
                             "drop_out_sent_encoder_out": 1, "drop_out_word_encoder_out": 1, "dir_word_encoder": 1,
                             "n_layers_word_encoder": 1, "dir_sent_encoder": 1, "word_recurrent_cell_decoder": "LSTM", "word_recurrent_cell_encoder":"LSTM",
                             "hidden_size_sent_encoder": 50, "hidden_size_decoder": 50, "batch_size": 10}
          params_intuition = {"hidden_size_encoder": 40, "output_dim": 50, "char_embedding_dim": 30,
                              "dropout_sent_encoder": 0., "drop_out_word_encoder": 0., "dropout_word_decoder": 0.,
                              "drop_out_sent_encoder_out": 0, "drop_out_word_encoder_out": 0, "dir_word_encoder": 1,
                              "n_layers_word_encoder": 1, "dir_sent_encoder": 1, "word_recurrent_cell_decoder": "LSTM",
                              "word_recurrent_cell_encoder": "LSTM",
                              "hidden_size_sent_encoder": 15, "hidden_size_decoder": 40, "batch_size": 10
                             }
          params_strong = {"hidden_size_encoder": 100, "output_dim": 100, "char_embedding_dim": 50,
                             "dropout_sent_encoder": 0.3, "drop_out_word_encoder": 0.3, "dropout_word_decoder": 0.,
                             "drop_out_word_encoder_out": 0.3, "drop_out_sent_encoder_out": 0.3, "drop_out_char_embedding_decoder":0.3, "dropout_bridge":0.01,
                             "n_layers_word_encoder": 1, "dir_sent_encoder": 2,"word_recurrent_cell_decoder": "LSTM", "word_recurrent_cell_encoder":"LSTM",
                             "hidden_size_sent_encoder": 100, "hidden_size_decoder": 200, "batch_size": 10}

          params_strong_tryal = {"hidden_size_encoder": 20, "output_dim": 20, "char_embedding_dim": 40,
                                "dropout_sent_encoder": 0, "drop_out_word_encoder": 0, "dropout_word_decoder": 0.,
                                "drop_out_word_encoder_out": 0., "drop_out_sent_encoder_out": 0.,
                                "drop_out_char_embedding_decoder": 0., "dropout_bridge": 0.0,
                                "n_layers_word_encoder": 1, "dir_sent_encoder": 2, "word_recurrent_cell_decoder": "LSTM",
                                "word_recurrent_cell_encoder": "LSTM",
                                "hidden_size_sent_encoder": 24, "hidden_size_decoder": 30, "batch_size": 10}
          params_dozat = {"hidden_size_encoder": 200, "output_dim": 100, "char_embedding_dim": 100,
                            "dropout_sent_encoder": 0.5, "drop_out_word_encoder": 0.5, "dropout_word_decoder": 0.3,
                            "drop_out_word_encoder_out": 0.5, "drop_out_sent_encoder_out": 0.0,
                            "drop_out_char_embedding_decoder": 0.1, "dropout_bridge": 0.5,
                            "n_layers_word_encoder": 1, "dir_sent_encoder": 2, "word_recurrent_cell_decoder": "LSTM",
                            "word_recurrent_cell_encoder": "LSTM",
                            "hidden_size_sent_encoder": 200, "hidden_size_decoder": 100, "batch_size": 500}

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

          # default not used but could be
          params, labels, default_all, analysed, fixed = grid_param_label_generate(
                                                                                  params_dozat,
                                                                                  warmup=False,
                                                                                  grid_label="0",
                                                                                  stable_decoding_state_ls=[False],
                                                                                  word_decoding_ls=[False],
                                                                                  batch_size_ls=[50,100,200,400],
                                                                                  #auxilliary_task_pos_ls=[False],
                                                                                  word_embed_ls=[False],
                                                                                  dir_sent_encoder_ls=[2], lr_ls=[0],
                                                                                  word_embed_init_ls=[None],
                                                                                  teacher_force_ls=[True],
                                                                                  proportion_pred_train_ls=[None],
                                                                                  shared_context_ls=["all"],
                                                                                  word_embedding_projected_dim_ls=[None],
                                                                                  #auxilliary_task_norm_not_norm_ls=[True],
                                                                                  tasks_ls=[["pos"]],
                                                                                  char_src_attention_ls=[True],
                                                                                  n_layers_sent_cell_ls=[2],
                                                                                  unrolling_word_ls=[True],
                                                                                  scale_ls=[1]
                                                                                  )


          # grid information
          to_enrich = " ".join([a for a, _ in fixed])+" "+" ".join(analysed)
          to_analysed = " ".join(analysed)
          to_keep_only = " ".join([a+","+str(b) for a, b in fixed])
          metric_add = ""
          if "auxilliary_task_norm_not_norm " in to_analysed or "auxilliary_task_norm_not_norm,True" in to_keep_only :
              metric_add+=" precision-norm_not_norm accuracy-norm_not_norm recall-norm_not_norm"
          if "auxilliary_task_pos" in to_analysed  or "auxilliary_task_pos,True"  in to_keep_only:
              metric_add += " accuracy-pos"
          print("GRID_INFO metric    =  ", metric_add)
          print("GRID_INFO enrch vars=  ", to_enrich)
          print("GRID_INFO analy vars=  ", to_analysed)
          print("GRID_INFO fixed vals=   ", to_keep_only)
          # only for cloud run :

          warmup = True
          if platform != "darwin":
              printing("ENV : running not from os x assuming we are in command shell run", verbose=0, verbose_level=0)
              parser = argparse.ArgumentParser()
              parser.add_argument("--test_before_run", help="test_before_run", action="store_true")
              args = parser.parse_args()
              test_before_run = args.test_before_run
              print("GRID : test_before_run set to {} ".format(test_before_run))
              warmup = False
          else:
              test_before_run = False

          RUN_ID = str(uuid4())[0:5]
          LABEL_GRID = grid_label if not warmup else "WARMUP-unrolling-False"
          LABEL_GRID = "test_before_run-"+LABEL_GRID if test_before_run else LABEL_GRID

          OAR = os.environ.get('OAR_JOB_ID')+"_rioc-" if os.environ.get('OAR_JOB_ID', None) is not None else ""
          print("OAR=",OAR)
          OAR = RUN_ID if OAR == "" else OAR
          LABEL_GRID = OAR+"-"+LABEL_GRID

          GRID_FOLDER_NAME = LABEL_GRID if len(LABEL_GRID) > 0 else RUN_ID
          GRID_FOLDER_NAME += "-summary"
          dir_grid = os.path.join(CHECKPOINT_DIR, GRID_FOLDER_NAME)
          os.mkdir(dir_grid)
          printing("GRID RUN : Grid directory : dir_grid {} made".format(dir_grid), verbose=0, verbose_level=0)
          train_path, dev_path =EN_LINES_EWT_TRAIN, EWT_DEV # MTNT_EN_FR_TRAIN, MTNT_EN_FR_DEV #MTNT_TOK_TRAIN, MTNT_TOK_DEV#EN_LINES_EWT_TRAIN, EWT_DEV#CP_PASTE_WR_TRAIN, CP_WR_PASTE_DEV#TRAINING, EWT_DEV #LIU_TRAIN, LIU_DEV ## EWT_DEV, DEV
          i = 0

          row, col = append_reporting_sheet(git_id="0", rioc_job=LABEL_GRID, description="test",
                                            log_dir="log", target_dir="--",
                                            env="local", status="running", verbose=1)

          for param, model_id_pref in zip(params, labels):
              i += 1
              printing("GRID RUN : RUN_ID {} as prefix".format(RUN_ID), verbose=0, verbose_level=0)
              epochs = 50 if not test_before_run else 2
              if warmup:

                train_path, dev_path = DEMO, DEMO2
                param["word_embed_init"] = None

                if False:
                    param = {"hidden_size_encoder": 100, "output_dim": 15, "char_embedding_dim": 10,
                             "dropout_sent_encoder": 0.,
                             "drop_out_word_encoder": 0., "dropout_word_decoder": 0., "drop_out_sent_encoder_out": 0,
                             "drop_out_word_encoder_out": 0, "dir_word_encoder": 2, "n_layers_word_encoder": 1,
                             "dir_sent_encoder": 2,
                             "word_recurrent_cell_decoder": "LSTM", "word_recurrent_cell_encoder": "LSTM",
                             "hidden_size_sent_encoder": 20,
                             "hidden_size_decoder": 50, "batch_size": 2}
                    param["batch_size"] = 2
                    param["auxilliary_task_norm_not_norm"] = True
                    param["weight_binary_loss"] = 1
                    param["unrolling_word"] = True
                    param["char_src_attention"] = True
                    param["shared_context"] = "all"
                    param["dense_dim_auxilliary"] = None
                    param["gradient_clipping"] = None
                    param["dense_dim_auxilliary_2"] = None
                    param["stable_decoding_state"] = True
                    param["init_context_decoder"] = False
                    param["word_decoding"] = False
                    param["dense_dim_word_pred"] = None
                    param["tasks"] = ["pos"]
                    param["dense_dim_word_pred_2"] = None
                    param["dense_dim_word_pred_3"] = None

                    param["char_decoding"] = not param["word_decoding"]
                    param["auxilliary_task_pos"] = False
                    param["dense_dim_auxilliary_pos"] = 10
                    param["dense_dim_auxilliary_pos_2"] = 20
                    param["word_embed_init"] = None
                    param["word_embed"] = True
                    param["word_embedding_dim"] = 10
                    param["word_embedding_projected_dim"] = None
                    param["n_layers_sent_cell"] = 2
                    param["proportion_pred_train"] = 10
                    param["teacher_force"] = True

                    import torch.nn as nn
                    #param["activation_char_decoder"] = "nn.LeakyReLU"
                    #param["activation_word_decoder"] = "nn.LeakyReLU"
                    param["lr"] = 0.05

              model_id_pref = LABEL_GRID + model_id_pref + "-model_"+str(i)
              if warmup:
                  epochs = 1
                  print("GRID RUN : MODEL {} with param {} ".format(model_id_pref, param))
                  print("GRID_INFO analy vars=    dense_dim_auxilliary_pos_2 dense_dim_auxilliary_pos")
                  print("GRID_INFO fixed vars=  word_embed ")
                  print("GRID_INFO fixed vals=  word_embed,False ")

              model_full_name, model_dir = train_eval(train_path, dev_path, model_id_pref,
                                                      expand_vocab_dev_test=True,
                                                      test_path=[EWT_TEST,EWT_DEV, EN_LINES_EWT_TRAIN, TEST]\
                                                      #[TEST_SENT, MTNT_EN_FR_TEST, MTNT_EN_FR_DEV]\
                                                      if not warmup else DEMO,
                                                      overall_report_dir=dir_grid, overall_label=LABEL_GRID,
                                                      compute_mean_score_per_sent=True, print_raw=False,
                                                      get_batch_mode_all=True, compute_scoring_curve=False,
                                                      freq_scoring=10, bucketing_train=True, freq_checkpointing=1,
                                                      symbolic_root=True, symbolic_end=True,
                                                      freq_writer=1 if not test_before_run else 1,
                                                      extend_n_batch=2,
                                                      score_to_compute_ls=["exact", "norm_not_norm-F1",
                                                                           "norm_not_norm-Precision",
                                                                           "norm_not_norm-Recall",
                                                                           "norm_not_norm-accuracy"],
                                                      warmup=warmup, args=param, use_gpu=None, n_epochs=epochs,
                                                      debug=False,
                                                      verbose=1)

              run_dir = os.path.join(dir_grid, RUN_ID+"-run-log")
              open(run_dir, "a").write("model : done "+model_full_name+" in "+model_dir+" \n")
              print("GRID : Log RUN is : {} to see model list ".format(run_dir))
              print("GRID RUN : DONE MODEL {} with param {} ".format(model_id_pref, param))
              if warmup:
                  break

          update_status(row=row, new_status="done",verbose=1)

      elif FINE_TUNE:
        from training.fine_tune import fine_tune
        #train_path = LIU_TRAIN
        #dev_path = LIU_DEV
        #test_path = TEST#[TEST, CP_WR_PASTE_TEST_269]
        fine_tune_label = "fine_tuning"
        OAR = os.environ.get('OAR_JOB_ID')+"_rioc-" if os.environ.get('OAR_JOB_ID', None) is not None else ""
        print("OAR=",OAR)
        fine_tune_label = OAR+"-"+fine_tune_label
        fine_tune(train_path=LIU_TRAIN, dev_path=LIU_DEV, evaluation=True,batch_size=100,
                  test_path=[LIU_TRAIN, LIU_DEV, TEST, CP_WR_PASTE_TEST_269], n_epochs=30, fine_tune_label=fine_tune_label+"fine_tune_for_real-BACK_NORMALIZE-tenth",
                  model_full_name="99428_rioc--DEBUG_NO_LOSS_PADDING-0-model_1-model_1_8fb8",
                  learning_rate=0.00005, freeze_ls_param_prefix=["char_embedding","encoder","bridge"],
                  tasks=["normalize"],
                  debug=False, verbose=1)
        to_enrich = "lr  char_decoding char_src_attention "
        to_analysed = to_enrich
        to_keep_only = ""
        print("GRID_INFO enrch vars= batch_size lr ", to_enrich)
        print("GRID_INFO analy vars=  ", to_analysed)
        print("GRID_INFO fixed vals=  batch_size,2 lr,0.0001 ", to_keep_only)






#oarsub -q gpu -l /core=2,walltime=48:00:00  -p "host='gpu004'" -O ./logs/%jobid%-job.stdout -E ./logs/%jobid%-job.stderr ./train/train_mt_norm.sh 
