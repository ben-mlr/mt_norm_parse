
from io_.info_print import printing
import os
from training.train_eval import train_eval
from toolbox.grid_tool import grid_param_label_generate
from env.project_variables import PROJECT_PATH, TRAINING,LIU_TRAIN, LIU_DEV, DEV, DIR_TWEET_W2V, TEST, DIR_TWEET_W2V, CHECKPOINT_DIR, DEMO, DEMO2, CP_PASTE_WR_TRAIN,CP_WR_PASTE_DEV, CP_WR_PASTE_TEST, CP_PASTE_DEV, CP_PASTE_TRAIN, CP_PASTE_TEST
from uuid import uuid4
import argparse
from sys import platform

#4538

FINE_TUNE = False
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
                                "hidden_size_sent_encoder": 24, "hidden_size_decoder": 100, "batch_size": 10}


          grid_label = "DEBUG_NO_LOSS_PADDING-"#"POS-2LSMT-2dense+no_aux_task-sent_only-EWT_DEV-PONDERATION-1pos-0_norm"
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
          params,labels, default_all, analysed, fixed = grid_param_label_generate(
                                                                                  params_strong,
                                                                                  warmup=False,
                                                                                  grid_label="0",
                                                                                  stable_decoding_state_ls=[False],
                                                                                  word_decoding_ls=[False],
                                                                                  batch_size_ls=[100],
                                                                                  auxilliary_task_pos_ls=[False],
                                                                                  word_embed_ls=[False],
                                                                                  dir_sent_encoder_ls=[2], lr_ls=[0.001],
                                                                                  word_embed_init_ls=[None],
                                                                                  teacher_force_ls=[True],
                                                                                  proportion_pred_train_ls=[None],
                                                                                  shared_context_ls=["all"],
                                                                                  word_embedding_projected_dim_ls=[100],
                                                                                  auxilliary_task_norm_not_norm_ls=[False],
                                                                                  char_src_attention_ls=[True],
                                                                                  n_layers_sent_cell_ls=[1],
                                                                                  unrolling_word_ls=[True],
                                                                                  scale_ls=[1]
                                                                                  )


          # grid information
          to_enrich = " ".join([a for a, _ in default_all])+" "+" ".join(analysed)
          to_analysed = " ".join(analysed)
          to_keep_only = " ".join([a+","+str(b) for a, b in default_all])
          metric_add = ""
          if "auxilliary_task_norm_not_norm " in default_all:
              metric_add+=" precision-norm_not_norm accuracy-norm_not_norm recall-norm_not_norm"
          if "auxilliary_task_pos" in default_all:
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
          train_path, dev_path = CP_PASTE_WR_TRAIN, CP_WR_PASTE_DEV #LIU_TRAIN, LIU_DEV ## EWT_DEV, DEV
          i = 0
          for param, model_id_pref in zip(params, labels):
              i += 1
              printing("GRID RUN : RUN_ID {} as prefix".format(RUN_ID), verbose=0, verbose_level=0)
              epochs = 5 if not test_before_run else 2
              if warmup:
                param = {
                         "hidden_size_encoder": 100, "output_dim": 15, "char_embedding_dim": 10, "dropout_sent_encoder": 0.,
                         "drop_out_word_encoder": 0., "dropout_word_decoder": 0., "drop_out_sent_encoder_out": 0,
                         "drop_out_word_encoder_out": 0, "dir_word_encoder": 2, "n_layers_word_encoder": 1, "dir_sent_encoder": 2,
                         "word_recurrent_cell_decoder": "LSTM", "word_recurrent_cell_encoder": "LSTM", "hidden_size_sent_encoder": 20,
                         "hidden_size_decoder": 50, "batch_size": 2
                         }
                param["batch_size"] = 20
                param["auxilliary_task_norm_not_norm"] = False
                param["weight_binary_loss"] = 1
                param["unrolling_word"] = True
                param["char_src_attention"] = True
                train_path, dev_path = DEMO, DEMO2
                param["shared_context"] = "all"
                param["dense_dim_auxilliary"] = None
                param["gradient_clipping"] = None
                param["dense_dim_auxilliary_2"] = None
                param["stable_decoding_state"] = True
                param["init_context_decoder"] = False
                param["word_decoding"] = False
                param["dense_dim_word_pred"] = None
                param["dense_dim_word_pred_2"] = None
                param["dense_dim_word_pred_3"] = None

                param["char_decoding"] = not param["word_decoding"]
                param["auxilliary_task_pos"] = False
                param["dense_dim_auxilliary_pos"] = 0
                param["dense_dim_auxilliary_pos_2"] = None
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
                  epochs = 5
                  print("GRID RUN : MODEL {} with param {} ".format(model_id_pref, param))
                  print("GRID_INFO analy vars=    dense_dim_auxilliary_pos_2 dense_dim_auxilliary_pos")
                  print("GRID_INFO fixed vars=  word_embed ")
                  print("GRID_INFO fixed vals=  word_embed,False ")

              model_full_name, model_dir = train_eval(train_path, dev_path, model_id_pref,
                                                      expand_vocab_dev_test=True,
                                                      test_path=[CP_WR_PASTE_TEST] if not warmup else DEMO,
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
                                                      warmup=False, args=param, use_gpu=None, n_epochs=epochs, debug=False,
                                                      verbose=1)

              run_dir = os.path.join(dir_grid, RUN_ID+"-run-log")
              open(run_dir, "a").write("model : done "+model_full_name+" in "+model_dir+" \n")
              print("GRID : Log RUN is : {} to see model list ".format(run_dir))
              print("GRID RUN : DONE MODEL {} with param {} ".format(model_id_pref, param))
              if warmup:
                  break

      elif FINE_TUNE:
        from training.fine_tune import fine_tune
        train_path = LIU_TRAIN
        dev_path = LIU_DEV
        test_path = [DEMO, DEMO2]
        fine_tune(train_path=train_path, dev_path=dev_path, evaluation=True,
                  test_path=test_path, n_epochs=1,
                  model_full_name="98752_rioc--DEBUG_NO_LOSS_PADDING-0-model_1-model_1_25ea",
                  word_decoding=False, char_decoding=True)






#oarsub -q gpu -l /core=2,walltime=48:00:00  -p "host='gpu004'" -O ./logs/%jobid%-job.stdout -E ./logs/%jobid%-job.stderr ./train/train_mt_norm.sh 
