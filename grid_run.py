from training.train import train
from io_.info_print import printing
import os
from evaluate.evaluate_epoch import evaluate
import numpy as np
import torch
from toolbox.grid_tool import grid_param_label_generate

from env.project_variables import PROJECT_PATH, TRAINING, DEV, DIR_TWEET_W2V, \
    TEST, DIR_TWEET_W2V, CHECKPOINT_DIR, DEMO, DEMO2, REPO_DATASET, LIU, LEX_TRAIN, LEX_TEST, SEED_NP, SEED_TORCH, LEX_LIU_TRAIN, LIU_DEV, LIU_TRAIN, EWT_DEV
from uuid import uuid4
import argparse
from sys import platform
import time

np.random.seed(SEED_NP+1)
torch.manual_seed(SEED_TORCH)


def train_eval(train_path, dev_path, model_id_pref,pos_specific_path=None,
               n_epochs=11,test_path=None, args=None,
               overall_report_dir=CHECKPOINT_DIR, overall_label="DEFAULT",get_batch_mode_all=True,
               warmup=False, use_gpu=None, freq_checkpointing=1,debug=False,compute_scoring_curve=False,
               compute_mean_score_per_sent=False,print_raw=False,freq_scoring=5,bucketing_train=True,freq_writer=None,
               extend_n_batch=1, score_to_compute_ls=None,
               symbolic_end=False, symbolic_root=False,
               verbose=0):

    hidden_size_encoder = args.get("hidden_size_encoder", 10)
    word_embed = args.get("word_embed", False)
    word_embedding_projected_dim = args.get("word_embedding_projected_dim",None)
    word_embedding_dim  = args.get("word_embedding_dim", 0)
    output_dim = args.get("output_dim", 10)
    char_embedding_dim = args.get("char_embedding_dim", 10)
    hidden_size_sent_encoder = args.get("hidden_size_sent_encoder", 10)
    hidden_size_decoder = args.get("hidden_size_decoder", 10)
    batch_size = args.get("batch_size", 2)
    dropout_sent_encoder, dropout_word_encoder, dropout_word_decoder = args.get("dropout_sent_encoder",0), \
    args.get("dropout_word_encoder", 0), args.get("dropout_word_decoder", 0)
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

    auxilliary_task_norm_not_norm = args.get("auxilliary_task_norm_not_norm",False)
    char_src_attention = args.get("char_src_attention",False)
    weight_binary_loss = args.get("weight_binary_loss", 1)
    dir_word_encoder = args.get("dir_word_encoder", 1)
    shared_context = args.get("shared_context", "all")

    schedule_training_policy = args.get("policy", None)
    lr = args.get("lr", 0.001)
    gradient_clipping = args.get("gradient_clipping", None)

    teacher_force = args.get("teacher_force", True)

    stable_decoding_state = args.get("stable_decoding_state", False)
    init_context_decoder = args.get("init_context_decoder", True)

    word_decoding = args.get("word_decoding", False)
    dense_dim_word_pred = args.get("dense_dim_word_pred", None)
    dense_dim_word_pred_2 = args.get("dense_dim_word_pred_2", None)
    dense_dim_word_pred_3 = args.get("dense_dim_word_pred_3", 0)
    word_embed_init = args.get("word_embed_init", None)

    char_decoding = args.get("char_decoding", True)

    auxilliary_task_pos = args.get("auxilliary_task_pos", False)
    dense_dim_auxilliary_pos = args.get("dense_dim_auxilliary_pos", None)
    dense_dim_auxilliary_pos_2 = args.get("dense_dim_auxilliary_pos_2", None)

    n_epochs = 1 if warmup else n_epochs

    if warmup:
        printing("Warm up : running 1 epoch ", verbose=verbose, verbose_level=0)
    printing("GRID : START TRAINING ", verbose_level=0, verbose=verbose)
    model_full_name = train(train_path, dev_path, pos_specific_path=pos_specific_path,
                            auxilliary_task_norm_not_norm=auxilliary_task_norm_not_norm,
                            dense_dim_auxilliary=dense_dim_auxilliary, dense_dim_auxilliary_2=dense_dim_auxilliary_2,
                            lr=lr,extend_n_batch=extend_n_batch,
                            n_epochs=n_epochs, normalization=True,get_batch_mode_all=get_batch_mode_all,
                            batch_size=batch_size, model_specific_dictionary=True, freq_writer=freq_writer,
                            dict_path=None, model_dir=None, add_start_char=1, freq_scoring=freq_scoring,
                            add_end_char=1, use_gpu=use_gpu, dir_sent_encoder=dir_sent_encoder,
                            dropout_sent_encoder_cell=dropout_sent_encoder,
                            dropout_word_encoder_cell=dropout_word_encoder,
                            dropout_word_decoder_cell=dropout_word_decoder,
                            policy=schedule_training_policy,
                            dir_word_encoder=dir_word_encoder, compute_mean_score_per_sent=compute_mean_score_per_sent,
                            overall_label=overall_label, overall_report_dir=overall_report_dir,
                            label_train=REPO_DATASET[train_path], label_dev=REPO_DATASET[dev_path],
                            word_recurrent_cell_encoder=word_recurrent_cell_encoder, word_recurrent_cell_decoder=word_recurrent_cell_decoder,
                            drop_out_sent_encoder_out=drop_out_sent_encoder_out, drop_out_char_embedding_decoder=drop_out_char_embedding_decoder,
                            word_embedding_dim=word_embedding_dim, word_embed=word_embed, word_embedding_projected_dim=word_embedding_projected_dim,
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
                            teacher_force=teacher_force,
                            clipping=gradient_clipping,
                            auxilliary_task_pos=auxilliary_task_pos, dense_dim_auxilliary_pos=dense_dim_auxilliary_pos,
                            dense_dim_auxilliary_pos_2=dense_dim_auxilliary_pos_2,
                            word_decoding=word_decoding, dense_dim_word_pred=dense_dim_word_pred,
                            dense_dim_word_pred_2=dense_dim_word_pred_2, dense_dim_word_pred_3=dense_dim_word_pred_3,
                            char_decoding=char_decoding,
                            symbolic_end=symbolic_end, symbolic_root=symbolic_root,
                            stable_decoding_state=stable_decoding_state, init_context_decoder=init_context_decoder,
                            test_path=test_path[0] if isinstance(test_path,list) else test_path,
                            checkpointing=True, verbose=verbose)

    model_dir = os.path.join(CHECKPOINT_DIR, model_full_name+"-folder")
    if test_path is not None :
      dict_path = os.path.join(CHECKPOINT_DIR, model_full_name+"-folder", "dictionaries")
      printing("GRID : START EVALUATION FINAL ", verbose_level=0, verbose=verbose)
      eval_data_paths = [train_path, dev_path]
      if warmup:
          eval_data_paths = test_path if isinstance(test_path, list) else [test_path]
      else:
          if isinstance(test_path, list):
              eval_data_paths.extend(test_path)
          else:
              eval_data_paths.append(test_path)
      eval_data_paths = list(set(eval_data_paths))
      start_eval = time.time()
      for get_batch_mode_evaluate in [False]:
        for eval_data in eval_data_paths:
                eval_label = REPO_DATASET[eval_data]
                evaluate(model_full_name=model_full_name, data_path=eval_data,
                         dict_path=dict_path, use_gpu=use_gpu,
                         label_report=eval_label, overall_label=overall_label+"-last+bucket_True_eval-get_batch_"+str(get_batch_mode_evaluate),
                         score_to_compute_ls=score_to_compute_ls, mode_norm_ls=["all", "NEED_NORM", "NORMED"],
                         normalization=True, print_raw=print_raw,
                         model_specific_dictionary=True, get_batch_mode_evaluate=get_batch_mode_evaluate, bucket=True,
                         compute_mean_score_per_sent=compute_mean_score_per_sent,
                         batch_size=batch_size, debug=debug,
                         word_decoding=word_decoding, char_decoding=char_decoding,
                         dir_report=model_dir, verbose=1)
        print("GRID : END EVAL", time.time()-start_eval)

    return model_full_name, model_dir

#4538


if __name__ == "__main__":

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
                         "dropout_sent_encoder": 0, "drop_out_word_encoder": 0, "dropout_word_decoder": 0.,
                         "drop_out_word_encoder_out": 0.3, "drop_out_sent_encoder_out": 0.3, "drop_out_char_embedding_decoder":0.3, "dropout_bridge":0.01,
                         "n_layers_word_encoder": 1, "dir_sent_encoder": 2,"word_recurrent_cell_decoder": "LSTM", "word_recurrent_cell_encoder":"LSTM",
                         "hidden_size_sent_encoder": 100, "hidden_size_decoder": 150, "batch_size": 10}

      params_strong_tryal = {"hidden_size_encoder": 50, "output_dim": 100, "char_embedding_dim": 50,
                            "dropout_sent_encoder": 0, "drop_out_word_encoder": 0, "dropout_word_decoder": 0.,
                            "drop_out_word_encoder_out": 0.3, "drop_out_sent_encoder_out": 0.3,
                            "drop_out_char_embedding_decoder": 0.3, "dropout_bridge": 0.01,
                            "n_layers_word_encoder": 1, "dir_sent_encoder": 2, "word_recurrent_cell_decoder": "LSTM",
                            "word_recurrent_cell_encoder": "LSTM",
                            "hidden_size_sent_encoder": 100, "hidden_size_decoder": 150, "batch_size": 10}

      label_0 = "origin_small-batch10-LSTM_sent_bi_dir-word_uni_LSTM"

      ABLATION_NETWORK_SIZE = False

      if ABLATION_NETWORK_SIZE:
          params = [param_0]
          labels = [label_0]
          for level in [2,4,8,16]:
            for _, arg in enumerate(ls_param):
              param = {}
              param[arg] = param_0[arg]*2 if param_0[arg]*2<=100 else 100
            labels.append(str(level)+"-level-"+label_0[12:])
            params.append(param)
      i = 0

      ABLATION_DROPOUT = False

      if ABLATION_DROPOUT:
          params = []
          labels = [""]
          for add_dropout_encoder in [0, 0.1, 0.2, 0.5, 0.8]:
            for batch_size in [10, 20, 50, 100]:
              for dir_sent_encoder in [1, 2]:
                param = params_baseline.copy()
                param["drop_out_sent_encoder_out"] = 0.2#add_dropout_encoder
                param["drop_out_word_encoder_out"] = 0.2#add_dropout_encoder
                param["dropout_bridge"] = 0.2 #add_dropout_encoder
                param["drop_out_char_embedding_decoder"] = add_dropout_encoder
                param["dir_sent_encoder"] = dir_sent_encoder
                param["batch_size"] = batch_size

                label = str(add_dropout_encoder)+"-to_char_src-"+str(dir_sent_encoder)+"_dir_sent-"+str(batch_size)+"_batch_size"
                params.append(param)
                labels.append(label)

      ABLATION_DIR_WORD = False

      if ABLATION_DIR_WORD:
          params = [params_strong]
          labels = ["dir_word_encoder_1-strong-sent_source_dir_2-dropout_0.2_everywhere-LSTM-batch_10"]
          param = params_strong.copy()
          param["dir_word_encoder"] = 2
          param["hidden_size_encoder"] = int(param["hidden_size_encoder"]/2)
          params.append(param)
          labels.append("dir_word_encoder_2-sent_source_dir_2-dropout_0.2_everywhere-LSTM-batch_10")

      WITH_AUX = False
      if WITH_AUX:
          params = []
          labels = []
          n_model = 0
          for replicate in ["replicate1","replicate2"]:
            for dense_dim_auxilliary in [200, None]:
                dense_dim_auxilliary_2_ls = [0, 50] if dense_dim_auxilliary is not None else [0]
                for dense_dim_auxilliary_2 in dense_dim_auxilliary_2_ls:
                  policy_ls = [None] #if dense_dim_auxilliary is not None else [None]
                  auxilliary_task_norm_not_norm = False if dense_dim_auxilliary is None else True
                  for policy in policy_ls:
                  #for weight_binary_loss in [0.001,1,10,100]:
                    #auxilliary_task_norm_not_norm = False if dense_dim_auxilliary else True
                    for dir_word_encoder in [2, 1]:
                        for drop_out_char_embedding_decoder in [0.2]:
                            for char_src_attention in [False]:
                              for unrolling_word in [True]:
                                if char_src_attention == True and unrolling_word == False:
                                  continue
                                for word_embed in [False, True]:
                                    i += 1
                                    for scale in [1, 2]:
                                      param = params_strong.copy()
                                      param["hidden_size_encoder"] *= int(scale)
                                      param["hidden_size_sent_encoder"] *= int(scale)
                                      param["hidden_size_decoder"] *= int(scale)
                                      param["output_dim"] *= int(scale*0.5)+1
                                      param["drop_out_sent_encoder_out"] = 0.0#add_dropout_encoder
                                      param["drop_out_word_encoder_out"] = 0.0#add_dropout_encoder
                                      param["dropout_bridge"] = 0.1 #add_dropout_encoder
                                      param["drop_out_char_embedding_decoder"] = drop_out_char_embedding_decoder
                                      param["dir_word_encoder"] = dir_word_encoder
                                      param["dir_sent_encoder"] = 1
                                      param["batch_size"] = 40
                                      param["char_src_attention"] = char_src_attention
                                      param["dense_dim_auxilliary"] = dense_dim_auxilliary
                                      param["dense_dim_auxilliary_2"] = dense_dim_auxilliary_2
                                      param["unrolling_word"] = unrolling_word
                                      param["auxilliary_task_norm_not_norm"] = auxilliary_task_norm_not_norm
                                      param["shared_context"] = "all"
                                      #param["weight_binary_loss"] = weight_binary_loss
                                      param["policy"] = policy
                                      param["gradient_clipping"] = 1
                                      #label = str(dense_dim_auxilliary)+"-dense_dim_auxilliary"
                                      label = "REP_-"+replicate+"-"+str(dir_word_encoder)+"dir-scale_"+str(scale)#+str(dense_dim_auxilliary)+"_aux"
                                              #"dense_bin"+str(param["drop_out_word_encoder_out"])+"-do_char-" +\
                                              #str(param["dir_sent_encoder"])+"_dir_sent-"+str(param["batch_size"])+"_batch" + "-dir_word_src_" +\
                                              #str(param["dir_word_encoder"])+"-unrolling_word_"+str(unrolling_word)+
                                      params.append(param)
                                      #labels.append("model_"+str(n_model))
                                      labels.append(label)
      FROM_BEST = False
      if FROM_BEST:
          params = []
          labels = []
          n_model = 0
          for batch_size in [25]:
            for scale in [2]:
              for gradient_clipping in [1]:
                for dir_word_encoder in [2]:
                    for teacher_force in [False]:
                      for char_src_attention in [True]:
                        for auxilliary_task_norm_not_norm in [True,False]:
                            for shared_context in ["all","word"]:
                              if auxilliary_task_norm_not_norm:
                                dense_dim_auxilliary, dense_dim_auxilliary_2 = 200, 50
                              else:
                                dense_dim_auxilliary, dense_dim_auxilliary_2 = 0,0
                              for stable_decoding_state in [False]:
                                for word_decoding in [ True, False]:
                                  for auxilliary_task_pos in [False]:
                                    for word_embed in [True]:
                                      for lr in [0.001]:
                                          if shared_context == "sent":
                                            scale_sent_context = 1.5
                                            scale_word = 0.5
                                          else:
                                            scale_sent_context, scale_word = 1, 1
                                          for word_embed_init in [DIR_TWEET_W2V]:
                                              param = params_strong.copy()
                                              param["char_src_attention"] = char_src_attention
                                              param["hidden_size_encoder"] = int(param["hidden_size_encoder"]*scale*scale_word)
                                              param["hidden_size_sent_encoder"] = int(param["hidden_size_sent_encoder"]*scale*scale_sent_context)
                                              param["hidden_size_decoder"] = int(param["hidden_size_decoder"]*scale)
                                              param["output_dim"] *= int(scale*0.5)+1
                                              param["batch_size"] = batch_size
                                              param["unrolling_word"] = True
                                              param["auxilliary_task_norm_not_norm"] = auxilliary_task_norm_not_norm
                                              param["dense_dim_auxilliary"] = dense_dim_auxilliary
                                              param["dense_dim_auxilliary_2"] = dense_dim_auxilliary_2
                                              param["drop_out_char_embedding_decoder"] = 0.2
                                              param["dropout_bridge"] = 0.1
                                              param["dir_word_encoder"] = dir_word_encoder
                                              param["dir_sent_encoder"] = 1
                                              param["gradient_clipping"] = gradient_clipping
                                              param["teacher_force"] = teacher_force
                                              param["shared_context"] = shared_context
                                              param["stable_decoding_state"] = stable_decoding_state
                                              param["init_context_decoder"] = not param["stable_decoding_state"]

                                              param["word_decoding"] = word_decoding
                                              param["char_decoding"] = not param["word_decoding"]

                                              param["dense_dim_word_pred"] = 200 if word_decoding else None
                                              param["dense_dim_word_pred_2"] = 200 if word_decoding else None
                                              param["dense_dim_word_pred_3"] = 100 if word_decoding else None

                                              param["auxilliary_task_pos"] = auxilliary_task_pos
                                              param["dense_dim_auxilliary_pos"] = None if not auxilliary_task_pos else 200
                                              param["dense_dim_auxilliary_pos_2"] = None

                                              param["word_embed"] = word_embed
                                              param["word_embedding_dim"] = 400 if word_embed else 0

                                              param["lr"] = lr
                                              param["word_embed_init"] = word_embed_init

                                              params.append(param)
                                  #labels.append("word_char-level_contextxteacher_force-{}-stable_decod-init_con_{}-teachforce10_{}".format(shared_context,\
                                  #              param["sx@table_decoding_state"],param["init_context_decoder"],teacher_force))
                                              labels.append("externTrue-auxFalse-scale{}-sha_context_{}-auxnorm_not_norm_{}-word_de_{}".format(scale,shared_context,auxilliary_task_norm_not_norm, word_decoding))

          print("vword_embed_init shared_context word_decoding  n_trainable_parameters ")
          print("GRID_INFO fixed vals=  word_embed,True auxilliary_task_norm_not_norm,True auxilliary_task_pos,False lr,0.001  batch_size,25 stable_decoding_state,False teacher_force,True char_src_attention,True ")
          print("GRID_INFO fixed vars=  word_embed batch_size auxilliary_task_pos auxilliary_task_norm_not_norm stable_decoding_state teacher_force char_src_attention ")
      grid_label = "more_sent_layers"
      params,labels, default_all, analysed , fixed = grid_param_label_generate(
                                                                               params_strong_tryal, warmup=False,
                                                                               grid_label="0",
                                                                               word_decoding_ls=[True, False],
                                                                               auxilliary_task_pos_ls=[False],
                                                                               auxilliary_task_norm_not_norm_ls=[True],
                                                                               dir_sent_encoder_ls=[2], lr_ls=[0.001],
                                                                               word_embed_init_ls=[None],
                                                                               shared_context_ls=["all", "sent"],
                                                                               word_embedding_projected_dim_ls=[50],
                                                                               n_layers_sent_cell_ls=[1, 2],
                                                                               unrolling_word_ls=[True]
                                                                               )

      to_enrich = " ".join([a for a, _ in default_all])+" "+" ".join(analysed)
      to_keep_only = " ".join([a+","+str(b) for a, b in default_all])

      print("GRID_INFO analy vars=  ", to_enrich)
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
      train_path, dev_path = EWT_DEV, DEV #LIU_TRAIN, LIU_DEV ## EWT_DEV, DEV
      for param, model_id_pref in zip(params, labels):
          i += 1
          printing("GRID RUN : RUN_ID {} as prefix".format(RUN_ID), verbose=0, verbose_level=0)
          epochs = 50 if not test_before_run else 2
          if warmup:
            param = {
                     "hidden_size_encoder": 100, "output_dim": 15, "char_embedding_dim": 10, "dropout_sent_encoder": 0.,
                     "drop_out_word_encoder": 0., "dropout_word_decoder": 0., "drop_out_sent_encoder_out": 0,
                     "drop_out_word_encoder_out": 0, "dir_word_encoder": 2, "n_layers_word_encoder": 1, "dir_sent_encoder": 2,
                     "word_recurrent_cell_decoder": "LSTM", "word_recurrent_cell_encoder": "LSTM", "hidden_size_sent_encoder": 20,
                     "hidden_size_decoder": 50, "batch_size": 2
                     }
            param["batch_size"] = 20
            param["auxilliary_task_norm_not_norm"] = True
            param["weight_binary_loss"] = 1
            param["unrolling_word"] = True
            param["char_src_attention"] = False
            train_path, dev_path = DEMO, DEMO2
            param["shared_context"] = "all"
            param["dense_dim_auxilliary"] = None
            param["gradient_clipping"] = None
            param["teacher_force"] = True
            param["dense_dim_auxilliary_2"] = None
            param["stable_decoding_state"] = True
            param["init_context_decoder"] = False
            param["word_decoding"] = True
            param["dense_dim_word_pred"] = 100
            param["dense_dim_word_pred_2"] = 200
            param["dense_dim_word_pred_3"] = 500

            param["char_decoding"] = not param["word_decoding"]
            param["auxilliary_task_pos"] = False
            param["dense_dim_auxilliary_pos"] = 0
            param["dense_dim_auxilliary_pos_2"] = None
            param["word_embed_init"] = None
            param["word_embed"] = True
            param["word_embedding_dim"] = 400
            param["word_embedding_projected_dim"] = 50
            param["n_layers_sent_cell"] = 2
            param["lr"] = 0.05

          model_id_pref = LABEL_GRID + model_id_pref + "-model_"+str(i)
          if warmup:
              epochs = 1
              print("GRID RUN : MODEL {} with param {} ".format(model_id_pref, param))
              print("GRID_INFO analy vars=    dense_dim_auxilliary_pos_2 dense_dim_auxilliary_pos")
              print("GRID_INFO fixed vars=  word_embed ")
              print("GRID_INFO fixed vals=  word_embed,False ")

          model_full_name, model_dir = train_eval(train_path, dev_path, model_id_pref,
                                                  #pos_specific_path=DEV,
                                                  test_path=[TEST] if not warmup else DEMO,
                                                  verbose=1,
                                                  overall_report_dir=dir_grid, overall_label=LABEL_GRID,
                                                  compute_mean_score_per_sent=True, print_raw=False,
                                                  get_batch_mode_all=True, compute_scoring_curve=False,
                                                  freq_scoring=10, bucketing_train=True, freq_checkpointing=1,
                                                  symbolic_root=True, symbolic_end=True,
                                                  freq_writer=10 if not test_before_run else 1,
                                                  extend_n_batch=2,
                                                  score_to_compute_ls=["exact", "norm_not_norm-F1",
                                                                       "norm_not_norm-Precision",
                                                                       "norm_not_norm-Recall",
                                                                       "norm_not_norm-accuracy"],
                                                  warmup=False, args=param, use_gpu=None,
                                                  n_epochs=epochs,
                                                  debug=False)

          run_dir = os.path.join(dir_grid, RUN_ID+"-run-log")
          open(run_dir, "a").write("model : done "+model_full_name+" in "+model_dir+" \n")
          print("GRID : Log RUN is : {} to see model list ".format(run_dir))
          print("GRID RUN : DONE MODEL {} with param {} ".format(model_id_pref, param))
          if warmup:
            break

#oarsub -q gpu -l /core=2,walltime=48:00:00  -p "host='gpu004'" -O ./logs/%jobid%-job.stdout -E ./logs/%jobid%-job.stderr ./train/train_mt_norm.sh 
