from training.train import train
from io_.info_print import printing
import os
from evaluate.evaluate_epoch import evaluate
import numpy as np
import torch
from env.project_variables import CHECKPOINT_DIR, REPO_DATASET, SEED_NP, SEED_TORCH

import time

np.random.seed(SEED_NP+1)
torch.manual_seed(SEED_TORCH)


def train_eval(train_path, dev_path, model_id_pref,pos_specific_path=None,
               expand_vocab_dev_test=False,
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
                 "cause teacher_force mode",var=[proportion_pred_train], verbose=verbose, verbose_level=0)
        proportion_pred_train = None

    stable_decoding_state = args.get("stable_decoding_state", False)
    init_context_decoder = args.get("init_context_decoder", True)
    optimizer = args.get("optimizer", "bahdanu-adadelta")

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
    n_epochs = 1 if warmup else n_epochs

    if warmup:
        printing("Warm up : running 1 epoch ", verbose=verbose, verbose_level=0)
    printing("GRID : START TRAINING ", verbose_level=0, verbose=verbose)
    printing("SANITY CHECK : TASKS {} ", var=[tasks], verbose=verbose, verbose_level=1)
    normalization = "normalize" in tasks or "norm_not_norm" in tasks
    printing("SANITY CHECK : normalization {} ", var=normalization, verbose=verbose, verbose_level=1)
    model_full_name = train(train_path, dev_path, pos_specific_path=pos_specific_path,expand_vocab_dev_test=expand_vocab_dev_test,
                            #auxilliary_task_norm_not_norm=auxilliary_task_norm_not_norm,
                            dense_dim_auxilliary=dense_dim_auxilliary, dense_dim_auxilliary_2=dense_dim_auxilliary_2,
                            lr=lr,extend_n_batch=extend_n_batch,
                            n_epochs=n_epochs, normalization=normalization,get_batch_mode_all=get_batch_mode_all,
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
                            stable_decoding_state=stable_decoding_state, init_context_decoder=init_context_decoder,
                            test_path=test_path[0] if isinstance(test_path, list) else test_path,
                            checkpointing=True, verbose=verbose)

    model_dir = os.path.join(CHECKPOINT_DIR, model_full_name+"-folder")
    if test_path is not None:
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
                         normalization=normalization, print_raw=print_raw,
                         model_specific_dictionary=True, get_batch_mode_evaluate=get_batch_mode_evaluate, bucket=True,
                         compute_mean_score_per_sent=compute_mean_score_per_sent,
                         batch_size=batch_size, debug=debug,
                         word_decoding=word_decoding, char_decoding=char_decoding,
                         dir_report=model_dir, verbose=1)
        print("GRID : END EVAL", time.time()-start_eval)

    return model_full_name, model_dir