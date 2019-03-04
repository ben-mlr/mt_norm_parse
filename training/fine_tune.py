from training.train import train
from io_.info_print import printing
import os
from evaluate.evaluate_epoch import evaluate
import numpy as np
import torch
from env.project_variables import PROJECT_PATH, TRAINING, DEV, DIR_TWEET_W2V, LIU_TRAIN, LIU_DEV, TEST, DIR_TWEET_W2V, CHECKPOINT_DIR, DEMO, DEMO2, CP_PASTE_WR_TRAIN,CP_WR_PASTE_DEV, CP_WR_PASTE_TEST, CP_PASTE_DEV, CP_PASTE_TRAIN, CP_PASTE_TEST
from toolbox.gpu_related import use_gpu_
from io_.info_print import printing
import os
import numpy as np
import torch
from env.project_variables import CHECKPOINT_DIR, REPO_DATASET, SEED_NP, SEED_TORCH
from evaluate.evaluate_epoch import evaluate
from model.generator import Generator
from model.seq2seq import LexNormalizer
from training.train import train
import pdb
from uuid import uuid4

np.random.seed(SEED_NP)
torch.manual_seed(SEED_TORCH)


def fine_tune(train_path, dev_path, test_path, n_epochs,  model_full_name, learning_rate,
              evaluation=False,
              freq_checkpointing=1, freq_writer=1, fine_tune_label="",batch_size=2,
              freeze_ls_param_prefix=None,
              debug=False, verbose=0):
    if not debug:
        pdb.set_trace = lambda: 1
    use_gpu = use_gpu_(None)
    hardware_choosen = "GPU" if use_gpu else "CPU"
    printing("{} mode ", var=([hardware_choosen]), verbose_level=0, verbose=verbose)
    dict_path = os.path.join(CHECKPOINT_DIR, model_full_name + "-folder", "dictionaries")
    model_dir = os.path.join(CHECKPOINT_DIR, model_full_name+"-folder")

    #batch_size = 50#model.arguments["info_checkpoint"]["batch_size"]
    word_decoding = False#model.arguments["hyperparameters"]["decoder_arch"]["word_decoding"]
    char_decoding = True#model.arguments["hyperparameters"]["decoder_arch"]["char_decoding"]
    #learning_rate = 0.00001#model.arguments["other"]["lr"]
    printing("LOADED Optimization arguments from last checkpoint are "
             " learning rate {} batch_size {} ", var=[learning_rate, batch_size], verbose_level=0, verbose=verbose)
    print("WARNING : char_decoding {} and word_decoding should not be loaded here ".format(char_decoding, word_decoding))

    warmup = False
    test_before_run=False
    RUN_ID = str(uuid4())[0:5]
    LABEL_GRID = fine_tune_label if not warmup else "WARMUP"
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

    train(train_path=train_path, dev_path=dev_path, test_path=test_path, n_epochs=n_epochs,
          batch_size=batch_size, get_batch_mode_all=True, bucketing=True,
          dict_path=dict_path, model_full_name=model_full_name, reload=True, model_dir=model_dir,
          symbolic_root=True, symbolic_end=True,
          overall_label=LABEL_GRID, overall_report_dir=dir_grid,
          model_specific_dictionary=True,  print_raw=False,
          compute_mean_score_per_sent=False,
          word_decoding=word_decoding, char_decoding=char_decoding,
          checkpointing=True, normalization=True,
          pos_specific_path=None, expand_vocab_dev_test=True,
          lr=learning_rate,
          extend_n_batch=2,#model.arguments["info_checkpoint"]["other"]["extend_n_batch"],
          freq_writer=freq_writer, freq_checkpointing=freq_checkpointing, #compute_mean_score_per_sent=True,
          score_to_compute_ls=["exact"], mode_norm_ls=["all", "NEED_NORM", "NORMED"], compute_scoring_curve=False,
          add_start_char=1, add_end_char=1,
          extra_arg_specific_label=fine_tune_label,
          freeze_ls_param_prefix=freeze_ls_param_prefix, freezing_mode=freeze_ls_param_prefix is not None,
          debug=False, use_gpu=None, verbose=verbose)

    if evaluation:
        if test_path is not None:
            dict_path = os.path.join(CHECKPOINT_DIR, model_full_name + "-folder", "dictionaries")
            printing("GRID : START EVALUATION FINAL ", verbose_level=0, verbose=verbose)
            eval_data_paths = []#[train_path, dev_path]
            if isinstance(test_path, list):
                eval_data_paths.extend(test_path)
            else:
                eval_data_paths.append(test_path)
            eval_data_paths = list(set(eval_data_paths))
            for eval_data in eval_data_paths:
                eval_label = REPO_DATASET[eval_data]
                evaluate(model_full_name=model_full_name, data_path=eval_data,
                         dict_path=dict_path, use_gpu=use_gpu,
                         label_report=eval_label,
                         overall_label=LABEL_GRID,
                         score_to_compute_ls=["exact"], mode_norm_ls=["all", "NEED_NORM", "NORMED"],
                         normalization=True, print_raw=False,
                         model_specific_dictionary=True, get_batch_mode_evaluate=False,
                         bucket=True,
                         compute_mean_score_per_sent=True,
                         batch_size=batch_size, debug=debug,
                         extra_arg_specific_label=fine_tune_label,
                         word_decoding=word_decoding, char_decoding=char_decoding,
                         dir_report=model_dir, verbose=1)

    # dropout_sent_encoder_cell = dropout_sent_encoder,
    # dropout_word_encoder_cell = dropout_word_encoder,
    # dropout_word_decoder_cell = dropout_word_decoder,
    # drop_out_sent_encoder_out = drop_out_sent_encoder_out
    # drop_out_char_embedding_decoder = drop_out_char_embedding_decoder,
    # policy = schedule_training_policy,
    # drop_out_word_encoder_out = drop_out_word_encoder_out, dropout_bridge = dropout_bridge,
    # weight_binary_loss = weight_binary_loss,
    # clipping = gradient_clipping,


if __name__ == "__main__":
    train_path = DEMO
    dev_path = DEMO2
    test_path = TEST
    fine_tune(train_path=train_path, dev_path=dev_path,
              test_path=test_path, n_epochs=11, fine_tune_label="fine_tune_ab",
              model_full_name="17bcb-WARMUP-unrolling-False0-model_1-model_1_780c",
           )


