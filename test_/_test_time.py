# !/usr/local/bin/python
# -*- coding: utf-8 -*-

import os
import sys

# to be run from command line 
sys.path.insert(0,"/scratch/bemuller/mt_norm_parse")

from training.train import train
from env.project_variables import PROJECT_PATH, TRAINING, DEV, TEST, CHECKPOINT_DIR, DEMO, DEMO2, REPO_DATASET, LIU

train_path = LIU
test_path = LIU
use_gpu = False
batch_size = 2
n_epochs = 1
dropout_sent_encoder = 0
dropout_word_encoder = 0
dropout_word_decoder = 0
model_id_pref = "time"
hidden_size_encoder = 250
output_dim = 100
char_embedding_dim = 50
hidden_size_sent_encoder = 125
hidden_size_decoder = 300
n_layers_word_encoder = 2
dir_sent_src = 1
model_full_name = train(train_path, test_path, n_epochs=n_epochs, normalization=True,
                        batch_size=batch_size, model_specific_dictionary=True,
                        dict_path=None, model_dir=None, add_start_char=1,
                        add_end_char=1, use_gpu=use_gpu, verbose=1,
                        dropout_sent_encoder=dropout_sent_encoder, dropout_word_encoder=dropout_word_encoder,
                        dropout_word_decoder=dropout_word_decoder,
                        label_train=REPO_DATASET[train_path], label_dev=REPO_DATASET[test_path],
                        freq_checkpointing=10, reload=False, model_id_pref=model_id_pref,
                        hidden_size_encoder=hidden_size_encoder, output_dim=output_dim, char_embedding_dim=char_embedding_dim,
                        hidden_size_sent_encoder=hidden_size_sent_encoder, hidden_size_decoder=hidden_size_decoder,
                        n_layers_word_encoder=n_layers_word_encoder,
                        print_raw=True, debug=False, timing=False,
                        dir_sent_encoder=2,
                        checkpointing=True)
