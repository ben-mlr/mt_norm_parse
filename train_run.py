from training.train import train
import os
import sys
from env.project_variables import PROJECT_PATH, TRAINING, DEV, TEST, CHECKPOINT_DIR, DEMO, DEMO2, REPO_DATASET

if __name__ == "__main__":

    # we assume the normpar project located ../parsing/
    train_path = TRAINING
    dev_path = DEMO2
    test_path = DEMO2
    n_epochs = 10

    normalization = True
    batch_size = 2
    hidden_size_encoder = None
    output_dim = None
    char_embedding_dim = None
    hidden_size_decoder = None
    checkpointing = True
    freq_checkpointing = 10
    reload = True
    model_full_name = "test_dbc4"
    model_id_pref = ""
    add_start_char = 1
    add_end_char = 1
    dict_path = "./dictionaries"

    model_dir = os.path.join(CHECKPOINT_DIR, model_full_name+"-folder")

    if reload:
        dict_path = os.path.join(model_dir, "dictionaries")
        train(test_path, test_path, n_epochs=n_epochs, normalization=normalization, batch_size=batch_size,
              dict_path=dict_path, model_dir=model_dir, add_start_char=add_start_char, add_end_char=add_end_char,
              freq_checkpointing=freq_checkpointing, reload=reload, model_specific_dictionary=True, debug=False,
              model_full_name=model_full_name,
              )
    else:
        train(dev_path, test_path, n_epochs=n_epochs, normalization=normalization,
              batch_size=batch_size,
              model_specific_dictionary=True,
              dict_path=dict_path, model_dir=None, add_start_char=add_start_char,
              add_end_char=add_end_char, use_gpu=None,
              label_train=REPO_DATASET[test_path], label_dev=REPO_DATASET[test_path],
              freq_checkpointing=freq_checkpointing, reload=reload, model_id_pref="test",#"compare_normalization_all",
              hidden_size_encoder=250, output_dim=300, char_embedding_dim=300, debug=False,
              hidden_size_sent_encoder=250,
              hidden_size_decoder=300, print_raw=False, checkpointing=True
              )
# TODO : add DEV_5 to decode_sequence : 3d shapes in the decoded sequence also
# make it run in Van der Goot settings