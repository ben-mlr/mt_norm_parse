from training.train import train
import os
from env.env_variables import PROJECT_PATH

if __name__ == "__main__":

    # we assume the normpar project located ../parsing/
    train_path = os.path.join(PROJECT_PATH, "../parsing/normpar/data/en-ud-train.conllu")
    dev_path = os.path.join(PROJECT_PATH, "../parsing/normpar/data/owoputi.integrated")
    test_path = os.path.join(PROJECT_PATH, "../parsing/normpar/data/lexnorm.integrated")
    n_epochs = 300
    normalization = True
    batch_size = 2
    hidden_size_encoder = None
    output_dim = None
    char_embedding_dim = None
    hidden_size_decoder = None
    dict_path = "./dictionaries/"
    checkpointing = True
    freq_checkpointing = 10
    reload = False
    model_full_name = "auto_encoder_TEST_21ac"
    model_id_pref = ""
    add_start_char = 1
    add_end_char = 1

    model_dir = "./checkpoints"

    if reload:
        train(test_path, test_path, n_epochs=n_epochs, normalization=normalization, batch_size=batch_size,
              dict_path=dict_path, model_dir=model_dir,add_start_char=add_start_char, add_end_char=add_end_char,
              freq_checkpointing=freq_checkpointing, reload=reload,
              model_full_name=model_full_name)
    else:
        train(test_path, test_path, n_epochs=n_epochs, normalization=normalization, batch_size=batch_size,
              dict_path=dict_path, model_dir=None, add_start_char=add_start_char,add_end_char=add_end_char,
              freq_checkpointing=freq_checkpointing, reload=reload, model_id_pref="normalizer_lexnorm",
              hidden_size_encoder=35, output_dim=50, char_embedding_dim=20,
              hidden_size_decoder=40,
              )
