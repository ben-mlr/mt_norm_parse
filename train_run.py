from training.train import train
import os
from env.project_variables import PROJECT_PATH, TRAINING, DEV, TEST, CHECKPOINT_DIR, DEMO2

if __name__ == "__main__":

    # we assume the normpar project located ../parsing/
    train_path = TRAINING
    dev_path = DEV
    test_path = TEST
    n_epochs = 5

    # -> if not DEV_4:
    # output_seq.size()
    # torch.Size([2, 27])
    # conditioning.size()
    # torch.Size([1, 2, 40])
    # output_mask.size()
    # torch.Size([2, 27, 27])
    # output_word_len
    # output_word_len.size()
    # torch.Size([2, 1])
    ## if Dev 4 :
    # (Pdb) output.size()output_mask
    # torch.Size([48, 28])
    # (Pdb) output_mask.size()output_word_len
    # torch.Size([48, 28, 28])
    # (Pdb) output_word_len.size()
    # torch.Size([48])
    # (Pdb) conditioning.size()
    # torch.Size([1, 2, 40]) --> to transform in 1, 48, 40 : one same conditionning for each vector

    normalization = False
    batch_size = 2
    hidden_size_encoder = None
    output_dim = None
    char_embedding_dim = None
    hidden_size_decoder = None
    dict_path = "./dictionaries/"
    checkpointing = True
    freq_checkpointing = 5
    reload = True
    model_full_name = "auto_encoder_all_data_bddf"
    model_id_pref = ""
    add_start_char = 1
    add_end_char = 1

    model_dir = CHECKPOINT_DIR

    if reload:
        train(test_path, test_path, n_epochs=n_epochs, normalization=normalization, batch_size=batch_size,
              dict_path=dict_path, model_dir=model_dir,add_start_char=add_start_char, add_end_char=add_end_char,
              freq_checkpointing=freq_checkpointing, reload=reload,
              model_full_name=model_full_name)
    else:
        train(test_path, test_path, n_epochs=n_epochs, normalization=normalization, batch_size=batch_size,
              dict_path=dict_path, model_dir=None, add_start_char=add_start_char,add_end_char=add_end_char,
              freq_checkpointing=freq_checkpointing, reload=reload, model_id_pref="normalization_all_data",
              hidden_size_encoder=35, output_dim=50, char_embedding_dim=20, debug=False,
              hidden_size_decoder=40, print_raw=False, checkpointing=True
              )
