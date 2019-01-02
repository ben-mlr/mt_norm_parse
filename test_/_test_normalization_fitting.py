"""
only testing if the model runs for now

"""

import sys

#sys.path.insert(0,"/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/ELMoLex_sosweet/")
from io_.dat import conllu_data
from model.seq2seq import LexNormalizer
from model.generator import Generator
from io_.data_iterator import data_gen_conllu
from training.epoch_train import run_epoch
from model.loss import LossCompute
from tracking.plot_loss import simple_plot

import torch
import os
from tqdm import tqdm
import pdb
from io_.info_print import disable_tqdm_level, printing
from env.project_variables import PROJECT_PATH, TRAINING, DEV, TEST, CHECKPOINT_DIR, DEMO, DEMO2, REPO_DATASET

dict_path = os.path.join(PROJECT_PATH, "dictionaries")
train_path = TRAINING
dev_pat = DEV
test_path = TEST


if __name__ == "__main__":

    pdb.set_trace = lambda: 1
    normalization = True
    loss_training = []
    verbose = 1
    epochs = 60
    batch_size = 20
    print_raw = False
    checkpointing = True
    nbatch = 20
    lr = 0.001
    add_start_char = 1
    add_end_char = 0
    word_dictionary, char_dictionary, pos_dictionary,\
    xpos_dictionary, type_dictionary = \
            conllu_data.create_dict(dict_path=dict_path,
                                    train_path=test_path,
                                    dev_path=test_path,
                                    test_path=None,
                                    word_embed_dict={},
                                    dry_run=False,
                                    vocab_trim=True, add_start_char=add_start_char,
                                    )

    printing("char_dictionary".format(char_dictionary.instance2index), verbose=verbose, verbose_level=0)
    V = len(char_dictionary.instance2index)+1
    printing("Character vocabulary is {} length".format(V), verbose=verbose, verbose_level=0)
    model = LexNormalizer(generator=Generator, char_embedding_dim=20, voc_size=V, hidden_size_encoder=50, output_dim=50,
                          hidden_size_sent_encoder=10,
                          hidden_size_decoder=50, verbose=verbose)
    adam = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    loss_former = 1000
    checkpointing_freq = 10

    if checkpointing and False:
        model_dir = os.path.join("../checkpoints","{}-folder".format(model.model_full_name))
        os.mkdir(model_dir)
        printing("Dir {} created".format(model_dir), verbose=verbose, verbose_level=0)

    for epoch in tqdm(range(epochs), disable_tqdm_level(verbose=verbose, verbose_level=0)):

        printing("Starting new epoch {} ".format(epoch), verbose=verbose, verbose_level=1)
        model.train()
        batchIter = data_gen_conllu(test_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                                    type_dictionary,
                                    add_start_char=add_start_char,
                                    add_end_char=add_end_char,
                                    batch_size=batch_size, print_raw=print_raw,
                                    normalization=normalization,
                                    verbose=verbose)

        run_epoch(batchIter, model, LossCompute(model.generator, opt=adam, verbose=verbose), verbose=verbose, i_epoch=epoch, n_epochs=epochs,
                  log_every_x_batch=100)

        model.eval()
        batchIter_eval = data_gen_conllu(test_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                                         type_dictionary, batch_size=batch_size,add_start_char=add_start_char, add_end_char=add_end_char,
                                         normalization = normalization,
                                         nbatch=nbatch, verbose=verbose)
        printing("Starting evaluation ", verbose=verbose, verbose_level=1)
        loss = run_epoch(batchIter_eval, model, LossCompute(model.generator, verbose=verbose),
                         i_epoch=epoch, n_epochs=epochs,
                         verbose=verbose,
                         log_every_x_batch=100)
        #except ZeroDivisionError as e:
        #    print("ERROR {} e for epoch {} ".format(e,epoch))
        loss_training.append(loss)

        printing("Final Loss epoch {} ".format(loss), verbose=verbose, verbose_level=1)

        simple_plot(final_loss=loss, loss_ls=loss_training, epochs=epoch, save=True,
                    verbose=verbose, verbose_level=1,
                    lr=lr, prefix="INT-test-LARGER-normalization_test", show=False)


        if loss < loss_former and checkpointing and epoch%checkpointing_freq == 0 :
            model.save(model_dir, model, suffix_name="{}ep-outof{}ep".format(epoch, epochs), verbose=verbose)
            loss_former = loss
        elif checkpointing:
            # TODO : load former checkpoint : and do change loss append
            #
            pass



    model.save("./test_models", model)
    simple_plot(final_loss=loss, loss_ls=loss_training, epochs=epochs, save=True,
                lr=lr, prefix=model.model_full_name+"-LAST-test-LARGER-overfit_conll_dummy")
