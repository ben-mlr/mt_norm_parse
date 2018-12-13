from io_.dat import conllu_data
from model.seq2seq import LexNormalizer, Generator
from io_.data_iterator import data_gen_conllu
from training.epoch_train import run_epoch
from model.loss import LossCompute
from tracking.plot_loss import simple_plot
import torch
from tqdm import tqdm
import pdb
from toolbox.checkpointing import checkpoint
import os
from io_.info_print import disable_tqdm_level, printing

PROJECT_PATH = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/mt_norm_parse"
def train(train_path, dev_path, n_epochs, normalization, dict_path , batch_size=10,
          hidden_size_encoder=None, output_dim=None, char_embedding_dim=None, hidden_size_decoder=None,
          checkpointing=True, freq_checkpointing=None, model_dir=None,
          reload=False, model_full_name=None, model_id_pref="", add_start_char=1,
          debug=False,
          verbose=1):

    freq_checkpointing = int(n_epochs/10) if checkpointing and freq_checkpointing is None else freq_checkpointing
    printing("Warning : add_start_char is {} ".format(add_start_char), verbose=verbose, verbose_level=0)
    if reload:
        assert model_full_name is not None and len(model_id_pref) == 0 and model_dir is not None
    else:
        assert model_full_name is None and model_dir is None

    if not debug:
        pdb.set_trace = lambda: 1
    loss_training = []
    loss_developing = []

    print_raw = False
    nbatch = 60
    lr = 0.001
    add_end_char = 0

    printing("WARNING : n_batch {} lr {} and add_end_char {} are hardcoded ".format(nbatch, lr, add_end_char), verbose=verbose, verbose_level=0)

    printing("INFO : dictionary is computed (re)created from scratcch on train_path {} and dev_path {}".format(train_path, dev_path), verbose=verbose, verbose_level=1)
    word_dictionary, char_dictionary, pos_dictionary,\
    xpos_dictionary, type_dictionary = \
            conllu_data.create_dict(dict_path=dict_path,
                                    train_path=train_path,
                                    dev_path=dev_path,
                                    test_path=None,
                                    word_embed_dict={},
                                    dry_run=False,
                                    vocab_trim=True, add_start_char=add_start_char)

    voc_size = len(char_dictionary.instance2index)+1
    printing("char_dictionary".format(char_dictionary.instance2index), verbose=verbose, verbose_level=0)
    printing("Character vocabulary is {} length".format(len(char_dictionary.instance2index)+1), verbose=verbose, verbose_level=0)

    model = LexNormalizer(generator=Generator, load=reload,
                          char_embedding_dim=char_embedding_dim, voc_size=voc_size,
                          dir_model=model_dir,
                          hidden_size_encoder=hidden_size_encoder, output_dim=output_dim,
                          model_id_pref="auto_encoder_TEST", model_full_name=model_full_name,
                          hidden_size_decoder=hidden_size_decoder, verbose=verbose)

    if not reload:
        model_dir = os.path.join(PROJECT_PATH,"checkpoints", "{}-folder".format(model.model_full_name))
        os.mkdir(model_dir)
        printing("Dir {} created".format(model_dir), verbose=verbose, verbose_level=0)

    starting_epoch = model.arguments["info_checkpoint"]["n_epochs"] if reload else 0
    n_epochs += starting_epoch

    starting_epoch = 0

    adam = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    _loss_dev = 1000

    printing("STARTING running from {} to {} ".format(starting_epoch, n_epochs), verbose=verbose, verbose_level=0)
    for epoch in tqdm(range(starting_epoch, n_epochs), disable_tqdm_level(verbose=verbose, verbose_level=0)):

        printing("Starting new epoch {} ".format(epoch), verbose=verbose, verbose_level=1)
        model.train()
        batchIter = data_gen_conllu(train_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                                    type_dictionary,
                                    add_start_char=add_start_char,
                                    add_end_char=add_end_char,
                                    normalization=normalization,
                                    batch_size=batch_size, print_raw=print_raw,
                                    nbatch=nbatch, verbose=verbose)

        loss_train = run_epoch(batchIter, model, LossCompute(model.generator, opt=adam, verbose=verbose),
                               verbose=verbose, i_epoch=epoch, n_epochs=n_epochs,
                               log_every_x_batch=100)
        model.eval()
        batchIter_eval = data_gen_conllu(dev_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                                         type_dictionary, batch_size=batch_size,add_start_char=add_start_char,
                                         add_end_char=add_end_char,
                                         normalization=normalization,
                                         nbatch=nbatch, verbose=verbose)
        printing("Starting evaluation ", verbose=verbose, verbose_level=1)
        loss_dev = run_epoch(batchIter_eval, model, LossCompute(model.generator, verbose=verbose),
                         i_epoch=epoch, n_epochs=n_epochs,
                         verbose=verbose,
                         log_every_x_batch=100)

        loss_training.append(loss_train)
        loss_developing.append(loss_dev)

        # WARNING : only saving if we decrease not loading former model if we relaod
        if checkpointing and epoch%freq_checkpointing==0:
            model, _loss_dev = checkpoint(loss_former=_loss_dev, loss=loss_dev, model=model, model_dir=model_dir,
                                          info_checkpoint={"n_epochs": n_epochs, "batch_size": batch_size,
                                                           "train_data_path": train_path, "dev_data_path": dev_path},
                                          epoch=epoch, epochs=n_epochs,
                                          verbose=verbose)

        printing("LOSS train {:.3f}, dev {:.3f} for epoch {} out of {} epochs ".format(loss_train, loss_dev,
                                                                                       epoch, n_epochs),
                 verbose=verbose,
                 verbose_level=1)

        dir_plot = simple_plot(final_loss=loss_train, loss_2=loss_developing,loss_ls=loss_training, epochs=epoch, save=True,
                               verbose=verbose, verbose_level=1,
                               lr=lr, prefix="INT-test-LARGER-overfit_conll_dummy",
                               show=False)

    model.save(model_dir, model,info_checkpoint={"n_epochs": n_epochs, "batch_size": batch_size,
                                                 "train_data_path": train_path, "dev_data_path": dev_path,
                                                 "other": {"error_curves": dir_plot}})

    #report_model(parameters=True, ,arguments_dic=model.arguments, dir_models_repositories=REPOSITORIES)

    simple_plot(final_loss=loss_dev, loss_ls=loss_training, loss_2=loss_developing,epochs=n_epochs, save=True,
                lr=lr, prefix=model.model_full_name+"-LAST-test-LARGER-overfit_conll_dummy")
