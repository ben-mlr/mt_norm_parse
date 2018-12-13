import sys

#sys.path.insert(0,"/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/ELMoLex_sosweet/")
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
from training.train import train
dict_path = "../dictionaries/"
train_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/en-ud-train.conllu"
dev_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/owoputi.integrated"
test_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/lexnorm.integrated"
test_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/lexnorm.integrated"


if __name__ == "__main__":

    train(train_path, dev_path, n_epochs=2, normalization=False, batch_size=10, add_start_char=0,
          freq_checkpointing=2, reload=True, model_full_name="auto_encoder_TEST_93a3")

    if False:

        pdb.set_trace = lambda: 1
        checkpointing = True
        freq_checkpoint = 5
        loss_training = []
        loss_developing = []
        verbose = 1
        epochs = 21
        batch_size = 10
        print_raw = False
        nbatch = 60

        lr = 0.001
        add_start_char = 0
        add_end_char = 0

        word_dictionary, char_dictionary, pos_dictionary,\
        xpos_dictionary, type_dictionary = \
                conllu_data.create_dict(dict_path=dict_path,
                                        train_path=train_path,
                                        dev_path=dev_path,
                                        test_path=None,
                                        word_embed_dict={},
                                        dry_run=False,
                                        vocab_trim=True, add_start_char=add_start_char)

        printing("char_dictionary".format(char_dictionary.instance2index), verbose=verbose, verbose_level=0)
        V = len(char_dictionary.instance2index)+1
        printing("Character vocabulary is {} length".format(V), verbose=verbose, verbose_level=0)
        #model = LexNormalizer(generator=Generator, char_embedding_dim=20, voc_size=V, hidden_size_encoder=50, output_dim=50,
        #                      model_id_pref="auto_encoder_TEST",
        #                      hidden_size_decoder=50, verbose=verbose)

        starting_epoch = 0
        reload = True
        if reload:
            print("WARNING : reloading model ")
            _dir = os.path.dirname(os.path.realpath(__file__))
            model = LexNormalizer(generator=Generator, load=True, model_full_name="auto_encoder_TEST_93a3",#"6437",
                                  dir_model=os.path.join(_dir, "..", "checkpoints"),
                                  verbose=verbose)
            starting_epoch = model.arguments["info_checkpoint"]["n_epochs"]
            print(starting_epoch)
            epochs +=starting_epoch

        adam = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

        model_dir = os.path.join("../checkpoints","{}-folder".format(model.model_full_name))

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        else:
            assert reload, "{} dir already exists ".format(model)
            print("Writing into existing {} ".format(model_dir))
        printing("Dir {} created".format(model_dir), verbose=verbose, verbose_level=0)

        _loss_dev = 1000

        for epoch in tqdm(range(starting_epoch, epochs), disable_tqdm_level(verbose=verbose, verbose_level=0)):

            printing("Starting new epoch {} ".format(epoch), verbose=verbose, verbose_level=1)
            model.train()
            batchIter = data_gen_conllu(train_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                                        type_dictionary,
                                        add_start_char=add_start_char,
                                        add_end_char=add_end_char,
                                        normalization=False,
                                        batch_size=batch_size, print_raw=print_raw,
                                        nbatch=nbatch, verbose=verbose)

            loss_train = run_epoch(batchIter, model, LossCompute(model.generator, opt=adam, verbose=verbose),
                                   verbose=verbose, i_epoch=epoch, n_epochs=epochs,
                                   log_every_x_batch=100)
            model.eval()
            batchIter_eval = data_gen_conllu(dev_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                                             type_dictionary, batch_size=batch_size,add_start_char=add_start_char,
                                             add_end_char=add_end_char,
                                             normalization=False,
                                             nbatch=nbatch, verbose=verbose)
            printing("Starting evaluation ", verbose=verbose, verbose_level=1)
            loss_dev = run_epoch(batchIter_eval, model, LossCompute(model.generator, verbose=verbose),
                             i_epoch=epoch, n_epochs=epochs,
                             verbose=verbose,
                             log_every_x_batch=100)
            #except ZeroDivisionError as e:
            #    print("ERROR {} e for epoch {} ".format(e,epoch))

            loss_training.append(loss_train)
            loss_developing.append(loss_dev)

            # WARNING : only saving if we decrease not loading former model if we relaod
            if checkpointing and epoch%freq_checkpoint==0:
                model, _loss_dev = checkpoint(loss_former=_loss_dev, loss=loss_dev, model=model, model_dir=model_dir,
                                          info_checkpoint={"n_epochs": epochs, "batch_size": batch_size, "train_data_path": train_path, "dev_data_path": dev_path},
                                          epoch=epoch, epochs=epochs,verbose=verbose)

            printing("LOSS train {:.2f}, dev {:.2f} for epoch {} out of {} epochs ".format(loss_train, loss_dev, epoch, epochs), verbose=verbose,
                     verbose_level=1)

            dir_plot = simple_plot(final_loss=loss_train, loss_2=loss_developing,loss_ls=loss_training, epochs=epoch, save=True,
                        verbose=verbose, verbose_level=1,
                        lr=lr, prefix="INT-test-LARGER-overfit_conll_dummy", show=False)

        model.save(model_dir, model,info_checkpoint={"n_epochs": epochs, "batch_size": batch_size, "train_data_path": train_path, "dev_data_path": dev_path,
                   "other": {"error_curves": dir_plot}}, )

        batchIter_test = data_gen_conllu(test_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                                         type_dictionary, batch_size=batch_size,add_start_char=add_start_char,
                                         add_end_char=add_end_char,
                                         normalization=False,
                                         nbatch=nbatch, verbose=verbose)
        loss_test = run_epoch(batchIter_test, model, LossCompute(model.generator, verbose=verbose),
                              verbose=verbose,
                              log_every_x_batch=100)

        #report_model(parameters=True, ,arguments_dic=model.arguments, dir_models_repositories=REPOSITORIES)
        simple_plot(final_loss=loss_dev, loss_ls=loss_training, loss_2=loss_developing,epochs=epochs, save=True,
                    lr=lr, prefix=model.model_full_name+"-LAST-test-LARGER-overfit_conll_dummy")
