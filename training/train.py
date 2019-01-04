from io_.dat import conllu_data
from model.seq2seq import LexNormalizer
from model.generator import  Generator
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
from env.project_variables import PROJECT_PATH, REPO_DATASET
import time
from toolbox.gpu_related import use_gpu_
from toolbox.sanity_check import get_timing
from collections import OrderedDict


def train(train_path, dev_path, n_epochs, normalization, dict_path =None,
          batch_size=10,
          label_train="", label_dev="",
          use_gpu=None,
          n_layers_word_encoder=1,
          dropout_sent_encoder=0, dropout_word_encoder=0, dropout_word_decoder=0,
          hidden_size_encoder=None, output_dim=None, char_embedding_dim=None,
          hidden_size_decoder=None, hidden_size_sent_encoder=None,
          checkpointing=True, freq_checkpointing=None, model_dir=None,
          reload=False, model_full_name=None, model_id_pref="", print_raw=False,
          model_specific_dictionary=False, dir_sent_encoder=1,
          add_start_char=None, add_end_char=1,
          debug=False,timing=False,
          verbose=1):

    use_gpu = use_gpu_(use_gpu)

    if use_gpu:
        printing("GPU was found use_gpu set to True ", verbose_level=0, verbose=verbose)
    else:
        printing("CPU mode ", verbose_level=0, verbose=verbose)
    freq_checkpointing = int(n_epochs/10) if checkpointing and freq_checkpointing is None else freq_checkpointing
    assert add_start_char == 1, "ERROR : add_start_char must be activated due decoding behavior of output_text_"
    printing("Warning : add_start_char is {} and add_end_char {}  ".format(add_start_char, add_end_char), verbose=verbose, verbose_level=0)

    if reload:
        assert model_full_name is not None and len(model_id_pref) == 0 and model_dir is not None and dict_path is not None

    else:
        assert model_full_name is None and model_dir is None

    if not debug:
        pdb.set_trace = lambda: 1
    loss_training = []
    loss_developing = []

    lr = 0.001

    printing("WARNING :  lr {} ".format(lr, add_start_char, add_end_char), verbose=verbose, verbose_level=0)
    printing("INFO : dictionary is computed (re)created from scratch on train_path {} and dev_path {}".format(train_path, dev_path), verbose=verbose, verbose_level=1)

    if not model_specific_dictionary:
        word_dictionary, char_dictionary, pos_dictionary, \
        xpos_dictionary, type_dictionary = \
        conllu_data.load_dict(dict_path=dict_path,
                              train_path=train_path,
                              dev_path=dev_path,
                              test_path=None,
                              word_embed_dict={},
                              dry_run=False,
                              vocab_trim=True,
                              force_new_dic=True,
                              add_start_char=add_start_char, verbose=1)

        voc_size = len(char_dictionary.instance2index)+1
        printing("char_dictionary", var=(char_dictionary.instance2index), verbose=verbose, verbose_level=0)
        printing("Character vocabulary is {} length", var=(len(char_dictionary.instance2index)+1), verbose=verbose,
                 verbose_level=0)
        _train_path, _dev_path, _add_start_char = None, None, None
    else:
        voc_size = None
        if not reload:
            # we need to feed the model the data so that it computes the model_specific_dictionary
            _train_path, _dev_path, _add_start_char = train_path, dev_path, add_start_char
        else:
            # as it reload : we don't need data
            _train_path, _dev_path, _add_start_char = None, None, None

    model = LexNormalizer(generator=Generator, load=reload,
                          char_embedding_dim=char_embedding_dim, voc_size=voc_size,
                          dir_model=model_dir, use_gpu=use_gpu,dict_path=dict_path,
                          train_path=_train_path, dev_path=_dev_path, add_start_char=_add_start_char,
                          model_specific_dictionary=model_specific_dictionary,
                          dropout_sent_encoder=dropout_sent_encoder, dropout_word_encoder=dropout_word_encoder,
                          dropout_word_decoder=dropout_word_decoder,
                          n_layers_word_encoder=n_layers_word_encoder,dir_sent_encoder=dir_sent_encoder,
                          hidden_size_encoder=hidden_size_encoder, output_dim=output_dim,
                          model_id_pref=model_id_pref, model_full_name=model_full_name,
                          hidden_size_sent_encoder=hidden_size_sent_encoder,
                          hidden_size_decoder=hidden_size_decoder, verbose=verbose, timing=timing)
    if use_gpu:
        model = model.cuda()
        printing("TYPE model is cuda : {} ", var=(next(model.parameters()).is_cuda), verbose=verbose, verbose_level=0)
    if not model_specific_dictionary:
        model.word_dictionary, model.char_dictionary, model.pos_dictionary, \
        model.xpos_dictionary, model.type_dictionary = word_dictionary, char_dictionary, pos_dictionary, \
                                                       xpos_dictionary, type_dictionary

    starting_epoch = model.arguments["info_checkpoint"]["n_epochs"] if reload else 0
    reloading = "" if not reload else " reloaded from "+str(starting_epoch)
    n_epochs += starting_epoch

    adam = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    #if use_gpu:
    #  printing("Setting adam to GPU", verbose=verbose, verbose_level=0)
    #  adam = adam.cuda()
    _loss_dev = 1000
    _loss_train = 1000

    printing("Running from {} to {} epochs : training on {} evaluating on {}", var=(starting_epoch, n_epochs, train_path, dev_path), verbose=verbose, verbose_level=0)
    starting_time = time.time()
    total_time = 0

    data_read_train = conllu_data.read_data_to_variable(train_path, model.word_dictionary, model.char_dictionary,
                                                         model.pos_dictionary,
                                                        model.xpos_dictionary, model.type_dictionary,
                                                        use_gpu=use_gpu, symbolic_root=False,
                                                        symbolic_end=False, dry_run=0, lattice=False, verbose=verbose,
                                                        normalization=normalization,
                                                        add_start_char=add_start_char, add_end_char=add_end_char)
    data_read_dev = conllu_data.read_data_to_variable(dev_path, model.word_dictionary, model.char_dictionary,
                                                       model.pos_dictionary,
                                                       model.xpos_dictionary, model.type_dictionary,
                                                       use_gpu=use_gpu, symbolic_root=False,
                                                       symbolic_end=False, dry_run=0, lattice=False, verbose=verbose,
                                                       normalization=normalization,
                                                       add_start_char=add_start_char, add_end_char=add_end_char)

    for epoch in tqdm(range(starting_epoch, n_epochs), disable_tqdm_level(verbose=verbose, verbose_level=0)):

        printing("Starting new epoch {} ", var=(epoch), verbose=verbose, verbose_level=1)
        model.train()
        batchIter = data_gen_conllu(data_read_train,
                                    model.word_dictionary, model.char_dictionary, model.pos_dictionary, model.xpos_dictionary, model.type_dictionary,
                                    add_start_char=add_start_char,
                                    add_end_char=add_end_char,
                                    normalization=normalization,
                                    use_gpu=use_gpu,
                                    batch_size=batch_size,
                                    print_raw=print_raw,timing=timing, 
                                    verbose=verbose)
        start = time.time()
        loss_train = run_epoch(batchIter, model, LossCompute(model.generator, opt=adam,
                                                             use_gpu=use_gpu,verbose=verbose, timing=timing),
                               verbose=verbose, i_epoch=epoch, n_epochs=n_epochs,timing=timing,
                               log_every_x_batch=100)
        _train_ep_time, start = get_timing(start)
        model.eval()
        batchIter_eval = data_gen_conllu(data_read_dev,
                                         model.word_dictionary, model.char_dictionary, model.pos_dictionary,
                                         model.xpos_dictionary, model.type_dictionary,
                                         batch_size=batch_size, add_start_char=add_start_char,
                                         add_end_char=add_end_char,use_gpu=use_gpu,
                                         normalization=normalization,
                                         verbose=verbose)
        printing("Starting evaluation ", verbose=verbose, verbose_level=1)
        _create_iter_time, start = get_timing(start)
        loss_dev = run_epoch(batchIter_eval, model, LossCompute(model.generator, use_gpu=use_gpu,verbose=verbose),
                             i_epoch=epoch, n_epochs=n_epochs,
                             verbose=verbose,timing=timing,
                             log_every_x_batch=100)
        _eval_time, start = get_timing(start)
        loss_training.append(loss_train)
        loss_developing.append(loss_dev)
        time_per_epoch = time.time() - starting_time
        total_time += time_per_epoch
        starting_time = time.time()
        # WARNING : only saving if we decrease not loading former model if we relaod
        if (checkpointing and epoch % freq_checkpointing == 0) or (epoch+1 == n_epochs):
            print("epochs ,", epoch, loss_training)
            dir_plot = simple_plot(final_loss=loss_train, loss_2=loss_developing, loss_ls=loss_training,
                                   epochs=str(epoch)+reloading,
                                   label=label_train+"-train",
                                   label_2=label_dev+"-dev",
                                   save=True, dir=model.dir_model,
                                   verbose=verbose, verbose_level=1,
                                   lr=lr, prefix=model.model_full_name,
                                   show=False)

            model, _loss_train = checkpoint(loss_former=_loss_dev, loss=loss_dev, model=model, model_dir=model.dir_model,
                                            info_checkpoint={"n_epochs": n_epochs, "batch_size": batch_size,
                                                           "train_data_path": train_path, "dev_data_path": dev_path,
                                                           "other": {"error_curves": dir_plot, 
                                                           "time_training(min)":"{0:.2f}".format(total_time/60), "average_per_epoch(min)":"{0:.2f}".format((total_time/n_epochs)/60)}},
                                            epoch=epoch, epochs=n_epochs,
                                            verbose=verbose)

        printing("LOSS train {:.3f}, dev {:.3f} for epoch {} out of {} epochs ", var=(loss_train, loss_dev,
                                                                                       epoch, n_epochs),
                 verbose=verbose,
                 verbose_level=1)

        if timing:
            print("Summaru : {}".format(OrderedDict([("_train_ep_time", _train_ep_time),("_create_iter_time", _create_iter_time), ("_eval_time",_eval_time) ])))

    #model.save(model_dir, model, info_checkpoint={"n_epochs": n_epochs, "batch_size": batch_size,
    #                                             "train_data_path": train_path, "dev_data_path": dev_path,
    #                                             "other": {"error_curves": dir_plot}})

    #report_model(parameters=True, ,arguments_dic=model.arguments, dir_models_repositories=REPOSITORIES)

    simple_plot(final_loss=loss_dev, loss_ls=loss_training, loss_2=loss_developing, epochs=n_epochs, save=True,
                dir=model.dir_model, label=label_train, label_2=label_dev,
                lr=lr, prefix=model.model_full_name+"-LAST")
   
    return model.model_full_name