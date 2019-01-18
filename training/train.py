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
from toolbox.checkpointing import checkpoint, update_curve_dic
import os
import numpy as np
from io_.info_print import disable_tqdm_level, printing
from env.project_variables import PROJECT_PATH, REPO_DATASET, SEED_TORCH, BREAKING_NO_DECREASE, CHECKPOINT_DIR
from env.project_variables import SEED_NP, SEED_TORCH
import time
from toolbox.gpu_related import use_gpu_
from toolbox.sanity_check import get_timing
from collections import OrderedDict
from tracking.plot_loss import simple_plot_ls
from evaluate.evaluate_epoch import evaluate

np.random.seed(SEED_NP)
torch.manual_seed(SEED_TORCH)

  
def train(train_path, dev_path, n_epochs, normalization, dict_path =None,
          batch_size=10,
          label_train="", label_dev="",
          use_gpu=None,
          n_layers_word_encoder=1, get_batch_mode_evaluate=True,
          dropout_sent_encoder_cell=0, dropout_word_encoder_cell=0, dropout_word_decoder_cell=0,
          dropout_bridge=0, drop_out_word_encoder_out=0, drop_out_sent_encoder_out=0,
          dir_word_encoder=1,
          word_recurrent_cell_encoder=None, word_recurrent_cell_decoder=None,drop_out_char_embedding_decoder=0,
          hidden_size_encoder=None, output_dim=None, char_embedding_dim=None,
          hidden_size_decoder=None, hidden_size_sent_encoder=None, freq_scoring=5,
          compute_scoring_curve=False, score_to_compute_ls=None, mode_norm_ls=None,
          checkpointing=True, freq_checkpointing=None, model_dir=None,
          reload=False, model_full_name=None, model_id_pref="", print_raw=False,
          model_specific_dictionary=False, dir_sent_encoder=1,
          add_start_char=None, add_end_char=1,
          overall_label="DEFAULT",overall_report_dir=CHECKPOINT_DIR,
          compute_mean_score_per_sent=False,
          auxilliary_task_norm_not_norm=False, weight_binary_loss=1,
          unrolling_word=False,
          debug=False,timing=False,
          verbose=1):
    
    if auxilliary_task_norm_not_norm:
        printing("MODEL : training model with auxillisary task (loss weighted with {})", var=[weight_binary_loss], verbose=verbose, verbose_level=1)
    if compute_scoring_curve:
        assert score_to_compute_ls is not None and mode_norm_ls is not None and freq_scoring is not None, \
            "ERROR score_to_compute_ls and mode_norm_ls should not be None"
    use_gpu = use_gpu_(use_gpu)
    hardware_choosen = "GPU" if use_gpu else "CPU"
    printing("{} mode ", var=([hardware_choosen]), verbose_level=0, verbose=verbose)
    freq_checkpointing = int(n_epochs/10) if checkpointing and freq_checkpointing is None else freq_checkpointing
    assert add_start_char == 1, "ERROR : add_start_char must be activated due decoding behavior of output_text_"
    printing("WARNING : add_start_char is {} and add_end_char {}  ".format(add_start_char, add_end_char), verbose=verbose, verbose_level=0)

    if reload:
        assert model_full_name is not None and len(model_id_pref) == 0 and model_dir is not None and dict_path is not None

    else:
        assert model_full_name is None and model_dir is None

    if not debug:
        pdb.set_trace = lambda: 1

    loss_training = []
    loss_developing = []
    lr = 0.001
    evaluation_set_reporting = list(set([train_path, dev_path]))
    curve_scores = {score + "-" + mode_norm+"-"+REPO_DATASET[data]: [] for score in score_to_compute_ls
                    for mode_norm in mode_norm_ls for data in evaluation_set_reporting} if compute_scoring_curve else None

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
        printing("DICTIONARY ; character vocabulary is len {} : {} ", var=str(len(char_dictionary.instance2index)+1,
                                                                              char_dictionary.instance2index),
         verbose=verbose, verbose_level=0)
        _train_path, _dev_path, _add_start_char = None, None, None
    else:
        voc_size = None
        if not reload:
            # we need to feed the model the data so that it computes the model_specific_dictionary
            _train_path, _dev_path, _add_start_char = train_path, dev_path, add_start_char
        else:
            # as it reload : we don't need data
            _train_path, _dev_path, _add_start_char = None, None, None

    model = LexNormalizer(generator=Generator,
                          auxilliary_task_norm_not_norm=auxilliary_task_norm_not_norm,
                          load=reload,
                          char_embedding_dim=char_embedding_dim, voc_size=voc_size,
                          dir_model=model_dir, use_gpu=use_gpu,dict_path=dict_path,
                          word_recurrent_cell_decoder=word_recurrent_cell_decoder, word_recurrent_cell_encoder=word_recurrent_cell_encoder,
                          train_path=_train_path, dev_path=_dev_path, add_start_char=_add_start_char,
                          model_specific_dictionary=model_specific_dictionary,
                          dir_word_encoder=dir_word_encoder,
                          drop_out_sent_encoder_cell=dropout_sent_encoder_cell, drop_out_word_encoder_cell=dropout_word_encoder_cell,
                          drop_out_word_decoder_cell=dropout_word_decoder_cell, drop_out_bridge=dropout_bridge,drop_out_char_embedding_decoder=drop_out_char_embedding_decoder,
                          drop_out_word_encoder_out=drop_out_word_encoder_out, drop_out_sent_encoder_out=drop_out_sent_encoder_out,
                          n_layers_word_encoder=n_layers_word_encoder,dir_sent_encoder=dir_sent_encoder,
                          hidden_size_encoder=hidden_size_encoder, output_dim=output_dim,
                          model_id_pref=model_id_pref, model_full_name=model_full_name,
                          hidden_size_sent_encoder=hidden_size_sent_encoder,
                          unrolling_word=unrolling_word,
                          hidden_size_decoder=hidden_size_decoder, verbose=verbose, timing=timing)
    if use_gpu:
        model = model.cuda()
        printing("TYPE model is cuda : {} ", var=(next(model.parameters()).is_cuda), verbose=verbose, verbose_level=4)
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
    counter_no_deacrease = 0
    saved_epoch = 1

    printing("GENERAL : Running from {} to {} epochs : training on {} evaluating on {}", var=(starting_epoch, n_epochs, train_path, dev_path), verbose=verbose, verbose_level=0)
    starting_time = time.time()
    total_time = 0
    x_axis_epochs = []

    data_read_train = conllu_data.read_data_to_variable(train_path, model.word_dictionary, model.char_dictionary,
                                                         model.pos_dictionary,
                                                        model.xpos_dictionary, model.type_dictionary,
                                                        use_gpu=use_gpu, symbolic_root=False,
                                                        norm_not_norm=auxilliary_task_norm_not_norm,
                                                        symbolic_end=False, dry_run=0, lattice=False, verbose=verbose,
                                                        normalization=normalization,
                                                        add_start_char=add_start_char, add_end_char=add_end_char)
    data_read_dev = conllu_data.read_data_to_variable(dev_path, model.word_dictionary, model.char_dictionary,
                                                      model.pos_dictionary,
                                                      model.xpos_dictionary, model.type_dictionary,
                                                      use_gpu=use_gpu, symbolic_root=False,
                                                      norm_not_norm=auxilliary_task_norm_not_norm,
                                                      symbolic_end=False, dry_run=0, lattice=False, verbose=verbose,
                                                      normalization=normalization,
                                                      add_start_char=add_start_char, add_end_char=add_end_char)

    for epoch in tqdm(range(starting_epoch, n_epochs), disable_tqdm_level(verbose=verbose, verbose_level=0)):

        printing("TRAINING : Starting {} epoch out of {} ", var=(epoch+1, n_epochs), verbose= verbose, verbose_level=1)

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
                                                             weight_binary_loss=weight_binary_loss,
                                                             auxilliary_task_norm_not_norm=auxilliary_task_norm_not_norm,
                                                             use_gpu=use_gpu,verbose=verbose, timing=timing),
                               verbose=verbose, i_epoch=epoch, n_epochs=n_epochs,timing=timing,
                               log_every_x_batch=100)
        _train_ep_time, start = get_timing(start)
        model.eval()
        # TODO : should be added in the freq_checkpointing orhterwise useless
        batchIter_eval = data_gen_conllu(data_read_dev,
                                         model.word_dictionary, model.char_dictionary, model.pos_dictionary,
                                         model.xpos_dictionary, model.type_dictionary,
                                         batch_size=batch_size, add_start_char=add_start_char,
                                         add_end_char=add_end_char,use_gpu=use_gpu,
                                         normalization=normalization,
                                         verbose=verbose)
        printing("EVALUATION : computing loss on dev ", verbose=verbose, verbose_level=1)
        _create_iter_time, start = get_timing(start)
        dev_report_loss = False
        if dev_report_loss:
            loss_dev = run_epoch(batchIter_eval, model, LossCompute(model.generator, use_gpu=use_gpu,verbose=verbose,
                                                                    weight_binary_loss=weight_binary_loss,
                                                                    auxilliary_task_norm_not_norm=auxilliary_task_norm_not_norm),
                             i_epoch=epoch, n_epochs=n_epochs,
                             verbose=verbose,timing=timing,
                             log_every_x_batch=100)
        else:
            loss_dev = 0
        _eval_time, start = get_timing(start)
        loss_training.append(loss_train)
        loss_developing.append(loss_dev)
        time_per_epoch = time.time() - starting_time
        total_time += time_per_epoch
        starting_time = time.time()

        # computing exact/edit score
        if compute_scoring_curve and ((epoch % freq_scoring == 0) or (epoch+1 == n_epochs)):
            if (epoch+1 == n_epochs):
              printing("EVALUATION : final scoring ", verbose, verbose_level=0)
            x_axis_epochs.append(epoch)
            printing("EVALUATION : Computing score on {} and {}  ", var=(score_to_compute_ls,mode_norm_ls), verbose=verbose, verbose_level=1)
            for eval_data in evaluation_set_reporting:
                eval_label = REPO_DATASET[eval_data]
                assert len(set(evaluation_set_reporting))==len(evaluation_set_reporting),\
                    "ERROR : twice the same dataset has been provided for reporting which will mess up the loss"
                printing("EVALUATION on {} ", var=[eval_data], verbose=verbose, verbose_level=1)
                scores = evaluate(data_path=eval_data,
                                  use_gpu=use_gpu,
                                  overall_label=overall_label,overall_report_dir=overall_report_dir,
                                  score_to_compute_ls=score_to_compute_ls, mode_norm_ls=mode_norm_ls,
                                  label_report=eval_label, model=model,
                                  normalization=True, print_raw=False,
                                  model_specific_dictionary=True,
                                  get_batch_mode_evaluate=get_batch_mode_evaluate,
                                  compute_mean_score_per_sent=compute_mean_score_per_sent,
                                  batch_size=batch_size,
                                  dir_report=model.dir_model,
                                  debug=debug,
                                  verbose=1)
                curve_scores = update_curve_dic(score_to_compute_ls=score_to_compute_ls, mode_norm_ls=mode_norm_ls,
                                                eval_data=eval_label,
                                                former_curve_scores=curve_scores, scores=scores)
                curve_ls_tuple = [(loss_ls, label) for label, loss_ls in curve_scores.items() if isinstance(loss_ls, list) ]
                curves = [tupl[0] for tupl in curve_ls_tuple]
                val_ls = [tupl[1]+"({}tok)".format(info_token) for tupl in curve_ls_tuple for data, info_token in curve_scores.items() if not isinstance(info_token, list) if tupl[1].endswith(data)]

            for score_plot in score_to_compute_ls:
                simple_plot_ls(losses_ls=curves, labels=val_ls, final_loss="", save=True, filter_by_label=score_plot,  x_axis=x_axis_epochs,
                               dir=model.dir_model,prefix=model.model_full_name,epochs=str(epoch)+reloading,verbose=verbose, lr=lr,
                               label_color_0=REPO_DATASET[evaluation_set_reporting[0]], label_color_1=REPO_DATASET[evaluation_set_reporting[1]])

        # WARNING : only saving if we decrease not loading former model if we relaod
        if (checkpointing and epoch % freq_checkpointing == 0) or (epoch+1 == n_epochs):
            dir_plot = simple_plot(final_loss=loss_train, loss_2=loss_developing, loss_ls=loss_training,
                                   epochs=str(epoch)+reloading,
                                   label=label_train+"-train",
                                   label_2=label_dev+"-dev",
                                   save=True, dir=model.dir_model,
                                   verbose=verbose, verbose_level=1,
                                   lr=lr, prefix=model.model_full_name,
                                   show=False)

            model, _loss_dev, counter_no_deacrease, saved_epoch = \
                    checkpoint(loss_saved =_loss_dev, loss=loss_dev, model=model,
                               counter_no_decrease=counter_no_deacrease, saved_epoch=saved_epoch,
                               model_dir= model.dir_model,
                               info_checkpoint={"n_epochs": epoch, "batch_size": batch_size,
                                                "train_data_path": train_path, "dev_data_path": dev_path,
                                                 "other": {"error_curves": dir_plot, "loss":_loss_dev,
                                                           "weight_binary_loss": weight_binary_loss,
                                                            "data":"dev","seed(np/torch)":(SEED_TORCH, SEED_TORCH),
                                                            "time_training(min)": "{0:.2f}".format(total_time/60),
                                                            "average_per_epoch(min)": "{0:.2f}".format((total_time/n_epochs)/60)}},
                             epoch=epoch, epochs=n_epochs,
                             verbose=verbose)
            if counter_no_deacrease >= BREAKING_NO_DECREASE:
                assert freq_checkpointing == 1, "ERROR : to implement"
                printing("CHECKPOINTING : Breaking training : loss did not decrease on dev for 10 checkpoints "
                         "so keeping model from {} epoch  ".format(saved_epoch),
                         verbose=verbose, verbose_level=0)
                break

        printing("LOSS train {:.3f}, dev {:.3f} for epoch {} out of {} epochs ", var=(loss_train, loss_dev,
                                                                                       epoch, n_epochs),
                 verbose=verbose,
                 verbose_level=1)

        if timing:
            print("Summary : {}".format(OrderedDict([("_train_ep_time", _train_ep_time), ("_create_iter_time", _create_iter_time), ("_eval_time",_eval_time) ])))

    #model.save(model_dir, model, info_checkpoint={"n_epochs": n_epochs, "batch_size": batch_size,
    #                                             "train_data_path": train_path, "dev_data_path": dev_path,
    #                                             "other": {"error_curves": dir_plot}})

    #report_model(parameters=True, ,arguments_dic=model.arguments, dir_models_repositories=REPOSITORIES)

    simple_plot(final_loss=loss_dev, loss_ls=loss_training, loss_2=loss_developing, epochs=n_epochs, save=True,
                dir=model.dir_model, label=label_train, label_2=label_dev,
                lr=lr, prefix=model.model_full_name+"-LAST")
   
    return model.model_full_name