from io_.dat import conllu_data
from model.seq2seq import LexNormalizer
from model.generator import Generator

from io_.data_iterator import data_gen_conllu, readers_load, data_gen_multi_task_sampling_batch
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
from env.project_variables import PROJECT_PATH, REPO_DATASET, SEED_TORCH, BREAKING_NO_DECREASE, CHECKPOINT_DIR, LOSS_DETAIL_TEMPLATE_LS, AVAILABLE_OPTIMIZER
from env.project_variables import SEED_NP, SEED_TORCH
import time
from toolbox.gpu_related import use_gpu_
from toolbox.sanity_check import get_timing
from collections import OrderedDict
from tracking.plot_loss import simple_plot_ls
from evaluate.evaluate_epoch import evaluate
from model.schedule_training_policy import AVAILABLE_SCHEDULING_POLICIES
from toolbox.norm_not_norm import scheduling_policy
from toolbox.tensorboard_tools import writer_weights_and_grad
from tensorboardX import SummaryWriter
from env.project_variables import AVAILABLE_TASKS
from model.schedule_training_policy import policy_1, policy_2
np.random.seed(SEED_NP)
torch.manual_seed(SEED_TORCH)
ADAPTABLE_SCORING = True


def train(train_path, dev_path, n_epochs, normalization, dict_path=None, pos_specific_path=None,
          expand_vocab_dev_test=False,
          batch_size=10, test_path=None,
          label_train="", label_dev="",
          use_gpu=None,
          lr=0.001,
          n_layers_word_encoder=1, n_layers_sent_cell=1,
          get_batch_mode_all=True,
          dropout_sent_encoder_cell=0, dropout_word_encoder_cell=0, dropout_word_decoder_cell=0,
          dropout_bridge=0, drop_out_word_encoder_out=0, drop_out_sent_encoder_out=0,
          dir_word_encoder=1,
          word_embed=False, word_embedding_dim=None, word_embedding_projected_dim=None,
          mode_word_encoding="cat", char_level_embedding_projection_dim=0,
          word_recurrent_cell_encoder=None, word_recurrent_cell_decoder=None,drop_out_char_embedding_decoder=0,
          hidden_size_encoder=None, output_dim=None, char_embedding_dim=None,
          hidden_size_decoder=None, hidden_size_sent_encoder=None, freq_scoring=5,
          compute_scoring_curve=False, score_to_compute_ls=None, mode_norm_ls=None,
          checkpointing=True, freq_checkpointing=None,freq_writer=None,
          model_dir=None,
          reload=False, model_full_name=None, model_id_pref="", print_raw=False,
          model_specific_dictionary=False, dir_sent_encoder=1,
          add_start_char=None, add_end_char=1,
          overall_label="DEFAULT",overall_report_dir=CHECKPOINT_DIR,
          compute_mean_score_per_sent=False,
          auxilliary_task_norm_not_norm=False, weight_binary_loss=1,
          dense_dim_auxilliary=None, dense_dim_auxilliary_2=None,
          unrolling_word=False, char_src_attention=False,
          debug=False, timing=False, dev_report_loss=True,
          bucketing=True, policy=None, teacher_force=True, proportion_pred_train=None,
          shared_context="all", clipping=None, extend_n_batch=1,
          stable_decoding_state=False, init_context_decoder=True,
          auxilliary_task_pos=False, dense_dim_auxilliary_pos=None, dense_dim_auxilliary_pos_2=None,
          tasks=None,
          word_decoding=False, char_decoding=True,
          dense_dim_word_pred=None, dense_dim_word_pred_2=None,dense_dim_word_pred_3=None,
          symbolic_root=False, symbolic_end=False, extern_emb_dir=None,
          activation_word_decoder=None, activation_char_decoder=None,
          extra_arg_specific_label="",
          freezing_mode=False, freeze_ls_param_prefix=None,

          attention_tagging=False,
          optimizer="adam",
          verbose=1):

    assert optimizer in AVAILABLE_OPTIMIZER, "optimizer supported are {} ".format(AVAILABLE_OPTIMIZER)
    if teacher_force:
        assert proportion_pred_train is None, "proportion_pred_train should be None as teacher_force mode"
    else:
        assert 100 > proportion_pred_train > 0, "proportion_pred_train should be between 0 and 100"
    auxilliary_task_norm_not_norm = "norm_not_norm" in tasks  # auxilliary_task_norm_not_norm
    auxilliary_task_pos = "pos" in tasks
    if "normalize" not in tasks:
        word_decoding = False
        char_decoding = False
    if not unrolling_word:
        assert not char_src_attention, "ERROR attention requires step by step unrolling  "
    printing("WARNING bucketing is {} ", var=bucketing, verbose=verbose, verbose_level=1)
    if freq_writer is None:
        freq_writer = freq_checkpointing
        printing("REPORTING freq_writer set to freq_checkpointing {}", var=[freq_checkpointing], verbose=verbose, verbose_level=1)
    if auxilliary_task_norm_not_norm:
        printing("MODEL : training model with auxillisary task (loss weighted with {})", var=[weight_binary_loss], verbose=verbose, verbose_level=1)
    if compute_scoring_curve:
        assert score_to_compute_ls is not None and mode_norm_ls is not None and freq_scoring is not None, \
            "ERROR score_to_compute_ls and mode_norm_ls should not be None"
    use_gpu = use_gpu_(use_gpu)
    hardware_choosen = "GPU" if use_gpu else "CPU"
    printing("{} hardware mode ", var=([hardware_choosen]), verbose_level=0, verbose=verbose)
    freq_checkpointing = int(n_epochs/10) if checkpointing and freq_checkpointing is None else freq_checkpointing
    assert add_start_char == 1, "ERROR : add_start_char must be activated due decoding behavior of output_text_"
    printing("WARNING : add_start_char is {} and add_end_char {}  ".format(add_start_char, add_end_char), verbose=verbose, verbose_level=0)
    printing("TRAINING : checkpointing every {} epoch", var=freq_checkpointing, verbose=verbose, verbose_level=1)
    if reload:
        assert model_full_name is not None and len(model_id_pref) == 0 and model_dir is not None and dict_path is not None
    else:
        assert model_full_name is None and model_dir is None

    if not debug:
        pdb.set_trace = lambda: 1

    loss_training = []
    loss_developing = []
    # was not able to use the template cause no more reinitialization of the variable
    loss_details_template = {'loss_seq_prediction': [], 'other': {}, 'loss_binary': [], 'loss_overall': []} if auxilliary_task_norm_not_norm else None
    if isinstance(train_path, list):
        evaluation_set_reporting = train_path.copy()
        evaluation_set_reporting.extend(dev_path)
    else:
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
                              test_path=test_path,
                              word_embed_dict={},
                              dry_run=False,
                              force_new_dic=True,
                              add_start_char=add_start_char, verbose=1)

        voc_size = len(char_dictionary.instance2index)+1
        word_voc_input_size = len(word_dictionary.instance2index)+1
        printing("DICTIONARY ; character vocabulary is len {} : {} ", var=str(len(char_dictionary.instance2index)+1, char_dictionary.instance2index), verbose=verbose, verbose_level=0)
        _train_path, _dev_path, _add_start_char = None, None, None
    else:
        voc_size = None
        word_voc_input_size = 0
        if not reload:
            # we need to feed the model the data so that it computes the model_specific_dictionary
            _train_path = train_path
            _dev_path = dev_path
            _test_path = test_path
            _add_start_char = add_start_char
        else:
            # as it reload : we don't need data
            _train_path, _dev_path, _test_path, _add_start_char = None, None, None, None

    model = LexNormalizer(generator=Generator, expand_vocab_dev_test=expand_vocab_dev_test,
                          #auxilliary_task_norm_not_norm=auxilliary_task_norm_not_norm,
                          dense_dim_auxilliary=dense_dim_auxilliary, dense_dim_auxilliary_2=dense_dim_auxilliary_2,
                          tasks=tasks,
                          weight_binary_loss=weight_binary_loss,
                          #auxilliary_task_pos=auxilliary_task_pos,
                          dense_dim_auxilliary_pos=dense_dim_auxilliary_pos,
                          dense_dim_auxilliary_pos_2=dense_dim_auxilliary_pos_2,
                          load=reload,
                          char_embedding_dim=char_embedding_dim, voc_size=voc_size,
                          dir_model=model_dir, use_gpu=use_gpu, dict_path=dict_path,
                          word_recurrent_cell_decoder=word_recurrent_cell_decoder,
                          word_recurrent_cell_encoder=word_recurrent_cell_encoder,
                          train_path=_train_path, dev_path=_dev_path, pos_specific_path=pos_specific_path,
                          add_start_char=_add_start_char,
                          model_specific_dictionary=model_specific_dictionary,
                          dir_word_encoder=dir_word_encoder,
                          drop_out_sent_encoder_cell=dropout_sent_encoder_cell, drop_out_word_encoder_cell=dropout_word_encoder_cell,
                          drop_out_word_decoder_cell=dropout_word_decoder_cell, drop_out_bridge=dropout_bridge,
                          drop_out_char_embedding_decoder=drop_out_char_embedding_decoder,
                          drop_out_word_encoder_out=drop_out_word_encoder_out,
                          drop_out_sent_encoder_out=drop_out_sent_encoder_out,
                          n_layers_word_encoder=n_layers_word_encoder, dir_sent_encoder=dir_sent_encoder,
                          n_layers_sent_cell=n_layers_sent_cell,
                          hidden_size_encoder=hidden_size_encoder, output_dim=output_dim,
                          model_id_pref=model_id_pref, model_full_name=model_full_name,
                          hidden_size_sent_encoder=hidden_size_sent_encoder, shared_context=shared_context,
                          unrolling_word=unrolling_word, char_src_attention=char_src_attention,
                          word_decoding=word_decoding,  dense_dim_word_pred=dense_dim_word_pred, dense_dim_word_pred_2=dense_dim_word_pred_2,dense_dim_word_pred_3=dense_dim_word_pred_3,
                          char_decoding=char_decoding,
                          mode_word_encoding=mode_word_encoding, char_level_embedding_projection_dim=char_level_embedding_projection_dim,
                          stable_decoding_state=stable_decoding_state, init_context_decoder=init_context_decoder,
                          symbolic_root=symbolic_root, symbolic_end=symbolic_end,
                          word_embed=word_embed, word_embedding_dim=word_embedding_dim, word_embedding_projected_dim=word_embedding_projected_dim,
                          word_embed_dir=extern_emb_dir, word_voc_input_size=word_voc_input_size, teacher_force=teacher_force,
                          activation_char_decoder=activation_char_decoder, activation_word_decoder=activation_word_decoder,
                          test_path=_test_path, extend_vocab_with_test=_test_path is not None,
                          attention_tagging=attention_tagging,
                          hidden_size_decoder=hidden_size_decoder, verbose=verbose, timing=timing)

    pos_batch = auxilliary_task_pos

    if use_gpu:
        model = model.cuda()
        printing("TYPE model is cuda : {} ", var=(next(model.parameters()).is_cuda), verbose=verbose, verbose_level=4)
        #model.decoder.attn_layer = model.decoder.attn_layer.cuda()
    if not model_specific_dictionary:
        model.word_dictionary, model.char_dictionary, model.pos_dictionary, \
        model.xpos_dictionary, model.type_dictionary = word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary

    starting_epoch = model.arguments["info_checkpoint"]["n_epochs"] if reload else 1
    reloading = "" if not reload else " reloaded from "+str(starting_epoch)
    n_epochs += starting_epoch
    if freezing_mode:
        assert freeze_ls_param_prefix is not None, "freeze_ls_param_prefix should not be None"
        printing("TRAINING : freezing is on for layers {} ", var=[freeze_ls_param_prefix], verbose=verbose, verbose_level=1)
        for name, param in model.named_parameters():
            for freeze_param in freeze_ls_param_prefix:
                if name.startswith(freeze_param):
                    param.requires_grad = False
                    printing("TRAINING : freezing {} parameter ", var=[name], verbose=verbose, verbose_level=1)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer == "adam":
        adam = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.98), eps=1e-9)
    elif optimizer == "bahdanu-adadelta":
        adam = torch.optim.Adadelta(parameters, eps=10e-6, rho=0.95)
    _loss_dev = 1000
    _loss_train = 1000
    counter_no_deacrease = 0
    saved_epoch = 1
    if reload:
        printing("TRAINING : RELOADED MODE , starting from checkpointed epoch {} ", var=starting_epoch, verbose_level=0, verbose=verbose)
    printing("TRAINING : Running from {} to {} epochs : training on {} evaluating on {}", var=(starting_epoch, n_epochs, train_path, dev_path), verbose=verbose, verbose_level=0)
    starting_time = time.time()
    total_time = 0
    x_axis_epochs = []
    epoch_ls_dev = []
    epoch_ls_train = []

    train_path = [train_path] if isinstance(train_path, str) else train_path
    dev_path = [dev_path] if isinstance(dev_path, str) else dev_path

    readers_train = readers_load(datasets=train_path, tasks=tasks, word_dictionary=model.word_dictionary,
                                 word_dictionary_norm=model.word_nom_dictionary, char_dictionary=model.char_dictionary,
                                 pos_dictionary=model.pos_dictionary, xpos_dictionary=model.xpos_dictionary,
                                 type_dictionary=model.type_dictionary, use_gpu=use_gpu,
                                 norm_not_norm=auxilliary_task_norm_not_norm, word_decoder=word_decoding,
                                 add_start_char=add_start_char, add_end_char=add_end_char, symbolic_end=symbolic_end,
                                 symbolic_root=symbolic_root,
                                 verbose=verbose)

    readers_dev = readers_load(datasets=dev_path, tasks=tasks, word_dictionary=model.word_dictionary,
                               word_dictionary_norm=model.word_nom_dictionary, char_dictionary=model.char_dictionary,
                               pos_dictionary=model.pos_dictionary, xpos_dictionary=model.xpos_dictionary,
                               type_dictionary=model.type_dictionary, use_gpu=use_gpu,
                               norm_not_norm=auxilliary_task_norm_not_norm, word_decoder=word_decoding,
                               add_start_char=add_start_char, add_end_char=add_end_char, symbolic_end=symbolic_end,
                               symbolic_root=symbolic_root,
                               verbose=verbose)

    dir_writer = os.path.join(overall_report_dir, "runs", "{}-model".format(model.model_full_name))
    writer = SummaryWriter(log_dir=dir_writer)
    printing("REPORT : summary writer will be located {}", var=[dir_writer], verbose_level=1, verbose=verbose)
    step_train = 0
    step_dev = 0
    if ADAPTABLE_SCORING:
        printing("WARNING : scoring epochs not regular (more at the begining ", verbose_level=1, verbose=verbose)
        freq_scoring = 1
    checkpoint_dir_former = None

    for epoch in tqdm(range(starting_epoch, n_epochs), disable_tqdm_level(verbose=verbose, verbose_level=0)):
        assert policy in AVAILABLE_SCHEDULING_POLICIES
        policy_dic = eval(policy)(epoch) if policy is not None else None
        #TODO : no need of re-ouptuting multi_task_mode : tasks should be harmonized to read
        multi_task_mode, ponderation_normalize_loss, weight_binary_loss, weight_pos_loss = scheduling_policy(epoch=epoch, phases_ls=policy_dic, tasks=tasks)
        printing("TRAINING Tasks scheduling : ponderation_normalize_loss is {} weight_binary_loss is {}"
                 " weight_pos_loss is {} mode is {} ",
                 var=[ponderation_normalize_loss, weight_binary_loss, weight_pos_loss, multi_task_mode], verbose=verbose, verbose_level=2)

        printing("TRAINING : Starting {} epoch out of {} ", var=(epoch+1, n_epochs), verbose= verbose, verbose_level=1)
        model.train()
        #batchIter = data_gen_conllu(data_read_train,model.word_dictionary, model.char_dictionary,normalization=normalization,get_batch_mode=get_batch_mode_all,batch_size=batch_size, extend_n_batch=extend_n_batch,print_raw=print_raw, timing=timing, pos_dictionary=model.pos_dictionary,verbose=verbose)
        batchIter = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers_train, batch_size=batch_size,
                                                       word_dictionary=model.word_dictionary,
                                                       char_dictionary=model.char_dictionary,
                                                       pos_dictionary=model.pos_dictionary,
                                                       get_batch_mode=get_batch_mode_all,
                                                       extend_n_batch=extend_n_batch,
                                                       verbose=verbose)
        start = time.time()
        printing("TRAINING : TEACHER FORCE : Schedule Sampling proportion of train on prediction is {} ", var=[proportion_pred_train],
                 verbose=verbose, verbose_level=2)
        loss_train, loss_details_train, step_train = run_epoch(batchIter, model,
                                                               LossCompute(model.generator, opt=adam,
                                                                           auxilliary_task_norm_not_norm=auxilliary_task_norm_not_norm,
                                                                           model=model,
                                                                           writer=writer, use="train",
                                                                           use_gpu=use_gpu, verbose=verbose,
                                                                           char_decoding=char_decoding, word_decoding=word_decoding,
                                                                           pos_pred=auxilliary_task_pos,
                                                                           timing=timing),
                                                               verbose=verbose, i_epoch=epoch,
                                                               multi_task_mode=multi_task_mode,
                                                               n_epochs=n_epochs, timing=timing,
                                                               weight_binary_loss=weight_binary_loss,
                                                               weight_pos_loss=weight_pos_loss,
                                                               ponderation_normalize_loss=ponderation_normalize_loss,
                                                               step=step_train,
                                                               clipping=clipping,
                                                               pos_batch=pos_batch,
                                                               proportion_pred_train=proportion_pred_train,
                                                               log_every_x_batch=100)

        writer_weights_and_grad(model=model, freq_writer=freq_writer, epoch=epoch, writer=writer, verbose=verbose)

        _train_ep_time, start = get_timing(start)
        model.eval()
        # TODO : should be added in the freq_checkpointing orhterwise useless
        #batchIter_eval = data_gen_conllu(data_read_dev,model.word_dictionary, model.char_dictionary,batch_size=batch_size, get_batch_mode=False,normalization=normalization, extend_n_batch=1,pos_dictionary=model.pos_dictionary, verbose=verbose)
        batchIter_eval = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers_dev, batch_size=batch_size,
                                                            word_dictionary=model.word_dictionary,
                                                            char_dictionary=model.char_dictionary,
                                                            pos_dictionary=model.pos_dictionary,
                                                            extend_n_batch=1, get_batch_mode=False, verbose=verbose)
        _create_iter_time, start = get_timing(start)
        # TODO : should be able o factorize this to have a single run_epoch() for train and dev (I think the computaiton would be same )
        # TODO : should not evaluate for each epoch : should evalaute every x epoch : check if it decrease and checkpoint
        if (dev_report_loss and (epoch % freq_checkpointing == 0)) or (epoch + 1 == n_epochs):
            printing("EVALUATION : computing loss on dev epoch {}  ",var=epoch, verbose=verbose, verbose_level=1)
            loss_obj = LossCompute(model.generator, use_gpu=use_gpu, verbose=verbose,
                                   writer=writer, use="dev",
                                   pos_pred=auxilliary_task_pos,
                                   char_decoding=char_decoding, word_decoding=word_decoding,
                                   auxilliary_task_norm_not_norm=auxilliary_task_norm_not_norm)
            loss_dev, loss_details_dev, step_dev = run_epoch(batchIter_eval, model, loss_compute=loss_obj,
                                                             i_epoch=epoch, n_epochs=n_epochs,
                                                             verbose=verbose, timing=timing, step=step_dev,
                                                             weight_binary_loss=weight_binary_loss,
                                                             ponderation_normalize_loss=ponderation_normalize_loss,
                                                             weight_pos_loss=weight_pos_loss,
                                                             pos_batch=pos_batch,
                                                             log_every_x_batch=100)

            loss_developing.append(loss_dev)
            epoch_ls_dev.append(epoch)

            if auxilliary_task_norm_not_norm:
                # in this case we report loss detail
                for ind, loss_key in enumerate(loss_details_dev.keys()):
                    if loss_key != "other":
                        loss_details_template[loss_key].append(loss_details_dev[loss_key])
            else:
                loss_details_template = None

        _eval_time, start = get_timing(start)
        loss_training.append(loss_train)
        epoch_ls_train.append(epoch)
        time_per_epoch = time.time() - starting_time
        total_time += time_per_epoch
        starting_time = time.time()

        # computing exact/edit score
        exact_only = False
        if compute_scoring_curve and ((epoch % freq_scoring == 0) or (epoch+1 == n_epochs)):
            if epoch<1 and ADAPTABLE_SCORING:
                freq_scoring*=5
            if epoch>4 and epoch<6 and ADAPTABLE_SCORING:
                freq_scoring*=3
            if epoch > 14 and epoch < 15 and ADAPTABLE_SCORING:
                freq_scoring*=2
            if (epoch+1 == n_epochs):
              printing("EVALUATION : final scoring ", verbose, verbose_level=0)
            x_axis_epochs.append(epoch)
            printing("EVALUATION : Computing score on {} and {}  ", var=(score_to_compute_ls,mode_norm_ls), verbose=verbose, verbose_level=1)
            for eval_data in evaluation_set_reporting:
                eval_label = REPO_DATASET[eval_data]
                assert len(set(evaluation_set_reporting)) == len(evaluation_set_reporting),\
                    "ERROR : twice the same dataset has been provided for reporting which will mess up the loss"
                printing("EVALUATION on {} ", var=[eval_data], verbose=verbose, verbose_level=1)
                scores = evaluate(data_path=eval_data,
                                  use_gpu=use_gpu,
                                  overall_label=overall_label,overall_report_dir=overall_report_dir,
                                  score_to_compute_ls=score_to_compute_ls, mode_norm_ls=mode_norm_ls,
                                  label_report=eval_label, model=model,
                                  normalization=normalization, print_raw=False,
                                  model_specific_dictionary=True,
                                  get_batch_mode_evaluate=get_batch_mode_all,
                                  compute_mean_score_per_sent=compute_mean_score_per_sent,
                                  batch_size=batch_size,
                                  word_decoding=word_decoding,
                                  dir_report=model.dir_model,
                                  debug=debug,
                                  verbose=verbose)

                # dirty but do the job
                exact_only = True
                curve_scores = update_curve_dic(score_to_compute_ls=score_to_compute_ls, mode_norm_ls=mode_norm_ls,
                                                eval_data=eval_label,
                                                former_curve_scores=curve_scores, scores=scores, exact_only=exact_only)
                curve_ls_tuple = [(loss_ls, label) for label, loss_ls in curve_scores.items() if isinstance(loss_ls, list)]
                curves = [tupl[0] for tupl in curve_ls_tuple]
                val_ls = [tupl[1]+"({}tok)".format(info_token) for tupl in curve_ls_tuple for data, info_token in curve_scores.items() if not isinstance(info_token, list) if tupl[1].endswith(data)]
            score_to_compute_ls = ["exact"] if exact_only else score_to_compute_ls
            for score_plot in score_to_compute_ls:
                # dirty but do the job
                print(val_ls)
                if exact_only:
                    val_ls = [val for val in val_ls if val.startswith("exact-all") or val.startswith("exact-NORMED") or val.startswith("exact-NEED_NORM") ]
                    #val_ls = ["{}-all-{}".format(metric,REPO_DATASET[eval]) for eval in evaluation_set_reporting for metric in ["exact", "edit"]]
                    curves = [curve for curve in curves if len(curve)>0]

                simple_plot_ls(losses_ls=curves, labels=val_ls, final_loss="", save=True, filter_by_label=score_plot,  x_axis=x_axis_epochs,
                               dir=model.dir_model, prefix=model.model_full_name, epochs=str(epoch)+reloading, verbose=verbose, lr=lr,
                               label_color_0=REPO_DATASET[evaluation_set_reporting[0]], label_color_1=REPO_DATASET[evaluation_set_reporting[1]])

        # WARNING : only saving if we decrease not loading former model if we relaod
        if (checkpointing and epoch % freq_checkpointing == 0) or (epoch+1 == n_epochs):

            dir_plot_detailed = simple_plot(final_loss=0,
                                            epoch_ls_1=epoch_ls_dev,epoch_ls_2=epoch_ls_dev,
                                            loss_2=loss_details_template.get("loss_binary", None),
                                            loss_ls=loss_details_template["loss_seq_prediction"],
                                            epochs=str(epoch) + reloading,
                                            label="dev-seq_prediction",
                                            label_2="dev-binary",
                                            save=True, dir=model.dir_model,
                                            verbose=verbose, verbose_level=1,
                                            lr=lr, prefix=model.model_full_name+"-details",
                                            show=False) if loss_details_template is not None else None

            dir_plot = simple_plot(final_loss=loss_train, loss_2=loss_developing, loss_ls=loss_training,
                                   epochs=str(epoch)+reloading,
                                   epoch_ls_1=epoch_ls_train, epoch_ls_2=epoch_ls_dev, label=label_train+"-train",
                                   label_2=label_dev+"-dev", save=True, dir=model.dir_model, verbose=verbose,
                                   verbose_level=1, lr=lr, prefix=model.model_full_name, show=False)
            model, _loss_dev, counter_no_deacrease, saved_epoch, checkpoint_dir_former = \
                    checkpoint(loss_saved=_loss_dev, loss=loss_dev, model=model, counter_no_decrease=counter_no_deacrease,
                               checkpoint_dir_former=checkpoint_dir_former,
                               saved_epoch=saved_epoch, model_dir=model.dir_model,
                               extra_checkpoint_label="1st_train" if not reload else "start_{}_ep-{}".format(starting_epoch, extra_arg_specific_label),
                               extra_arg_specific_label=extra_arg_specific_label,
                               info_checkpoint={"n_epochs": epoch, "batch_size": batch_size, "optimizer": optimizer,
                                                "gradient_clipping": clipping,
                                                "tasks_schedule_policy": policy,
                                                "teacher_force": teacher_force,
                                                "proportion_pred_train": proportion_pred_train,
                                                "train_data_path": train_path, "dev_data_path": dev_path,
                                                "other": {"error_curves": dir_plot, "loss": _loss_dev,
                                                          "error_curves_details":dir_plot_detailed,
                                                          "weight_binary_loss": weight_binary_loss*int(auxilliary_task_norm_not_norm),
                                                          "weight_pos_loss": weight_pos_loss*int(auxilliary_task_pos),
                                                          "ponderation_normalize_loss": ponderation_normalize_loss,
                                                          "data": "dev", "seed(np/torch)": (SEED_TORCH, SEED_TORCH),
                                                          "extend_n_batch": extend_n_batch,
                                                          "lr": lr, "optim_strategy":"lr_constant",
                                                          "time_training(min)": "{0:.2f}".format(total_time/60),
                                                          "average_per_epoch(min)": "{0:.2f}".format((total_time/n_epochs)/60)}},
                               epoch=epoch, epochs=n_epochs-1,
                               keep_all_checkpoint=False if epoch > starting_epoch else True,# we have nothing to remove after 1st epoch
                               verbose=verbose)
            if counter_no_deacrease*freq_checkpointing >= BREAKING_NO_DECREASE:
                printing("CHECKPOINTING : Breaking training : loss did not decrease on dev for 10 checkpoints "
                         "so keeping model from {} epoch  ".format(saved_epoch),
                         verbose=verbose, verbose_level=0)
                break
        printing("LOSS train {:.3f}, dev {:.3f} for epoch {} out of {} epochs ",
                 var=(loss_train, loss_dev,
                      epoch, n_epochs),
                 verbose=verbose, verbose_level=1)

        if timing:
            print("Summary : {}".format(OrderedDict([("_train_ep_time", _train_ep_time), ("_create_iter_time", _create_iter_time), ("_eval_time",_eval_time) ])))

    writer.close()

    simple_plot(final_loss=loss_dev,
                loss_ls=loss_training, loss_2=loss_developing,
                epoch_ls_1=epoch_ls_train,epoch_ls_2=epoch_ls_dev,
                epochs=n_epochs, save=True,
                dir=model.dir_model, label=label_train, label_2=label_dev,
                lr=lr, prefix=model.model_full_name+"-LAST")
   
    return model.model_full_name