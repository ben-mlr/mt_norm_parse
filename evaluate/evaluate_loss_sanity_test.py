from io_.data_iterator import data_gen_multi_task_sampling_batch, readers_load
from model.loss import LossCompute
from training.epoch_train import run_epoch
from io_.info_print import printing
from env.project_variables import REPO_LABEL2SET


def get_loss(model, data_label, tasks, use_gpu, word_decoding, char_decoding,
             max_char_len, bucketing,batch_size,
             symbolic_end=1, add_end_char=1, add_start_char=1,
             symbolic_root=1,
             verbose=1):

    ponderation_normalize_loss = model.arguments["hyperparameters"]["ponderation_normalize_loss"]
    weight_pos_loss = model.arguments["hyperparameters"]["weight_pos_loss"]
    weight_binary_loss = model.arguments["hyperparameters"]["weight_binary_loss"]
    dataset = [REPO_LABEL2SET[_data_label] for _data_label in data_label]
    printing("SANITY TEST performed on {}".format(dataset), verbose=verbose, verbose_level=1)
    readers_dev = readers_load(datasets=dataset,
                               tasks=tasks, word_dictionary=model.word_dictionary,
                               word_dictionary_norm=model.word_nom_dictionary, char_dictionary=model.char_dictionary,
                               pos_dictionary=model.pos_dictionary, xpos_dictionary=model.xpos_dictionary,
                               type_dictionary=model.type_dictionary, use_gpu=use_gpu,
                               norm_not_norm="norm_not_norm" in tasks, word_decoder=word_decoding,
                               add_start_char=add_start_char, add_end_char=add_end_char, symbolic_end=symbolic_end,
                               symbolic_root=symbolic_root, bucket=bucketing, max_char_len=max_char_len,
                               verbose=verbose)

    batchIter_eval = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers_dev, batch_size=batch_size,
                                                        word_dictionary=model.word_dictionary,
                                                        char_dictionary=model.char_dictionary,
                                                        word_dictionary_norm=model.word_nom_dictionary,
                                                        pos_dictionary=model.pos_dictionary, dropout_input=0,
                                                        extend_n_batch=1, get_batch_mode=False, verbose=verbose)

    printing("SANITY TEST EVALUATION : computing loss ", verbose=verbose, verbose_level=2)

    loss_obj = LossCompute(model.generator, use_gpu=use_gpu, verbose=verbose,
                           multi_task_loss_ponderation=model.multi_task_loss_ponderation,
                           use="dev",
                           pos_pred="pos" in tasks,
                           tasks=tasks,
                           vocab_char_size=len(list(model.char_dictionary.instance2index.keys())) + 1,
                           char_decoding=char_decoding, word_decoding=word_decoding,
                           auxilliary_task_norm_not_norm="norm_not_norm" in tasks)

    print("PONDERATION", ponderation_normalize_loss)

    loss_dev, loss_details_dev, step_dev = run_epoch(batchIter_eval, model, loss_compute=loss_obj,
                                                     verbose=verbose, timing="", step=0,
                                                     weight_binary_loss=weight_binary_loss,
                                                     ponderation_normalize_loss=ponderation_normalize_loss,
                                                     weight_pos_loss=weight_pos_loss,
                                                     pos_batch="pos" in tasks,
                                                     log_every_x_batch=100)

    return loss_dev, loss_details_dev, step_dev