from io_.info_print import printing
import os
import torch
import pdb

def checkpoint(loss_saved, loss, model, model_dir, epoch, epochs, info_checkpoint, saved_epoch,
               counter_no_decrease, verbose, extra_checkpoint_label="",extra_arg_specific_label="",
               checkpointing_metric="loss-dev-all",
               checkpoint_dir_former=None, keep_all_checkpoint=False):
    pdb.set_trace()
    if loss < loss_saved:
        saved_epoch = epoch
        loss_saved = loss
        printing('Checkpoint info : {} decreased so saving model saved epoch is {} (counter_no_decrease set to 0)',
                 var=[checkpointing_metric, saved_epoch],
                 verbose=verbose, verbose_level=1)
        _,_, checkpoint_dir = model.save(model_dir, model, info_checkpoint=info_checkpoint,
                                         extra_arg_specific_label=extra_arg_specific_label,
                                         suffix_name="{}-{}of{}epoch".format(extra_checkpoint_label, epoch, epochs), verbose=verbose)
        if not keep_all_checkpoint:
            model.rm_checkpoint(checkpoint_dir_former, verbose=verbose)
        checkpoint_dir_former = checkpoint_dir
        counter_no_decrease = 0
    else:
        # could add loading former model if loss suddenly pick
        #printing('Checkpoint info : Loss decreased so saving model', verbose=verbose, verbose_level=1)
        #model.load_state_dict(torch.load(checkpoint_dir))
        # TODO : load former checkpoint : and do change loss append IF error suddendly pick
        counter_no_decrease += 1
        printing("Checkpoint info: {} did not decrease so keeping former model of epoch {} "
                 "counter_no_decrease is now {} ",
                 var=[checkpointing_metric, saved_epoch, counter_no_decrease], verbose=verbose, verbose_level=1)

    return model, loss_saved, counter_no_decrease, saved_epoch, checkpoint_dir_former


def update_curve_dic(score_to_compute_ls, mode_norm_ls, eval_data, scores, former_curve_scores, exact_only=True):
    if exact_only:
        score_to_compute_ls = ["exact"]
        mode_norm_ls = ["all", "NORMED", "NEED_NORM"]
    for mode in mode_norm_ls:
        for score in score_to_compute_ls:
            # adding count
            former_curve_scores[mode + "-" + eval_data] = scores[score + "-" + mode + "-total_tokens"]
            former_curve_scores[score + "-" + mode + "-" + eval_data].append(
                scores[score + "-" + mode] / scores[score + "-" + mode + "-total_tokens"])
    return former_curve_scores


def get_args(args, dropout_loading_strict=True, verbose=0):
    default_dropout = None if dropout_loading_strict else 0
    if dropout_loading_strict:
        printing("WARNING : errors might come from misloading of dropout ", verbose=verbose, verbose_level=0)
    return args["char_embedding_dim"], args["output_dim"], args["hidden_size_encoder"], args[
        "hidden_size_sent_encoder"], args["encoder_arch"].get("dropout_sent_encoder_cell", default_dropout), args["encoder_arch"].get(
        "dropout_word_encoder_cell", default_dropout), args["encoder_arch"].get("drop_out_sent_encoder_out", default_dropout), args[
               "encoder_arch"].get("drop_out_word_encoder_out", default_dropout), \
           args["encoder_arch"].get("n_layers_word_encoder"), args["encoder_arch"].get("n_layers_sent_cell",1),args["encoder_arch"].get("dir_sent_encoder"), \
           args["encoder_arch"].get("word_recurrent_cell_encoder", None), args["encoder_arch"].get("dir_word_encoder",1), \
           args["hidden_size_decoder"], args["decoder_arch"].get("cell_word", None), args["decoder_arch"].get(
              "drop_out_word_decoder_cell", default_dropout), args["decoder_arch"].get("drop_out_char_embedding_decoder", default_dropout),\
            args.get("auxilliary_arch", {}).get("auxilliary_task_norm_not_norm", False), args["decoder_arch"].get("unrolling_word", False), args["decoder_arch"].get("char_src_attention", False),\
            args.get("auxilliary_arch", {}).get("auxilliary_task_norm_not_norm-dense_dim", None), args.get("shared_context","all"),  args["decoder_arch"].get("teacher_force", 1), \
           args.get("auxilliary_arch", {}).get("auxilliary_task_norm_not_norm-dense_dim_2"), args["decoder_arch"].get("stable_decoding_state", False), args["decoder_arch"].get("init_context_decoder",True),\
           args["decoder_arch"].get("word_decoding", 0), args["decoder_arch"].get("char_decoding", 1), \
           args.get("auxilliary_arch", {}).get("auxilliary_task_pos", False), \
           args.get("auxilliary_arch", {}).get("dense_dim_auxilliary_pos", None), args.get("auxilliary_arch", {}).get("dense_dim_auxilliary_pos_2", None), \
            args["decoder_arch"].get("dense_dim_word_pred",0), args["decoder_arch"].get("dense_dim_word_pred_2",0), args["decoder_arch"].get("dense_dim_word_pred_3",0), \
           args.get("symbolic_root", False), args.get("symbolic_end", False), \
           args["encoder_arch"].get("word_embedding_dim", 0), args["encoder_arch"].get("word_embed", False), \
           args["encoder_arch"].get("word_embedding_projected_dim", None),  args["decoder_arch"].get("activation_char_decoder"), args["decoder_arch"].get("activation_word_decoder"), \
            args["encoder_arch"].get("attention_tagging", False), \
           args["encoder_arch"].get("char_level_embedding_projection_dim",0), args["encoder_arch"].get("mode_word_encoding","cat"), args.get("multi_task_loss_ponderation", "all")

