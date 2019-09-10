from env.importing import *
from io_.info_print import printing
from env.default_hyperparameters import *
from env.project_variables import MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE, DIC_ARGS, AVAILABLE_TASKS


def parse_argument_dictionary(argument_as_string, tasks=None, hyperparameter="multi_task_loss_ponderation", verbose=1):
    assert hyperparameter in DIC_ARGS, "ERROR only supported"
    if argument_as_string in MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE:
        return argument_as_string
    else:
        dic = OrderedDict()
        if hyperparameter == "multi_task_loss_ponderation":

            assert tasks is not None
            for task in tasks:
                if task == "parsing":
                    for sub in ["parsing-heads", "parsing-types"]:
                        pattern = "{}=([^=]*),".format(sub)
                        match = re.search(pattern, argument_as_string)
                        assert match is not None, "ERROR : pattern {} not found for task {} in argument_as_string {}  ".format(
                            pattern, task, argument_as_string)
                        dic[sub] = eval(match.group(1))
                elif task == "normalize":
                    for sub in ["normalize", "append_masks"]:
                        pattern = "{}=([^=]*),".format(sub)
                        match = re.search(pattern, argument_as_string)
                        if sub == "normalize":
                            assert match is not None, "ERROR : pattern {} not found for task {} in argument_as_string {}  ".format( pattern, task, argument_as_string)
                            dic[sub] = eval(match.group(1))
                        else:
                            if match is not None:
                                dic[sub] = eval(match.group(1))
                if task != "all" and task != "parsing":
                    pattern = "{}=([^=]*),".format(task)
                    match = re.search(pattern, argument_as_string)
                    assert match is not None, "ERROR : pattern {} not found for task {} in argument_as_string {}  ".format(pattern, task, argument_as_string)
                    dic[task] = eval(match.group(1))

            printing("SANITY CHECK : multi_task_loss_ponderation {} ", var=[argument_as_string], verbose_level=3, verbose=verbose)
        elif hyperparameter == "lr":
            # to handle several optimizers
            try :
                assert isinstance(eval(argument_as_string), float)
                return eval(argument_as_string)
            except:
                argument_as_string = argument_as_string.split(",")
                for arg in argument_as_string[:-1]:
                    # DIFFERENCE WITH ABOVE IS THE COMMA
                    pattern = "([^=]*)=([^=]*)"
                    match = re.search(pattern, arg)
                    assert match is not None, "ERROR : pattern {} not found in argument_as_string {}  ".format(pattern,  arg)
                    print("--> ", match.group(1),match.group(2) )
                    dic[match.group(1)] = float(match.group(2))
        elif hyperparameter == "tasks":
            argument_as_string = [task_simultaneous.split(",") for task_simultaneous in argument_as_string]

        return dic


def args_train(mode="command_line", script="train_evaluate_run"):
    """NB : arguments default used variables to be aligned with grid_tool default
    (because : default in grid tools means grid_list arguments is len 1 means not passed to argt_train means call default)
    (could thing of a more integrated way to do it )"""
    assert mode in ["command_line", "script"], "mode should be in '[command_line, script]"
    assert script in AVAILABLE_TRAINING_EVAL_SCRIPT, "ERROR script {} not supported here".format(script)
    parser = argparse.ArgumentParser()

    # training opti
    parser.add_argument("--batch_size", default=2, type=int, help="display a square of a given number")
    parser.add_argument("--epochs", default=1, type=int, help="display a square of a given number")
    parser.add_argument("--lr", default=DEFAULT_LR, help="display a square of a given number")
    # id and reporting
    parser.add_argument("--model_id_pref", required=mode == "command_line", help="display a square of a given number")
    parser.add_argument("--overall_label", default="DEFAULT", help="display a square of a given number")
    parser.add_argument("--overall_report_dir", required=mode == "command_line",
                        help="display a square of a given number")
    # logging and debugging
    parser.add_argument("--debug", action="store_true", help="display a square of a given number")
    parser.add_argument("--warmup", action="store_true", help="display a square of a given number")
    parser.add_argument("--verbose", default=1, type=int, help="display a square of a given number")

    parser.add_argument("--gpu", default=None, type=str, help="display a square of a given number")
    # data
    parser.add_argument("--train_path", required=mode == "command_line", nargs='+', help='<Required> Set flag')
    parser.add_argument("--dev_path", required=mode == "command_line", nargs='+', help='<Required> Set flag')
    parser.add_argument('--test_paths', nargs='+', help='<Required> Set flag', default=None,
                        required=(script == "train_evaluate_bert_normalizer"))

    parser.add_argument('--multitask', type=int, default=None)

    parser.add_argument('--tasks', nargs='+', help='<Required> Set flag', default=DEFAULT_TASKS)

    if script == "train_evaluate_run":
        parser.add_argument("--hidden_size_encoder", type=int, required=True, help="display a square of a given number")
        parser.add_argument("--hidden_size_sent_encoder", required=True, type=int, help="display a square of a given number")
        parser.add_argument("--hidden_size_decoder", required=True, type=int, help="display a square of a given number")

        parser.add_argument("--word_embed", type=int, default=DEFAULT_WORD_EMBED, help="display a square of a given number")
        parser.add_argument("--word_embedding_dim", required=True, type=int, help="display a square of a given number")
        parser.add_argument("--word_embedding_projected_dim", default=DEFAULT_WORD_EMBEDDING_PROJECTED, type=int, help="display a square of a given number")
        parser.add_argument("--output_dim", required=True, type=int, help="display a square of a given number")
        parser.add_argument("--char_embedding_dim",required=True, type=int, help="display a square of a given number")

        parser.add_argument("--mode_word_encoding", default=DEFAULT_MODE_WORD_ENCODING, type=str, help="display a square of a given number")
        parser.add_argument("--char_level_embedding_projection_dim", default=DEFAULT_CHAR_LEVEL_EMBEDDING_PROJECTION, type=int, help="display a square of a given number")

        parser.add_argument("--dropout_sent_encoder", default=0, type=float, help="display a square of a given number")
        parser.add_argument("--dropout_word_encoder_cell", default=DEFAULT_DROPOUT_WORD_ENCODER_CELL, type=float, help="display a square of a given number")
        parser.add_argument("--dropout_word_decoder", default=0, type=float, help="display a square of a given number")
        parser.add_argument("--drop_out_word_encoder_out", default=0, type=float, help="display a square of a given number")
        parser.add_argument("--drop_out_sent_encoder_out", default=0, type=float, help="display a square of a given number")
        parser.add_argument("--dropout_bridge", default=0, type=float, help="display a square of a given number")
        parser.add_argument("--drop_out_char_embedding_decoder", default=0, type=float, help="display a square of a given number")

        parser.add_argument("--n_layers_word_encoder", default=DEFAULT_LAYER_WORD_ENCODER, type=int, help="display a square of a given number")
        parser.add_argument("--n_layers_sent_cell", default=DEFAULT_LAYERS_SENT_CELL, type=int, help="display a square of a given number")

        parser.add_argument("--dir_sent_encoder", default=DEFAULT_DIR_SENT_ENCODER, type=int, help="display a square of a given number")

        parser.add_argument("--word_recurrent_cell_encoder", default=DEFAULT_WORD_RECURRENT_CELL, help="display a square of a given number")
        parser.add_argument("--word_recurrent_cell_decoder", default="LSTM", help="display a square of a given number")

        parser.add_argument("--dense_dim_auxilliary", default=None, type=int, help="display a square of a given number")
        parser.add_argument("--dense_dim_auxilliary_2", default=None, type=int, help="displaqy a square of a given number")

        parser.add_argument("--unrolling_word", default=DEFAULT_WORD_UNROLLING, type=int, help="display a square of a given number")

        parser.add_argument("--char_src_attention", default=DEFAULT_CHAR_SRC_ATTENTION, type=int, help="display a square of a given number")
        parser.add_argument("--dir_word_encoder", default=DEFAULT_DIR_WORD_ENCODER, type=int, help="display a square of a given number")
        parser.add_argument("--weight_binary_loss", default=1, type=float, help="display a square of a given number")
        parser.add_argument("--shared_context", default=DEFAULT_SHARED_CONTEXT, help="display a square of a given number")

        parser.add_argument("--policy", default=None, type=int, help="display a square of a given number")
        parser.add_argument("--gradient_clipping", type=int, default=DEFAULT_CLIPPING, help="display a square of a given number")
        parser.add_argument("--teacher_force", type=int, default=DEFAULT_TEACHER_FORCE, help="display a square of a given number")
        parser.add_argument("--proportion_pred_train", type=int, default=DEFAULT_PROPORTION_PRED_TRAIN, help="display a square of a given number")

        parser.add_argument("--stable_decoding_state", type=int, default=DEFAULT_STABLE_DECODING, help="display a square of a given number")
        parser.add_argument("--init_context_decoder", type=int, default=not DEFAULT_STABLE_DECODING, help="display a square of a given number")
        parser.add_argument("--optimizer", default="adam", help="display a square of a given number ")

        parser.add_argument("--word_decoding", default=DEFAULT_WORD_DECODING, type=int, help="display a square of a given number ")

        parser.add_argument("--attention_tagging", default=DEFAULT_ATTENTION_TAGGING, type=int, help="display a square of a given number ")

        parser.add_argument("--dense_dim_word_pred", type=int, default=None, help="display a square of a given number ")
        parser.add_argument("--dense_dim_word_pred_2", type=int, default=None, help="display a square of a given number ")
        parser.add_argument("--dense_dim_word_pred_3", type=int, default=0, help="display a square of a given number ")

        parser.add_argument("--word_embed_init", default=DEFAULT_WORD_EMBED_INIT,help="display a square of a given number")
        parser.add_argument("--char_decoding", type=int, default=True, help="display a square of a given number")

        parser.add_argument("--dense_dim_auxilliary_pos",type=int, default=None, help="display a square of a given number")
        parser.add_argument("--dense_dim_auxilliary_pos_2",type=int, default=None, help="display a square of a given number")
        parser.add_argument("--activation_char_decoder", default=None, help="display a square of a given number")
        parser.add_argument("--activation_word_decoder", default=None, help="display a square of a given number")

        parser.add_argument("--checkpointing_metric", required=mode == "command_line", nargs='+', help='<Required> Set flag')

        parser.add_argument("--pos_specific_path", default=None, type=str, help="display a square of a given number")
        parser.add_argument("--expand_vocab_dev_test", default=True, help="display a square of a given number")

        parser.add_argument("--multi_task_loss_ponderation", type=str, default="all", help="display a square of a given number")
        parser.add_argument("--scoring_func", default="exact_match", type=str, help="display a square of a given number")

        parser.add_argument("--dropout_input", default=0, type=float, help="display a square of a given number")



        args = parser.parse_args()

        if not args.word_embed:
            args.word_embedding_dim = 0

    elif script == "train_evaluate_bert_normalizer":

        parser.add_argument("--initialize_bpe_layer", required=True, type=int, help="display a square of a given number")
        parser.add_argument("--bert_model", required=True, type=str, help="display a square of a given number")
        parser.add_argument("--freeze_parameters", required=True, type=int, help="display a square of a given number")
        parser.add_argument('--freeze_layer_prefix_ls', nargs='+', help='<Required> Set flag', default=None)
        parser.add_argument('--dropout_classifier', type=float, default=None)
        parser.add_argument('--fine_tuning_strategy', type=str, default="standart")

        parser.add_argument('--heuristic_ls', nargs='+', help='<Required> Set flag', default=None)
        parser.add_argument('--gold_error_detection', type=int, default=0)
        parser.add_argument('--dropout_input_bpe', type=float, default=0.)
        parser.add_argument('--dropout_bert', type=float, default=0.)

        parser.add_argument('--masking_strategy', nargs='+', help='<Required> Set flag', default=None)
        parser.add_argument('--portion_mask', type=float, default=None)
        parser.add_argument('--init_args_dir', type=str, default=None)
        parser.add_argument('--norm_2_noise_training', type=float, default=None)
        parser.add_argument('--aggregating_bert_layer_mode', type=str, default=None)

        parser.add_argument('--bert_module', type=str, default=None)
        parser.add_argument('--layer_wise_attention', type=int, default=0)

        parser.add_argument('--tokenize_and_bpe', type=int, default=0)
        parser.add_argument('--append_n_mask', type=int, default=0)

        parser.add_argument('--multi_task_loss_ponderation', type=str, default=None)

        parser.add_argument('--memory_efficient_iterator', type=int, default=0)

        parser.add_argument("--n_iter_max_train", default=1000000, type=int, help="display a square of a given number")
        parser.add_argument("--saving_every_n_epoch", default=100, type=int, help="display a square of a given number")
        parser.add_argument("--name_inflation", default=0, type=int, help="display a square of a given number")
        parser.add_argument("--demo_run", default=0, type=int, help="means running for 5 iteration max ")

        args = parser.parse_args()

        if args.aggregating_bert_layer_mode is not None:
            try:
                assert isinstance(eval(args.aggregating_bert_layer_mode),int)
                # it should be an int
                args.aggregating_bert_layer_mode = eval(args.aggregating_bert_layer_mode)
            except:
                # it should be a string
                pass

    return args


