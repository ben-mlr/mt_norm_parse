import argparse

def args_train(mode="command_line"):

    assert mode in ["command_line", "script"], "mode should be in '[command_line, script]"

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size_encoder", type=int, default=10, help="display a square of a given number")
    parser.add_argument("--hidden_size_sent_encoder", default=10, help="display a square of a given number")
    parser.add_argument("--hidden_size_decoder", default=10, help="display a square of a given number")

    parser.add_argument("--word_embed", type=bool, default=False, help="display a square of a given number")
    parser.add_argument("--word_embedding_dim", default=10, help="display a square of a given number")
    parser.add_argument("--word_embedding_projected_dim", default=None, help="display a square of a given number")
    parser.add_argument("--output_dim", default=10, help="display a square of a given number")
    parser.add_argument("--char_embedding_dim", default=10, help="display a square of a given number")

    parser.add_argument("--batch_size", default=2, help="display a square of a given number")

    parser.add_argument("--dropout_sent_encoder", default=0, help="display a square of a given number")
    parser.add_argument("--dropout_word_encoder", default=0, help="display a square of a given number")
    parser.add_argument("--dropout_word_decoder", default=0, help="display a square of a given number")
    parser.add_argument("--drop_out_word_encoder_out", default=0, help="display a square of a given number")
    parser.add_argument("--drop_out_sent_encoder_out", default=0, help="display a square of a given number")
    parser.add_argument("--dropout_bridge", default=0, help="display a square of a given number")
    parser.add_argument("--drop_out_char_embedding_decoder", default=0, help="display a square of a given number")

    parser.add_argument("--n_layers_word_encoder", default=1, help="display a square of a given number")
    parser.add_argument("--n_layers_sent_cell", default=1, help="display a square of a given number")

    parser.add_argument("--dir_sent_encoder", default=2, help="display a square of a given number")

    parser.add_argument("--word_recurrent_cell_encoder", default="LSTM", help="display a square of a given number")
    parser.add_argument("--word_recurrent_cell_decoder", default="LSTM", help="display a square of a given number")

    parser.add_argument("--dense_dim_auxilliary", default=None, help="display a square of a given number")
    parser.add_argument("--dense_dim_auxilliary_2", default=None, help="display a square of a given number")

    parser.add_argument("--unrolling_word", default=True, help="display a square of a given number")

    parser.add_argument("--char_src_attention", default=False, help="display a square of a given number")
    parser.add_argument("--dir_word_encoder", default=1, type=int, help="display a square of a given number")
    parser.add_argument("--weight_binary_loss", default=1 ,type=float, help="display a square of a given number")
    parser.add_argument("--shared_context", default="all", help="display a square of a given number")

    parser.add_argument("--policy", default=None, help="display a square of a given number")
    parser.add_argument("--lr", default=0.001, help="display a square of a given number")
    parser.add_argument("--gradient_clipping", default=None, help="display a square of a given number")
    parser.add_argument("--teacher_force", default=True, help="display a square of a given number")
    parser.add_argument("--proportion_pred_train", default=None, help="display a square of a given number")

    parser.add_argument("--stable_decoding_state", default=False, help="display a square of a given number")
    parser.add_argument("--init_context_decoder", default=True, help="display a square of a given number")
    parser.add_argument("--optimizer", default="adam", help="display a square of a given number ")

    parser.add_argument("--word_decoding", default=False, help="display a square of a given number ")

    parser.add_argument("--dense_dim_word_pred", default=None, help="display a square of a given number ")
    parser.add_argument("--dense_dim_word_pred_2", default=None, help="display a square of a given number ")
    parser.add_argument("--dense_dim_word_pred_3", default=0, help="display a square of a given number ")

    parser.add_argument("--word_embed_init", default=None, help="display a square of a given number")
    parser.add_argument("--char_decoding", default=True, help="display a square of a given number")

    parser.add_argument("--dense_dim_auxilliary_pos", default=None, help="display a square of a given number")
    parser.add_argument("--dense_dim_auxilliary_pos_2", default=None, help="display a square of a given number")
    parser.add_argument("--activation_char_decoder", default=None, help="display a square of a given number")
    parser.add_argument("--activation_word_decoder", default=None, help="display a square of a given number")
    #parser.add_argument("--tasks", default=["normalize"], help="display a square of a given number")
    parser.add_argument('--tasks', nargs='+', help='<Required> Set flag', default=["normalize"])

    parser.add_argument("--train_path", required=mode == "command_line", help="display a square of a given number")
    parser.add_argument("--dev_path", required=mode == "command_line", help="display a square of a given number")
    parser.add_argument("--test_paths", default=None, help="display a square of a given number")

    parser.add_argument("--pos_specific_path", default=None, type=str, help="display a square of a given number")
    parser.add_argument("--expand_vocab_dev_test", default=True, help="display a square of a given number")

    parser.add_argument("--model_id_pref", required=mode == "command_line", help="display a square of a given number")
    parser.add_argument("--overall_label", default="DEFAULT", help="display a square of a given number")
    parser.add_argument("--overall_report_dir", required=mode == "command_line", help="display a square of a given number")
    parser.add_argument("--debug", action="store_true", help="display a square of a given number")
    parser.add_argument("--warmup", action="store_true", help="display a square of a given number")
    parser.add_argument("--verbose", default=1, help="display a square of a given number")

    args = parser.parse_args()


    return args
