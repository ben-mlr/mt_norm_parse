import argparse


def args_train(mode="command_line"):

    assert mode in ["command_line", "script"], "mode should be in '[command_line, script]"

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size_encoder", type=int, default=10, help="display a square of a given number")
    parser.add_argument("--hidden_size_sent_encoder", default=10, type=int, help="display a square of a given number")
    parser.add_argument("--hidden_size_decoder", default=10,type=int, help="display a square of a given number")

    parser.add_argument("--word_embed", type=int, default=False, help="display a square of a given number")
    parser.add_argument("--word_embedding_dim", default=10, type=int, help="display a square of a given number")
    parser.add_argument("--word_embedding_projected_dim", default=None, type=int, help="display a square of a given number")
    parser.add_argument("--output_dim", default=10, type=int, help="display a square of a given number")
    parser.add_argument("--char_embedding_dim", default=10, type=int, help="display a square of a given number")

    parser.add_argument("--batch_size", default=2, type=int, help="display a square of a given number")
    parser.add_argument("--epochs", default=1, type=int, help="display a square of a given number")

    parser.add_argument("--dropout_sent_encoder", default=0, type=float, help="display a square of a given number")
    parser.add_argument("--dropout_word_encoder_cell", default=0, type=float, help="display a square of a given number")
    parser.add_argument("--dropout_word_decoder", default=0, type=float, help="display a square of a given number")
    parser.add_argument("--drop_out_word_encoder_out", default=0, type=float, help="display a square of a given number")
    parser.add_argument("--drop_out_sent_encoder_out", default=0, type=float, help="display a square of a given number")
    parser.add_argument("--dropout_bridge", default=0, type=float, help="display a square of a given number")
    parser.add_argument("--drop_out_char_embedding_decoder", default=0, type=float, help="display a square of a given number")

    parser.add_argument("--n_layers_word_encoder", default=1, type=int, help="display a square of a given number")
    parser.add_argument("--n_layers_sent_cell", default=1, type=int,help="display a square of a given number")

    parser.add_argument("--dir_sent_encoder", default=2,type=int, help="display a square of a given number")

    parser.add_argument("--word_recurrent_cell_encoder", default="LSTM", help="display a square of a given number")
    parser.add_argument("--word_recurrent_cell_decoder", default="LSTM", help="display a square of a given number")

    parser.add_argument("--dense_dim_auxilliary", default=None,type=int, help="display a square of a given number")
    parser.add_argument("--dense_dim_auxilliary_2", default=None,type=int, help="displaqy a square of a given number")

    parser.add_argument("--unrolling_word", default=True, type=int,help="display a square of a given number")

    parser.add_argument("--char_src_attention", default=False,type=int, help="display a square of a given number")
    parser.add_argument("--dir_word_encoder", default=1, type=int, help="display a square of a given number")
    parser.add_argument("--weight_binary_loss", default=1, type=float, help="display a square of a given number")
    parser.add_argument("--shared_context", default="all", help="display a square of a given number")

    parser.add_argument("--policy", default=None,type=int, help="display a square of a given number")
    parser.add_argument("--lr", default=0.001, type=float,help="display a square of a given number")
    parser.add_argument("--gradient_clipping", type=int,default=None, help="display a square of a given number")
    parser.add_argument("--teacher_force", type=int,default=1, help="display a square of a given number")
    parser.add_argument("--proportion_pred_train", type=int, default=None, help="display a square of a given number")

    parser.add_argument("--stable_decoding_state", type=int, default=0, help="display a square of a given number")
    parser.add_argument("--init_context_decoder", type=int, default=1, help="display a square of a given number")
    parser.add_argument("--optimizer", default="adam", help="display a square of a given number ")

    parser.add_argument("--word_decoding", default=0, type=int, help="display a square of a given number ")

    parser.add_argument("--attention_tagging", default=0, type=int, help="display a square of a given number ")

    parser.add_argument("--dense_dim_word_pred", type=int, default=None, help="display a square of a given number ")
    parser.add_argument("--dense_dim_word_pred_2", type=int, default=None, help="display a square of a given number ")
    parser.add_argument("--dense_dim_word_pred_3", type=int, default=0, help="display a square of a given number ")

    parser.add_argument("--word_embed_init", default=None,help="display a square of a given number")
    parser.add_argument("--char_decoding", type=int, default=True, help="display a square of a given number")

    parser.add_argument("--dense_dim_auxilliary_pos",type=int, default=None, help="display a square of a given number")
    parser.add_argument("--dense_dim_auxilliary_pos_2",type=int, default=None, help="display a square of a given number")
    parser.add_argument("--activation_char_decoder", default=None, help="display a square of a given number")
    parser.add_argument("--activation_word_decoder", default=None, help="display a square of a given number")
    #parser.add_argument("--tasks", default=["normalize"], help="display a square of a given number")
    parser.add_argument('--tasks', nargs='+', help='<Required> Set flag', default=["normalize"])

    parser.add_argument("--train_path", required=mode == "command_line", help="display a square of a given number")
    parser.add_argument("--dev_path", required=mode == "command_line", help="display a square of a given number")
    parser.add_argument('--test_paths', nargs='+', help='<Required> Set flag', default=None)

    parser.add_argument("--pos_specific_path", default=None, type=str, help="display a square of a given number")
    parser.add_argument("--expand_vocab_dev_test", default=True, help="display a square of a given number")

    parser.add_argument("--model_id_pref", required=mode == "command_line", help="display a square of a given number")
    parser.add_argument("--overall_label", default="DEFAULT", help="display a square of a given number")
    parser.add_argument("--overall_report_dir", required=mode == "command_line", help="display a square of a given number")
    parser.add_argument("--debug", action="store_true", help="display a square of a given number")
    parser.add_argument("--warmup", action="store_true", help="display a square of a given number")
    parser.add_argument("--verbose", default=1, type=int,help="display a square of a given number")

    parser.add_argument("--gpu", default=None, type=str, help="display a square of a given number")

    args = parser.parse_args()
    if not args.word_embed:
        args.word_embedding_dim = 0


    return args
