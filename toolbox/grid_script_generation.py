
from env.project_variables import PROJECT_PATH
from toolbox.grid_tool import grid_param_label_generate
import os

from env.project_variables import PROJECT_PATH, TRAINING,LIU_TRAIN, DEMO_SENT, CP_WR_PASTE_TEST_269, \
    LIU_DEV, DEV, DIR_TWEET_W2V, TEST, DIR_TWEET_W2V, CHECKPOINT_DIR, DEMO, DEMO2, CP_PASTE_WR_TRAIN, \
    CP_WR_PASTE_DEV, CP_WR_PASTE_TEST, CP_PASTE_DEV, CP_PASTE_TRAIN, CP_PASTE_TEST, EWT_DEV, EWT_TEST, \
    LIU_DEV_SENT, LIU_TRAIN_SENT, DEV_SENT, TEST_SENT, DEMO_SENT, TRAINING_DEMO, EN_LINES_EWT_TRAIN, EN_LINES_DEV, EN_LINES_EWT_TRAIN, \
    MTNT_TOK_TRAIN, MTNT_TOK_DEV, MTNT_EN_FR_TRAIN, MTNT_EN_FR_DEV, MTNT_EN_FR_TEST

params_dozat = {"hidden_size_encoder": 200, "output_dim": 100, "char_embedding_dim": 100,
                "dropout_sent_encoder": 0.5, "drop_out_word_encoder": 0.5, "dropout_word_decoder": 0.3,
                "drop_out_word_encoder_out": 0.5, "drop_out_sent_encoder_out": 0.0,
                "drop_out_char_embedding_decoder": 0.1, "dropout_bridge": 0.5,
                "n_layers_word_encoder": 1, "dir_sent_encoder": 2, "word_recurrent_cell_decoder": "LSTM",
                "word_recurrent_cell_encoder": "LSTM",
                "hidden_size_sent_encoder": 200, "hidden_size_decoder": 100, "batch_size": 500}


def script_generation(grid_label, init_param, warmup,
                      train_path, dev_path, test_paths, overall_report_dir, overall_label,model_id_pref,
                      stable_decoding_state_ls, word_decoding_ls, batch_size_ls,
                      word_embed_ls, dir_sent_encoder_ls, lr_ls, word_embed_init_ls,
                      teacher_force_ls, proportion_pred_train_ls, shared_context_ls,
                      word_embedding_projected_dim_ls,
                      tasks_ls, char_src_attention_ls,
                      n_layers_sent_cell_ls, unrolling_word_ls,
                      scale_ls, pos_specific_path=None):
    params, labels, default_all, analysed, fixed = grid_param_label_generate(
        init_param,
        warmup=warmup,
        grid_label=grid_label,
        stable_decoding_state_ls=stable_decoding_state_ls,
        word_decoding_ls=word_decoding_ls,
        batch_size_ls=batch_size_ls,
        word_embed_ls=word_embed_ls,
        dir_sent_encoder_ls=dir_sent_encoder_ls, lr_ls=lr_ls,
        word_embed_init_ls=word_embed_init_ls,
        teacher_force_ls=teacher_force_ls,
        proportion_pred_train_ls=proportion_pred_train_ls,
        shared_context_ls=shared_context_ls,
        word_embedding_projected_dim_ls=word_embedding_projected_dim_ls,
        tasks_ls=tasks_ls,
        char_src_attention_ls=char_src_attention_ls,
        n_layers_sent_cell_ls=n_layers_sent_cell_ls,
        unrolling_word_ls=unrolling_word_ls,
        scale_ls=scale_ls )

    LIST_ARGS = ["tasks"]
    print(params, labels)
    for param, model_id_pref in zip(params, labels):
        script = "python {}".format(os.path.join(PROJECT_PATH, "train_evaluate_run.py"))
        for arg, val in param.items():
            if arg not in LIST_ARGS:
                script+=" --{} {}".format(arg,val)
            else:
                script += " --{} {}".format(arg, " ".join(val))
            script += " --{} {}".format("train_path", train_path)
            script += " --{} {}".format("dev_path", dev_path)
            script += " --{} {}".format("test_path", test_paths)
            script += " --{} {}".format("pos_specific_path", pos_specific_path)
            script += " --{} {}".format("overall_label", overall_label)
            script += " --{} {}".format("model_id_pref", model_id_pref)
            script += " --{} {}".format("overall_report_dir", overall_report_dir)
        print(script)


if __name__ == "__main__":

    script_generation(grid_label="0", init_param=params_dozat, warmup=False,
                      stable_decoding_state_ls=[False],
                      word_decoding_ls=[False],
                      batch_size_ls=[50, 100, 200, 400],
                      word_embed_ls=[False],
                      dir_sent_encoder_ls=[2], lr_ls=[0],
                      word_embed_init_ls=[None],
                      teacher_force_ls=[True],
                      proportion_pred_train_ls=[None],
                      shared_context_ls=["all"],
                      word_embedding_projected_dim_ls=[None],
                      tasks_ls=[["pos"]],
                      char_src_attention_ls=[True],
                      n_layers_sent_cell_ls=[2],
                      unrolling_word_ls=[True],
                      scale_ls=[1],
                      overall_report_dir="./", overall_label="test_label", model_id_pref="",
                      train_path=DEV, dev_path=TEST, test_paths=None)

# default not used but could be

# HANDLE SPECIFIC BEHAVIOR
## Default behaviorn : ex None for path , behavior for test_paths as list
## bug quick fix

# to add : hardwards : gpu force true, cpu or gpu number
# write to script as list of scripts

# Task farm script that call this grid_script_generation , that call tracking : google sheet then should fit in the pipeline
# test on rioc
# test on Neff
