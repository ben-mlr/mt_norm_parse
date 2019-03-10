
import os
print("CONDA : ",os.environ["CONDA_PREFIX"])

import sys
from env.project_variables import PROJECT_PATH, TRAINING,LIU_TRAIN, DEMO_SENT, CP_WR_PASTE_TEST_269, \
    LIU_DEV, DEV, DIR_TWEET_W2V, TEST, DIR_TWEET_W2V, CHECKPOINT_DIR, DEMO, DEMO2, CP_PASTE_WR_TRAIN, \
    CP_WR_PASTE_DEV, CP_WR_PASTE_TEST, CP_PASTE_DEV, CP_PASTE_TRAIN, CP_PASTE_TEST, EWT_DEV, EWT_TEST, \
    LIU_DEV_SENT, LIU_TRAIN_SENT, DEV_SENT, TEST_SENT, DEMO_SENT, TRAINING_DEMO, EN_LINES_EWT_TRAIN, EN_LINES_DEV, EN_LINES_EWT_TRAIN, \
    MTNT_TOK_TRAIN, MTNT_TOK_DEV, MTNT_EN_FR_TRAIN, MTNT_EN_FR_DEV, MTNT_EN_FR_TEST
from training.args_tool import args_train

from training.train_eval import train_eval


if __name__ == "__main__":
    

    #from tqdm import tqdm
    args = args_train()
    params = vars(args)
    print("PARAMS CHECK", params, args.word_embed)
    print("TASKS", params["tasks"])
    train_eval(args=params,
               model_id_pref=args.model_id_pref,
               train_path=args.train_path,
               dev_path=args.dev_path,
               expand_vocab_dev_test=args.expand_vocab_dev_test,
               pos_specific_path=args.pos_specific_path,
               overall_label=args.overall_label, overall_report_dir=args.overall_report_dir,
               extend_n_batch=2,
               get_batch_mode_all=True, compute_mean_score_per_sent=False, bucketing_train=True, freq_checkpointing=1,
               symbolic_end=True, symbolic_root=True, freq_writer=1, compute_scoring_curve=False,
               gpu=args.gpu,
               verbose=args.verbose, warmup=args.warmup, debug=args.debug)
