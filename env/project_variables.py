import os

# SEEDS
SEED_NP = 123+1
SEED_TORCH = 123

# ENVIRONMENT VARIABLES
PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")


# MODELS
# checkpoint dir if not checkpoint_dir as defined in args.json not found
CHECKPOINT_DIR = os.path.join(PROJECT_PATH, "checkpoints")

# architecture/model/training supported
SUPPORED_WORD_ENCODER = ["LSTM", "GRU"]
BREAKING_NO_DECREASE = 21
SUPPORTED_STAT = ["sum"]
LOSS_DETAIL_TEMPLATE = {"loss_overall": 0, "loss_seq_prediction": 0, "other": {}}
LOSS_DETAIL_TEMPLATE_LS = {"loss_overall": [], "loss_seq_prediction": [], "other": {}}
SCORE_AUX = ["norm_not_norm-F1", "norm_not_norm-Precision", "norm_not_norm-Recall", "norm_not_norm-accuracy"]
AVAILABLE_TASKS = ["all", "normalize", "norm_not_norm","pos"]



# DATASETS
TRAINING_LABEL, TRAINING = "en-ud-train", os.path.join(PROJECT_PATH, "../parsing/normpar/data/en-ud-train.conllu")

EWT_DEV_LABEL, EWT_DEV = "ewt_dev", os.path.join(PROJECT_PATH, "../parsing/normpar/data/en-ud-dev.integrated-po_as_norm")
DEV_LABEL, DEV = "owoputi", os.path.join(PROJECT_PATH, "../parsing/normpar/data/owoputi.integrated_fixed")
TEST_LABEL, TEST = "lexnorm", os.path.join(PROJECT_PATH, "../parsing/normpar/data/lexnorm.integrated")


DEMO_LABEL, DEMO = "lexnorm-Demo", os.path.join(PROJECT_PATH, "../parsing/normpar/data/lexnorm.integrated.demo")
DEMO2_LABEL, DEMO2 = "lexnorm-demo2", os.path.join(PROJECT_PATH, "../parsing/normpar/data/lexnorm.integrated.demo2")

LIU_LABEL, LIU = "liu_data", os.path.join(PROJECT_PATH, "./data/LiLiu/2577_tweets-li.conll")
LIU_DEV_LABEL, LIU_DEV = "liu_dev", os.path.join(PROJECT_PATH, "./data/LiLiu/2577_tweets-li-dev_500.conll")
LIU_TRAIN_LABEL, LIU_TRAIN = "liu_train", os.path.join(PROJECT_PATH, "./data/LiLiu/2577_tweets-li-train_2009.conll")


LEX_TRAIN_LABEL, LEX_TRAIN = "lex_norm2015_train", os.path.join(PROJECT_PATH, "./data/lexnorm2015/train_data.conll")
LEX_TEST_LABEL, LEX_TEST = "lex_norm2015_test", os.path.join(PROJECT_PATH, "./data/lexnorm2015/test_truth.conll")

LEX_LIU_LABEL, LEX_LIU_TRAIN = "lex_train+liu", os.path.join(PROJECT_PATH, "./data/lexnorm2015/lex_norm_train+liu_2577.conll")

CP_PASTE_TRAIN_LABEL, CP_PASTE_TRAIN = "copy_paste-train", os.path.join(PROJECT_PATH, "./data/copy_paste_train.conll")
CP_PASTE_DEV_LABEL, CP_PASTE_DEV = "copy_paste-dev", os.path.join(PROJECT_PATH, "./data/copy_paste_dev.conll")
CP_PASTE_TEST_LABEL, CP_PASTE_TEST = "copy_paste-test", os.path.join(PROJECT_PATH, "./data/copy_paste_test.conll")

CP_PASTE_WR_TRAIN_LABEL, CP_PASTE_WR_TRAIN = "copy_paste_real_word-train", os.path.join(PROJECT_PATH, "./data/copy_paste_real_word_train.conll")
CP_PASTE_WR_DEV_LABEL, CP_WR_PASTE_DEV = "copy_paste_real_word-dev", os.path.join(PROJECT_PATH, "./data/copy_paste_real_word_dev.conll")
CP_PASTE_WR_TEST_LABEL, CP_WR_PASTE_TEST = "copy_paste_real_word-test", os.path.join(PROJECT_PATH, "./data/copy_paste_real_word_test.conll")

CP_PASTE_WR_TEST_269_LABEL, CP_WR_PASTE_TEST_269 = "copy_paste_real_word_test-first269", os.path.join(PROJECT_PATH, "./data/copy_paste_real_word_test-first269.conll")


DIR_TWEET_W2V = os.path.join(PROJECT_PATH, "w2v", "tweets.en.w2v.txt")
W2V_LOADED_DIM = 400
MAX_VOCABULARY_SIZE_WORD_DIC = 20000


# SENT conll like
DEMO_SENT_LABEL, DEMO_SENT = "lexnorm-Demo_sent", os.path.join(PROJECT_PATH, "../parsing/normpar/data/char_test.demo")
TEST_SENT_LABEL, TEST_SENT = "lexnorm_sent", os.path.join(PROJECT_PATH, "../parsing/normpar/data/lexnorm-sent.conll")
DEV_SENT_LABEL, DEV_SENT = "owoputi_sent", os.path.join(PROJECT_PATH, "../parsing/normpar/data/owoputi-sent.conll")
LIU_TRAIN_SENT_LABEL, LIU_TRAIN_SENT = "liu_train_sent" , os.path.join(PROJECT_PATH, "./data/LiLiu/2577_tweets-li-sent-train_2009.conll")
LIU_DEV_SENT_LABEL, LIU_DEV_SENT = "liu_dev_sent" , os.path.join(PROJECT_PATH, "./data/LiLiu/2577_tweets-li-sent-dev_500.conll")


REPO_DATASET = {TRAINING: TRAINING_LABEL, DEV: DEV_LABEL, DEMO: DEMO_LABEL, DEMO2: DEMO2_LABEL,
                TEST: TEST_LABEL, LIU: LIU_LABEL,
                LEX_TRAIN:LEX_TRAIN_LABEL,
                LEX_TEST:LEX_TEST_LABEL,
                LEX_LIU_TRAIN:LEX_LIU_LABEL,
                LIU_DEV: LIU_DEV_LABEL, LIU_TRAIN:LIU_TRAIN_LABEL,
                EWT_DEV: EWT_DEV_LABEL,
                CP_PASTE_TRAIN: CP_PASTE_TRAIN_LABEL, CP_PASTE_DEV: CP_PASTE_DEV_LABEL, CP_PASTE_TEST: CP_PASTE_TEST_LABEL,
                CP_PASTE_WR_TRAIN: CP_PASTE_WR_TRAIN_LABEL, CP_WR_PASTE_DEV: CP_PASTE_WR_DEV_LABEL, CP_WR_PASTE_TEST: CP_PASTE_WR_TEST_LABEL, CP_WR_PASTE_TEST_269: CP_PASTE_WR_TEST_269_LABEL,
                DEMO_SENT: DEMO_SENT_LABEL,
                TEST_SENT: TEST_SENT_LABEL, DEV_SENT: DEV_SENT_LABEL, LIU_TRAIN_SENT: LIU_TRAIN_SENT_LABEL,
                LIU_DEV_SENT: LIU_DEV_SENT_LABEL
                }
# output dir for writing
WRITING_DIR = os.path.join(PROJECT_PATH, "predictions")