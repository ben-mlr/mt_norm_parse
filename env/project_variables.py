import os

# SEEDS
SEED_NP = 123+1
SEED_TORCH = 123

# ENVIRONMENT VARIABLES
PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
RUN_SCRIPTS_DIR = os.path.join(PROJECT_PATH, "run_scripts")

# MODELS
# checkpoint dir if not checkpoint_dir as defined in args.json not found
CHECKPOINT_DIR = os.path.join(PROJECT_PATH, "checkpoints")

CLIENT_GOOGLE_CLOUD = os.path.join(PROJECT_PATH, "tracking/google_api")
SHEET_NAME_DEFAULT, TAB_NAME_DEFAULT = "model_evaluation", "experiments_tracking"

LIST_ARGS = ["tasks"]
NONE_ARGS = ["gpu"]
BOOL_ARGS = ["word_embed", "teacher_force", "char_decoding", "unrolling_word", "init_context_decoder",
             "word_decoding", "stable_decoding_state", "char_src_attention"]
DIC_ARGS = ["multi_task_loss_ponderation"]
GPU_AVAILABLE_DEFAULT_LS = ["0", "1", "2", "3"]

# architecture/model/training supported
SUPPORED_WORD_ENCODER = ["LSTM", "GRU", "WeightDropLSTM"]
BREAKING_NO_DECREASE = 21
SUPPORTED_STAT = ["sum"]
LOSS_DETAIL_TEMPLATE = {"loss_overall": 0, "loss_seq_prediction": 0, "other": {}}
LOSS_DETAIL_TEMPLATE_LS = {"loss_overall": [], "loss_seq_prediction": [], "other": {}}
SCORE_AUX = ["norm_not_norm-F1", "norm_not_norm-Precision", "norm_not_norm-Recall", "norm_not_norm-accuracy"]
AVAILABLE_TASKS = ["all", "normalize", "norm_not_norm", "pos"]
TASKS_2_METRICS_STR = {"all": ["accuracy-normalization","InV-accuracy-normalization","OOV-accuracy-normalization","npv-normalization","recall-normalization","precision-normalization","tnr-normalization","accuracy-pos"],
                       "normalize": ["accuracy-normalization","InV-accuracy-normalization","OOV-accuracy-normalization","npv-normalization","recall-normalization","precision-normalization","tnr-normalization"],
                       "pos": ["accuracy-pos"],
                       "norm_not_norm": ["IoU-pred-normed","recall-norm_not_norm","accuracy-norm_not_norm","IoU-pred-need_norm","precision-norm_not_norm"]}

AVAILABLE_OPTIMIZER = ["adam", "bahdanu-adadelta"]
MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE = ["uniform", "normalization_100","pos_100","all","pos","normalize","norm_not_norm"]
DEFAULT_SCORING_FUNCTION = "exact_match"

# DATASETSproportion_pred_train_ls
TRAINING_LABEL, TRAINING = "en-ud-train", os.path.join(PROJECT_PATH, "../parsing/normpar/data/en-ud-train.conllu")
TRAINING_DEMO_LABEL, TRAINING_DEMO = "en-ud-train_demo", os.path.join(PROJECT_PATH, "../parsing/normpar/data/en-ud-train_demo.conllu")

EWT_TEST_LABEL, EWT_TEST = "ewt_test", os.path.join(PROJECT_PATH, "../parsing/normpar/data/en-ud-test.conllu")

EWT_PRED_TOKEN_UDPIPE_LABEL, EWT_PRED_TOKEN_UDPIPE = "ud_pred_tokens-ewt_dev",os.path.join(PROJECT_PATH, "data", "udpipe_pred_tokens", "en_ewt-udpipe.conllu")

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
DIR_FASTEXT_WIKI_NEWS_W2V = os.path.join(PROJECT_PATH, "w2v", "wiki-news-300d-1M.vec")
FASTEXT_WIKI_NEWS_W2V_LABEL = "wiki-news"
TWEET_W2V_LABEL = "tweets_en_w2v"


W2V_LOADED_DIM = 400
MAX_VOCABULARY_SIZE_WORD_DIC = 20000

# Michel
MTNT_TOK_TRAIN_LABEL, MTNT_TOK_TRAIN= "mtnt_tok_train", os.path.join(PROJECT_PATH, "./data/MTNT/monolingual/train.en.raw2tok.conll")
MTNT_TOK_DEV_LABEL, MTNT_TOK_DEV= "mtnt_tok_dev", os.path.join(PROJECT_PATH, "./data/MTNT/monolingual/dev.en.raw2tok.conll")

MTNT_EN_FR_TRAIN_LABEL, MTNT_EN_FR_TRAIN = "mtnt_train", os.path.join(PROJECT_PATH, "./data/MTNT/train/train.en-fr.conll")
MTNT_EN_FR_DEV_LABEL, MTNT_EN_FR_DEV = "mtnt_valid", os.path.join(PROJECT_PATH, "./data/MTNT/valid/valid.en-fr.conll")
MTNT_EN_FR_TEST_LABEL, MTNT_EN_FR_TEST = "mtnt_test", os.path.join(PROJECT_PATH, "./data/MTNT/test/test.en-fr.conll")
MTNT_EN_FR_TEST_DEMO_LABEL, MTNT_EN_FR_TEST_DEMO = "mtnt_test.demo", os.path.join(PROJECT_PATH, "./data/MTNT/test/test.en-fr.demo.conll")



# EN
EN_LINES_TRAIN_LABEL, EN_LINES_TRAIN = "en_lines_train", os.path.join(PROJECT_PATH, "../parsing/normpar/data/en_lines-ud-train.conllu")
EN_LINES_EWT_TRAIN_LABEL, EN_LINES_EWT_TRAIN = "en_lines_ewt_train", os.path.join(PROJECT_PATH, "../parsing/normpar/data/en_lines+ewt-ud-train.conllu")

EN_LINES_DEV_LABEL, EN_LINES_DEV = "en_lines_dev", os.path.join(PROJECT_PATH, "../parsing/normpar/data/en_lines-ud-dev.conllu")

# SENT conll like
DEMO_SENT_LABEL, DEMO_SENT = "lexnorm-Demo_sent", os.path.join(PROJECT_PATH, "../parsing/normpar/data/char_test.demo")
TEST_SENT_LABEL, TEST_SENT = "lexnorm_sent", os.path.join(PROJECT_PATH, "../parsing/normpar/data/lexnorm-sent.conll")
DEV_SENT_LABEL, DEV_SENT = "owoputi_sent", os.path.join(PROJECT_PATH, "../parsing/normpar/data/owoputi-sent.conll")
LIU_TRAIN_SENT_LABEL, LIU_TRAIN_SENT = "liu_train_sent" , os.path.join(PROJECT_PATH, "./data/LiLiu/2577_tweets-li-sent-train_2009.conll")
LIU_DEV_SENT_LABEL, LIU_DEV_SENT = "liu_dev_sent", os.path.join(PROJECT_PATH, "./data/LiLiu/2577_tweets-li-sent-dev_500.conll")


REPO_DATASET = {TRAINING: TRAINING_LABEL, DEV: DEV_LABEL, DEMO: DEMO_LABEL, DEMO2: DEMO2_LABEL,TRAINING_DEMO:TRAINING_DEMO_LABEL,
                TEST: TEST_LABEL, LIU: LIU_LABEL, EWT_TEST: EWT_TEST_LABEL,
                LEX_TRAIN:LEX_TRAIN_LABEL,
                LEX_TEST:LEX_TEST_LABEL,
                LEX_LIU_TRAIN:LEX_LIU_LABEL,
                LIU_DEV: LIU_DEV_LABEL, LIU_TRAIN:LIU_TRAIN_LABEL,
                EWT_DEV: EWT_DEV_LABEL,
                CP_PASTE_TRAIN: CP_PASTE_TRAIN_LABEL, CP_PASTE_DEV: CP_PASTE_DEV_LABEL, CP_PASTE_TEST: CP_PASTE_TEST_LABEL,
                CP_PASTE_WR_TRAIN: CP_PASTE_WR_TRAIN_LABEL, CP_WR_PASTE_DEV: CP_PASTE_WR_DEV_LABEL, CP_WR_PASTE_TEST: CP_PASTE_WR_TEST_LABEL, CP_WR_PASTE_TEST_269: CP_PASTE_WR_TEST_269_LABEL,
                DEMO_SENT: DEMO_SENT_LABEL,
                TEST_SENT: TEST_SENT_LABEL, DEV_SENT: DEV_SENT_LABEL, LIU_TRAIN_SENT: LIU_TRAIN_SENT_LABEL,
                LIU_DEV_SENT: LIU_DEV_SENT_LABEL,
                EN_LINES_TRAIN: EN_LINES_TRAIN_LABEL, EN_LINES_DEV: EN_LINES_DEV_LABEL, EN_LINES_EWT_TRAIN: EN_LINES_EWT_TRAIN_LABEL,
                MTNT_TOK_TRAIN: MTNT_TOK_TRAIN_LABEL, MTNT_TOK_DEV: MTNT_TOK_DEV_LABEL, MTNT_EN_FR_TRAIN: MTNT_EN_FR_TRAIN_LABEL,
                MTNT_EN_FR_DEV: MTNT_EN_FR_DEV_LABEL, MTNT_EN_FR_TEST: MTNT_EN_FR_TEST_LABEL,
                DIR_TWEET_W2V: TWEET_W2V_LABEL, FASTEXT_WIKI_NEWS_W2V_LABEL: DIR_FASTEXT_WIKI_NEWS_W2V,
                MTNT_EN_FR_TEST_DEMO: MTNT_EN_FR_TEST_DEMO_LABEL,
                EWT_PRED_TOKEN_UDPIPE:EWT_PRED_TOKEN_UDPIPE_LABEL

                }


REPO_W2V = {DIR_TWEET_W2V: {"label": TWEET_W2V_LABEL, "dim": 400},
            DIR_FASTEXT_WIKI_NEWS_W2V: {"label": FASTEXT_WIKI_NEWS_W2V_LABEL, "dim": 300},
            None: {"label": "random_init", "dim": -1}
}

# output dir for writing
WRITING_DIR = os.path.join(PROJECT_PATH, "predictions")

WARMUP_N_EPOCHS = 30