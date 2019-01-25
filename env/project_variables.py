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


# DATASETS
TRAINING_LABEL, TRAINING = "en-ud-train", os.path.join(PROJECT_PATH, "../parsing/normpar/data/en-ud-train.conllu")
DEV_LABEL, DEV = "owoputi", os.path.join(PROJECT_PATH, "../parsing/normpar/data/owoputi.integrated_fixed")
TEST_LABEL, TEST = "lexnorm", os.path.join(PROJECT_PATH, "../parsing/normpar/data/lexnorm.integrated")
DEMO_LABEL, DEMO = "lexnorm-Demo", os.path.join(PROJECT_PATH, "../parsing/normpar/data/lexnorm.integrated.demo")
DEMO2_LABEL, DEMO2 = "lexnorm-demo2", os.path.join(PROJECT_PATH, "../parsing/normpar/data/lexnorm.integrated.demo2")
LIU_LABEL, LIU = "liu_data", os.path.join(PROJECT_PATH, "./data/LiLiu/2577_tweets-li.conll")
LEX_TRAIN_LABEL, LEX_TRAIN = "lex_norm2015_train", os.path.join(PROJECT_PATH, "./data/lexnorm2015/train_data.conll")
LEX_TEST_LABEL, LEX_TEST = "lex_norm2015_test", os.path.join(PROJECT_PATH, "./data/lexnorm2015/test_truth.conll")

LEX_LIU_LABEL,LEX_LIU_TRAIN = "lex_train+liu", os.path.join(PROJECT_PATH, "./data/lexnorm2015/lex_norm_train+liu_2577.conll")

REPO_DATASET = {TRAINING: TRAINING_LABEL, DEV: DEV_LABEL, DEMO: DEMO_LABEL, DEMO2: DEMO2_LABEL,
                TEST: TEST_LABEL, LIU: LIU_LABEL,
                LEX_TRAIN:LEX_TRAIN_LABEL,
                LEX_TEST:LEX_TEST_LABEL,
                LEX_LIU_TRAIN:LEX_LIU_LABEL }