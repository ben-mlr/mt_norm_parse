import os

# ENVIRONMENT VARIABLES
PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

# MODELS
# checkpoint dir if not checkpoint_dir as defined in args.json not found
CHECKPOINT_DIR = os.path.join(PROJECT_PATH,"checkpoints")

# DATASETS
TRAINING = os.path.join(PROJECT_PATH, "../parsing/normpar/data/en-ud-train.conllu")
DEV = os.path.join(PROJECT_PATH, "../parsing/normpar/data/owoputi.integrated")
TEST = os.path.join(PROJECT_PATH, "../parsing/normpar/data/lexnorm.integrated")
DEMO = os.path.join(PROJECT_PATH, "../parsing/normpar/data/lexnorm.integrated.demo")
DEMO2 = os.path.join(PROJECT_PATH, "../parsing/normpar/data/lexnorm.integrated.demo2")