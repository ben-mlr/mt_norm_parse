
# ALLOWED PARAMETERS

AVAILABLE_TRAINING_EVAL_SCRIPT = ["train_evaluate_run", "train_evaluate_bert_normalizer"]
# 1 means all accepted
ARGUMENT_AVAILABLE_ALL = ["train_path", "dev_path", "test_path", "tasks"]
ARGUMENT_AVAILABLE_BERT = ["batch_size", "lr", "initialize_bpe_layer", "freeze_parameters",  "freeze_layer_prefix_ls",
                           "bert_model", "dropout_classifier", "fine_tuning_strategy", "dropout_input_bpe",
                           "checkpoint_dir", "norm_2_noise_training", "bert_module","append_n_mask",
                           "tasks", "masking_strategy",# "portion_mask",
                           "heuristic_ls", "gold_error_detection", "dropout_bert", "aggregating_bert_layer_mode",
                           "layer_wise_attention", "tokenize_and_bpe", "multi_task_loss_ponderation"]
ARGUMENT_AVAILABLE_BERT.extend(ARGUMENT_AVAILABLE_ALL)

ARGUMENT_AVAIALBLE_SEQ2SEQ = [1]

ARGS_AVAILABLE_PER_MODEL = {"train_evaluate_run": ARGUMENT_AVAIALBLE_SEQ2SEQ,
                            "train_evaluate_bert_normalizer": ARGUMENT_AVAILABLE_BERT}


# PARAMETERS
## DEFAULT FOR SEQ2SEQ
DEFAULT_LR = 0.001
DEFAULT_WORD_EMBED_INIT = None
DEFAULT_SHARED_CONTEXT = "all"
DEFAULT_DIR_WORD_ENCODER = 2
DEFAULT_CHAR_SRC_ATTENTION = 1
DEFAULT_DIR_SENT_ENCODER = 2
DEFAULT_CLIPPING = 1
DEFAULT_WORD_UNROLLING = 1
DEFAULT_TEACHER_FORCE = 1
DEFAULT_WORD_DECODING = 0
DEFAULT_STABLE_DECODING = 0
DEFAULT_WORD_EMBEDDING_PROJECTED = None
DEFAULT_PROPORTION_PRED_TRAIN = None
DEFAULT_LAYERS_SENT_CELL = 1
DEFAULT_WORD_EMBED = 1
DEFAULT_DROPOUT_WORD_ENCODER_CELL = 0.0
DEFAULT_ATTENTION_TAGGING = 0
DEFAULT_LAYER_WORD_ENCODER = 1
DEFAULT_MODE_WORD_ENCODING = "cat"
DEFAULT_CHAR_LEVEL_EMBEDDING_PROJECTION = 0
DEFAULT_WORD_RECURRENT_CELL = "LSTM"

DEFAULT_TASKS = ["normalize"]

DEFAULT_MULTI_TASK_LOSS_PONDERATION = "all"
