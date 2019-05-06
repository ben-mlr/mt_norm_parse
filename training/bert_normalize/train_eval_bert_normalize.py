from env.importing import os
from training.bert_normalize.fine_tune_bert import run
from model.bert_normalize import get_bert_token_classification
from io_.dat.constants import TOKEN_BPE_BERT_START, TOKEN_BPE_BERT_SEP, NULL_STR, PAD_BERT, PAD_ID_BERT
from env.models_dir import BERT_MODEL_DIC
from io_.info_print import printing

from toolbox.bert_tools.get_bert_info import get_bert_name


def train_eval_bert_normalize(args, verbose=1):

    tasks = ["normalize"]

    args.bert_model = get_bert_name(args.bert_model)
    voc_tokenizer = BERT_MODEL_DIC[args.bert_model]["vocab"]
    model_dir = BERT_MODEL_DIC[args.bert_model]["model"]
    vocab_size = BERT_MODEL_DIC[args.bert_model]["vocab_size"]

    initialize_bpe_layer = args.initialize_bpe_layer
    freeze_parameters = args.freeze_parameters
    freeze_layer_prefix_ls = args.freeze_layer_prefix_ls
    # ["bert"]
    model = get_bert_token_classification(pretrained_model_dir=model_dir,
                                          vocab_size=vocab_size,
                                          freeze_parameters=freeze_parameters,
                                          freeze_layer_prefix_ls=freeze_layer_prefix_ls,
                                          dropout_classifier=args.dropout_classifier,
                                          initialize_bpe_layer=initialize_bpe_layer)

    lr = args.lr
    batch_size = args.batch_size
    null_token_index = BERT_MODEL_DIC["bert-cased"]["vocab_size"]  # based on bert cased vocabulary
    description = "grid"
    dir_grid = args.overall_report_dir
    run(bert_with_classifier=model,
        voc_tokenizer=voc_tokenizer, tasks=tasks, train_path=args.train_path, dev_path=args.dev_path,
        auxilliary_task_norm_not_norm=True,
        saving_every_epoch=10, lr=lr,
        batch_size=batch_size, n_iter_max_per_epoch=10000, n_epoch=args.epochs,
        test_path_ls=args.test_paths,
        description=description, null_token_index=null_token_index, null_str=NULL_STR,
        model_suffix="{}".format(args.model_id_pref), debug=False,
        freeze_parameters=freeze_parameters, freeze_layer_prefix_ls=freeze_layer_prefix_ls, bert_model=args.bert_model,
        initialize_bpe_layer=initialize_bpe_layer, report_full_path_shared=dir_grid, shared_id=args.overall_label,
        fine_tuning_strategy=args.fine_tuning_strategy,
        args=args,
        report=True, verbose=1)

    printing("MODEL {} trained and evaluated", var=[args.model_id_pref], verbose_level=1, verbose=verbose)
