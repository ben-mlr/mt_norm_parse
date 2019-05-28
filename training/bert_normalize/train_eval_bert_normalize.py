from env.importing import os, nn, json, OrderedDict, pickle
from training.bert_normalize.fine_tune_bert import run
from env.project_variables import PROJECT_PATH
from model.bert_normalize import get_bert_token_classification
from io_.dat.constants import TOKEN_BPE_BERT_START, TOKEN_BPE_BERT_SEP, NULL_STR, PAD_BERT, PAD_ID_BERT, SPECIAL_TOKEN_LS
from env.models_dir import BERT_MODEL_DIC
from io_.info_print import printing

from toolbox.bert_tools.get_bert_info import get_bert_name


def train_eval_bert_normalize(args, verbose=1):

    #tasks = ["normalize"]

    args.bert_model = get_bert_name(args.bert_model)

    voc_tokenizer = BERT_MODEL_DIC[args.bert_model]["vocab"]
    model_dir = BERT_MODEL_DIC[args.bert_model]["model"]
    vocab_size = BERT_MODEL_DIC[args.bert_model]["vocab_size"]

    initialize_bpe_layer = args.initialize_bpe_layer
    freeze_parameters = args.freeze_parameters
    freeze_layer_prefix_ls = args.freeze_layer_prefix_ls
    # ["bert"]
    voc_pos_size = 21
    printing("MODEL : voc_pos_size hardcoded to {}", var=voc_pos_size, verbose_level=1, verbose=verbose)

    if args.checkpoint_dir is None:
        # TODO vocab_size should be loaded from args.json
        model = get_bert_token_classification(pretrained_model_dir=model_dir,
                                              vocab_size=vocab_size,
                                              freeze_parameters=freeze_parameters,
                                              freeze_layer_prefix_ls=freeze_layer_prefix_ls,
                                              dropout_classifier=args.dropout_classifier,
                                              dropout_bert=args.dropout_bert,
                                              tasks=args.tasks,
                                              voc_pos_size=voc_pos_size,
                                              bert_module=args.bert_module,layer_wise_attention=args.layer_wise_attention,
                                              initialize_bpe_layer=initialize_bpe_layer)
    else:
        printing("MODEL : reloading from checkpoint {}", var=[args.checkpoint_dir], verbose_level=1, verbose=verbose)
        # TODO args.original_task  , vocab_size is it necessary
        #assert args.original_task is not None
        original_task = ["normalize"]
        model = get_bert_token_classification(vocab_size=vocab_size, voc_pos_size=voc_pos_size,
                                              tasks=original_task,
                                              initialize_bpe_layer=None, bert_module=args.bert_module,
                                              checkpoint_dir=args.checkpoint_dir)

        add_task_2 = True
        if add_task_2:
            printing("MODEL : adding extra classifer for task_2  with {} label", var=[voc_pos_size],
                     verbose=verbose, verbose_level=1)
            model.classifier_task_2 = nn.Linear(model.bert.config.hidden_size, voc_pos_size)
            model.num_labels_2 = voc_pos_size

    lr = args.lr
    batch_size = args.batch_size
    null_token_index = BERT_MODEL_DIC["bert-cased"]["vocab_size"]  # based on bert cased vocabulary
    description = "grid"
    dir_grid = args.overall_report_dir

    #list_reference_heuristic_test = list(json.load(open(os.path.join(PROJECT_PATH, "./data/words_dictionary.json"),"r"),object_pairs_hook=OrderedDict).keys())
    list_reference_heuristic_test = pickle.load(open(os.path.join(PROJECT_PATH, "./data/wiki-news-FAIR-SG-top50000.pkl"), "rb"))
    slang_dic = json.load(open(os.path.join(PROJECT_PATH, "./data/urban_dic_abbreviations.json"), "r"))

    run(bert_with_classifier=model, 
        voc_tokenizer=voc_tokenizer, tasks=args.tasks, train_path=args.train_path, dev_path=args.dev_path,
        auxilliary_task_norm_not_norm=True,
        saving_every_epoch=10, lr=lr,
        batch_size=batch_size, n_iter_max_per_epoch=200000, n_epoch=args.epochs,
        test_path_ls=args.test_paths,
        description=description, null_token_index=null_token_index, null_str=NULL_STR,
        model_suffix="{}".format(args.model_id_pref), debug=False,
        freeze_parameters=freeze_parameters, freeze_layer_prefix_ls=freeze_layer_prefix_ls, bert_model=args.bert_model,
        initialize_bpe_layer=initialize_bpe_layer, report_full_path_shared=dir_grid, shared_id=args.overall_label,
        fine_tuning_strategy=args.fine_tuning_strategy,
        heuristic_ls=args.heuristic_ls, gold_error_detection=args.gold_error_detection,
        args=args, dropout_input_bpe=args.dropout_input_bpe,
        portion_mask=args.portion_mask, masking_strategy=args.masking_strategy,
        norm_2_noise_training=args.norm_2_noise_training,
        random_iterator_train=True,  bucket_test=False,
        compute_intersection_score_test=True,
        aggregating_bert_layer_mode=args.aggregating_bert_layer_mode,
        bert_module=args.bert_module,tokenize_and_bpe=False,
        list_reference_heuristic_test=list_reference_heuristic_test, case="lower",
        layer_wise_attention=args.layer_wise_attention,
        slang_dic_test=slang_dic,
        report=True, verbose=1)

    printing("MODEL {} trained and evaluated", var=[args.model_id_pref], verbose_level=1, verbose=verbose)
