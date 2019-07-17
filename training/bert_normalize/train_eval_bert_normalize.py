from env.importing import os, nn, json, OrderedDict, pickle, pdb
from training.bert_normalize.fine_tune_bert import run
from env.project_variables import PROJECT_PATH
from model.bert_normalize import get_bert_token_classification, make_bert_multitask
from io_.dat.constants import TOKEN_BPE_BERT_START, TOKEN_BPE_BERT_SEP, NULL_STR, PAD_BERT, PAD_ID_BERT, SPECIAL_TOKEN_LS
from env.models_dir import BERT_MODEL_DIC
from io_.info_print import printing
from toolbox.bert_tools.get_bert_info import get_bert_name
from env.project_variables import MULTITASK_BERT_LABELS_MLM_HEAD, MULTITASK_BERT_LABELS_MLM_HEAD_LOSS


def update_multitask_loss(multi_task_loss_ponderation):
    if multi_task_loss_ponderation is None:
        return None
    multi_task_loss_ponderation_new = {}
    for task, weight in multi_task_loss_ponderation.items():
        if task in MULTITASK_BERT_LABELS_MLM_HEAD:
            multi_task_loss_ponderation_new[MULTITASK_BERT_LABELS_MLM_HEAD_LOSS[task]] = weight
    return multi_task_loss_ponderation_new


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
    voc_pos_size = 19 #18+1 for alg_arabizi # 53+1 for ARABIZI 1# 21 is for ENGLISH
    printing("MODEL : voc_pos_size hardcoded to {}", var=voc_pos_size, verbose_level=1, verbose=verbose)

    debug = False
    if os.environ.get("ENV") in ["rioc", "neff"]:
        debug = False
    if args.checkpoint_dir is None:
        # TODO vocab_size should be loaded from args.json
        # TEMPORARY : should eventually keep only : model = make_bert_multitask()
        if args.multitask:
            model = make_bert_multitask(pretrained_model_dir=model_dir, tasks=["parsing"])
        else:
            model = get_bert_token_classification(pretrained_model_dir=model_dir,
                                              vocab_size=vocab_size,
                                              freeze_parameters=freeze_parameters,
                                              freeze_layer_prefix_ls=freeze_layer_prefix_ls,
                                              dropout_classifier=args.dropout_classifier,
                                              dropout_bert=args.dropout_bert,
                                              tasks=args.tasks,
                                              voc_pos_size=voc_pos_size,
                                              bert_module=args.bert_module,
                                              layer_wise_attention=args.layer_wise_attention,
                                              mask_n_predictor=args.append_n_mask,
                                              initialize_bpe_layer=initialize_bpe_layer,
                                              debug=debug)
    else:
        printing("MODEL : reloading from checkpoint {} all models parameters are ignored except task bert module and layer_wise_attention", var=[args.checkpoint_dir], verbose_level=1, verbose=verbose)
        # TODO args.original_task  , vocab_size is it necessary
        #assert args.original_task is not None
        original_task = ["normalize"]
        print("WARNING : HARDCODED add_task_2_for_downstream : True ")
        model = get_bert_token_classification(vocab_size=vocab_size, voc_pos_size=voc_pos_size,
                                              tasks=original_task,
                                              initialize_bpe_layer=None, bert_module=args.bert_module,
                                              layer_wise_attention=args.layer_wise_attention,
                                              mask_n_predictor=args.append_n_mask,
                                              add_task_2_for_downstream=True,
                                              checkpoint_dir=args.checkpoint_dir, debug=debug)

        add_task_2 = False
        if add_task_2:
            printing("MODEL : adding extra classifer for task_2  with {} label", var=[voc_pos_size],
                     verbose=verbose, verbose_level=1)
            model.classifier_task_2 = nn.Linear(model.bert.config.hidden_size, voc_pos_size)
            model.num_labels_2 = voc_pos_size

    lr = args.lr
    batch_size = args.batch_size
    null_token_index = BERT_MODEL_DIC[args.bert_model]["vocab_size"]  # based on bert cased vocabulary
    description = "grid"
    dir_grid = args.overall_report_dir

    list_reference_heuristic_test = pickle.load(open(os.path.join(PROJECT_PATH,"data/wiki-news-FAIR-SG-top50000.pkl"), "rb"))
    slang_dic = json.load(open(os.path.join(PROJECT_PATH, "data/urban_dic_abbreviations.json"), "r"))

    if "normalize" in args.tasks:
        early_stoppin_metric = "accuracy-exact-normalize"
    elif "pos" in args.tasks:
        early_stoppin_metric = "accuracy-exact-pos"
    else:
        raise(Exception("Neither normalize nor pos is in {} (cant define early_stoppin_metric)".format(args.tasks)))

    printing("INFO : tasks is {} so setting early_stoppin_metric to {} ", var=[args.tasks, early_stoppin_metric], verbose=verbose, verbose_level=1)
    printing("INFO : environ is {} so debug set to {}", var=[os.environ.get("ENV", "Unkwnown"),debug], verbose_level=1, verbose=verbose)

    # MLM in multitas mode is temporary and require task_i indexing : that's why we need to rename ponderation dictionary
    args.multi_task_loss_ponderation = update_multitask_loss(args.multi_task_loss_ponderation)
    run(model=model,
        voc_tokenizer=voc_tokenizer, tasks=args.tasks, train_path=args.train_path, dev_path=args.dev_path,
        append_n_mask=args.append_n_mask,
        auxilliary_task_norm_not_norm=True,
        saving_every_epoch=15, lr=lr, batch_size=batch_size,
        n_iter_max_per_epoch=100000, n_epoch=args.epochs,
        test_path_ls=args.test_paths,
        description=description, null_token_index=null_token_index, null_str=NULL_STR,
        model_suffix="{}".format(args.model_id_pref), debug=debug,
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
        bert_module=args.bert_module, tokenize_and_bpe=args.tokenize_and_bpe,
        list_reference_heuristic_test=list_reference_heuristic_test, case="lower",
        layer_wise_attention=args.layer_wise_attention,
        slang_dic_test=slang_dic, early_stoppin_metric=early_stoppin_metric,
        multi_task_loss_ponderation=args.multi_task_loss_ponderation,
        report=True, verbose=1)

    printing("MODEL {} trained and evaluated", var=[args.model_id_pref], verbose_level=1, verbose=verbose)
