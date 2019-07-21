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

    # ["bert"]
    voc_pos_size = 17+1 #18+1 for alg_arabizi # 53+1 for ARABIZI 1# 21 is for ENGLISH
    printing("MODEL : voc_pos_size hardcoded to {}", var=voc_pos_size, verbose_level=1, verbose=verbose)

    debug = False
    if os.environ.get("ENV") in ["rioc", "neff"]:
        debug = False
    if args.checkpoint_dir is None:
        # TODO vocab_size should be loaded from args.json
        # TEMPORARY : should eventually keep only : model = make_bert_multitask()
        if args.multitask:
            model = make_bert_multitask(pretrained_model_dir=model_dir, tasks=["pos"])
        else:
            model = get_bert_token_classification(pretrained_model_dir=model_dir,
                                                  vocab_size=vocab_size,
                                                  freeze_parameters=args.freeze_parameters,
                                                  freeze_layer_prefix_ls=args.freeze_layer_prefix_ls,
                                                  dropout_classifier=args.dropout_classifier,
                                                  dropout_bert=args.dropout_bert,
                                                  tasks=args.tasks,
                                                  voc_pos_size=voc_pos_size,
                                                  bert_module=args.bert_module,
                                                  layer_wise_attention=args.layer_wise_attention,
                                                  mask_n_predictor=args.append_n_mask,
                                                  initialize_bpe_layer=args.initialize_bpe_layer,
                                                  debug=debug)
    else:
        printing("MODEL : reloading from checkpoint {} all models parameters are "
                 "ignored except task bert module and layer_wise_attention", var=[args.checkpoint_dir], verbose_level=1, verbose=verbose)
        # TODO args.original_task  , vocab_size is it necessary
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

    null_token_index = BERT_MODEL_DIC[args.bert_model]["vocab_size"]  # based on bert cased vocabulary
    description = "grid"
    list_reference_heuristic_test = pickle.load(open(os.path.join(PROJECT_PATH, "data/wiki-news-FAIR-SG-top50000.pkl"), "rb"))
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

    run(args=args, model=model, voc_tokenizer=voc_tokenizer,
        description=description, null_token_index=null_token_index, null_str=NULL_STR,
        model_suffix="{}".format(args.model_id_pref), debug=debug,
        random_iterator_train=True,  bucket_test=False, compute_intersection_score_test=True,
        list_reference_heuristic_test=list_reference_heuristic_test, case="lower",
        n_iter_max_per_epoch=10,
        slang_dic_test=slang_dic, early_stoppin_metric=early_stoppin_metric,
        saving_every_epoch=15, auxilliary_task_norm_not_norm=True,
        report=True, verbose=1)

    printing("MODEL {} trained and evaluated", var=[args.model_id_pref], verbose_level=1, verbose=verbose)
