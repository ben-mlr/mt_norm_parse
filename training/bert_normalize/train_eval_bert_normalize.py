from env.importing import os, nn, json, OrderedDict, pickle, pdb
from training.bert_normalize.fine_tune_bert import run
from env.project_variables import PROJECT_PATH
from model.bert_tools_from_core_code.get_model import get_multi_task_bert_model
from model.bert_normalize import get_bert_token_classification, make_bert_multitask
from io_.dat.constants import TOKEN_BPE_BERT_START, TOKEN_BPE_BERT_SEP, NULL_STR, PAD_BERT, PAD_ID_BERT, SPECIAL_TOKEN_LS
from env.models_dir import BERT_MODEL_DIC
from toolbox.bert_tools.get_bert_info import get_bert_name
from env.project_variables import MULTITASK_BERT_LABELS_MLM_HEAD, MULTITASK_BERT_LABELS_MLM_HEAD_LOSS
from io_.info_print import printing


def update_multitask_loss(multi_task_loss_ponderation):
    if multi_task_loss_ponderation is None:
        return None
    multi_task_loss_ponderation_new = {}
    for task, weight in multi_task_loss_ponderation.items():
        if task in MULTITASK_BERT_LABELS_MLM_HEAD:
            multi_task_loss_ponderation_new[MULTITASK_BERT_LABELS_MLM_HEAD_LOSS[task]] = weight
    return multi_task_loss_ponderation_new


def train_eval_bert_normalize(args, verbose=1):

    args.bert_model = get_bert_name(args.bert_model)
    voc_tokenizer = BERT_MODEL_DIC[args.bert_model]["vocab"]
    model_dir = BERT_MODEL_DIC[args.bert_model]["model"]
    vocab_size = BERT_MODEL_DIC[args.bert_model]["vocab_size"]

    # ["bert"]
    voc_pos_size = 21 #18+1 for alg_arabizi # 53+1 for ARABIZI 1# 21 is for ENGLISH
    printing("MODEL : voc_pos_size hardcoded to {}", var=voc_pos_size, verbose_level=1, verbose=verbose)

    debug = False
    if os.environ.get("ENV") in ["rioc", "neff"]:
        debug = False

    model = get_multi_task_bert_model(args,  model_dir, vocab_size, voc_pos_size, debug, verbose)


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
    printing("INFO : environ is {} so debug set to {}", var=[os.environ.get("ENV", "Unkwnown"), debug], verbose_level=1, verbose=verbose)
    # MLM in multitas mode is temporary and require task_i indexing : that's why we need to rename ponderation dictionary
    if not args.multitask:
        args.multi_task_loss_ponderation = update_multitask_loss(args.multi_task_loss_ponderation)

    run(args=args, model=model, voc_tokenizer=voc_tokenizer,
        report_full_path_shared=args.overall_report_dir,
        description=description, null_token_index=null_token_index, null_str=NULL_STR,
        model_suffix="{}".format(args.model_id_pref), debug=debug,
        random_iterator_train=True,  bucket_test=False, compute_intersection_score_test=True,
        list_reference_heuristic_test=list_reference_heuristic_test, case="lower",
        n_iter_max_per_epoch=4,
        slang_dic_test=slang_dic, early_stoppin_metric=early_stoppin_metric,
        saving_every_epoch=15, auxilliary_task_norm_not_norm=True,
        report=True, verbose=1)

    printing("MODEL {} trained and evaluated", var=[args.model_id_pref], verbose_level=1, verbose=verbose)
