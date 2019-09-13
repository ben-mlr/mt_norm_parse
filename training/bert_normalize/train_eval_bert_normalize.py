from env.importing import os, nn, json, OrderedDict, pickle, pdb
from training.bert_normalize.fine_tune_bert import run
from env.project_variables import PROJECT_PATH

from io_.dat.constants import TOKEN_BPE_BERT_START, TOKEN_BPE_BERT_SEP, NULL_STR, PAD_BERT, PAD_ID_BERT, SPECIAL_TOKEN_LS
from env.models_dir import BERT_MODEL_DIC
from toolbox.bert_tools.get_bert_info import get_bert_name
from env.tasks_settings import TASKS_PARAMETER
from env.project_variables import MULTITASK_BERT_LABELS_MLM_HEAD, MULTITASK_BERT_LABELS_MLM_HEAD_LOSS
from io_.info_print import printing
from evaluate.scoring.early_stopping_metrics import get_early_stopping_metric


def update_multitask_loss_ponderation(multi_task_loss_ponderation):

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

    debug = True
    if os.environ.get("ENV") in ["rioc", "neff"]:
        debug = False

    null_token_index = BERT_MODEL_DIC[args.bert_model]["vocab_size"]  # based on bert cased vocabulary
    description = "grid"
    list_reference_heuristic_test = pickle.load(open(os.path.join(PROJECT_PATH, "data/wiki-news-FAIR-SG-top50000.pkl"),  "rb"))
    slang_dic = json.load(open(os.path.join(PROJECT_PATH, "data/urban_dic_abbreviations.json"), "r"))

    early_stoppin_metric, subsample_early_stoping_metric_val = get_early_stopping_metric(tasks=args.tasks,
                                                                                         early_stoppin_metric=None,
                                                                                         verbose=verbose)

    printing("INFO : tasks is {} so setting early_stoppin_metric to {} ", var=[args.tasks, early_stoppin_metric], verbose=verbose, verbose_level=1)
    printing("INFO : environ is {} so debug set to {}", var=[os.environ.get("ENV", "Unkwnown"), debug], verbose_level=1, verbose=verbose)
    # MLM in multitas mode is temporary and require task_i indexing : that's why we need to rename ponderation dictionary

    if not args.multitask:
        args.multi_task_loss_ponderation = update_multitask_loss_ponderation(args.multi_task_loss_ponderation)
    args.low_memory_foot_print_batch_mode = 1
    if args.low_memory_foot_print_batch_mode:
        args.batch_update_train = args.batch_size
        args.batch_size = 2
        assert isinstance(args.batch_update_train//args.batch_size, int), "ERROR batch_size {} should be a multiple of 2 ".format(args.batch_update_train)
        printing("INFO iterator : updating with {} equivalent batch size : forward pass is {} batch size",
                 var=[args.batch_update_train, args.batch_size], verbose=verbose, verbose_level=1)

    run(args=args, voc_tokenizer=voc_tokenizer, vocab_size=vocab_size, model_dir=model_dir,
        report_full_path_shared=args.overall_report_dir,
        description=description, null_token_index=null_token_index, null_str=NULL_STR,
        model_suffix="{}".format(args.model_id_pref), debug=debug,
        random_iterator_train=True,  bucket_test=False, compute_intersection_score_test=True,
        list_reference_heuristic_test=list_reference_heuristic_test,
        n_iter_max_per_epoch_train=args.n_iter_max_train if not args.demo_run else 5,
        n_iter_max_per_epoch_dev_test=1000000 if not args.demo_run else 5,
        slang_dic_test=slang_dic,
        early_stoppin_metric=early_stoppin_metric,
        subsample_early_stoping_metric_val=subsample_early_stoping_metric_val,
        saving_every_epoch=args.saving_every_n_epoch,
        auxilliary_task_norm_not_norm=True,
        name_with_epoch=args.name_inflation,
        report=True, verbose=1)

    printing("MODEL {} trained and evaluated", var=[args.model_id_pref], verbose_level=1, verbose=verbose)
