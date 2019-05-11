from env.models_dir import *
from model.bert_normalize import get_bert_token_classification
from training.bert_normalize.fine_tune_bert import run
from evaluate.interact import interact_bert_wrap
from model.bert_tools_from_core_code.tokenization import BertTokenizer
from predict.predict_string_bert import interact_bert
from io_.dat.constants import TOKEN_BPE_BERT_START, TOKEN_BPE_BERT_SEP, NULL_STR

PAD_ID_BERT = 0
PAD_BERT = "[PAD]"

train_path = [PERMUTATION_TRAIN_DIC[10000]]
dev_path = [PERMUTATION_TEST]
train_path = [DEMO]
dev_path = [DEMO]#[LIU_DEV]#[DEMO2]
#dev_path = None
test_paths_ls = [[DEV], [LIU_DEV], [TEST], [LIU_TRAIN]]#, [LIU_TRAIN], [LIU_DEV], [DEV], [LEX_TEST], [LEX_TRAIN], [LEX_LIU_TRAIN]]
test_paths_ls = [[TEST], [DEV], [EWT_DEV]]
test_paths_ls = [[DEMO]]

train = True
playwith = False


if train:
    # TODO : WARNING : why the delis still
    #  loaded even in vocab size not consistent with what is suppose to be the vocabulary of the model loaded

    voc_tokenizer = BERT_MODEL_DIC["bert-cased"]["vocab"]
    model_dir = BERT_MODEL_DIC["bert-cased"]["model"]
    vocab_size = BERT_MODEL_DIC["bert-cased"]["vocab_size"]

    initialize_bpe_layer = True
    freeze_parameters = True
    freeze_layer_prefix_ls = ["bert"]
    tasks = ["normalize"]
    voc_pos_size = 16
    #["bert"]
    model = get_bert_token_classification(pretrained_model_dir=model_dir,
                                          vocab_size=vocab_size, dropout_classifier=0.5,
                                          freeze_parameters=freeze_parameters,
                                          voc_pos_size=voc_pos_size, tasks=tasks,
                                          freeze_layer_prefix_ls=freeze_layer_prefix_ls,
                                          dropout_bert=0.0, initialize_bpe_layer=initialize_bpe_layer)
    lr = 0.0001
    batch_size = 1
    null_token_index = BERT_MODEL_DIC["bert-cased"]["vocab_size"]  # based on bert cased vocabulary
    description = "DEBUGGING_LEAK-AS_BEFORE"
    print("{} lr batch_size initialize_bpe_layer training_data".format(REPORT_FLAG_VARIABLES_ENRICH_STR))
    print("{} tnr accuracy f1 tnr precision recall npvr".format(REPORT_FLAG_VARIABLES_EXPAND_STR))
    print("{} ".format(REPORT_FLAG_VARIABLES_FIXED_STR))
    print("{} lr batch_size initialize_bpe_layer training_data".format(REPORT_FLAG_VARIABLES_ANALYSED_STR))

    model = run(bert_with_classifier=model,
                voc_tokenizer=voc_tokenizer, tasks=tasks, train_path=train_path, dev_path=dev_path,
                auxilliary_task_norm_not_norm=True,
                saving_every_epoch=10,
                lr=0.001,#OrderedDict([("bert", lr), ("classifier", lr)]),
                batch_size=batch_size, n_iter_max_per_epoch=5,
                n_epoch=10,
                test_path_ls=test_paths_ls,
                description=description, null_token_index=null_token_index, null_str=NULL_STR,
                model_suffix="{}".format(description), debug=False,
                fine_tuning_strategy="standart",
                #masking_strategy=["normed", 1.],
                freeze_parameters=freeze_parameters, freeze_layer_prefix_ls=freeze_layer_prefix_ls,
                initialize_bpe_layer=initialize_bpe_layer, args=None, skip_1_t_n=False, dropout_input_bpe=0.0,
                heuristic_ls=None, gold_error_detection=False,
                norm_2_noise_eval=False, norm_2_noise_training=0.,
                report=True, verbose=1)


null_token_index = BERT_MODEL_DIC["bert-cased"]["vocab_size"]  # based on bert cased vocabulary


if playwith:
    train_path = [EN_LINES_EWT_TRAIN]
    dev_path = [DEMO]  # [LIU_DEV]#[DEMO2]

    vocab_size = BERT_MODEL_DIC["bert-cased"]["vocab_size"]
    voc_tokenizer = BERT_MODEL_DIC["bert-cased"]["vocab"]
    tokenizer = BertTokenizer.from_pretrained(voc_tokenizer)
    model_location = "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/./checkpoints/bert/b5338-LOOK_THE_PREDICTIONS-2batch-0.0001lr"
    model_name = "b5338-LOOK_THE_PREDICTIONS-2batch-0.0001lr-ep24-checkpoint.pt"
    #model_location = "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/./checkpoints/bert/9319649-B-14cf0-9319649-B-model_0"
    #model_name = "9319649-B-14cf0-9319649-B-model_0-ep4-checkpoint.pt"
    checkpoint_dir = os.path.join(model_location, model_name)
    test_paths_ls = [[EN_LINES_EWT_TRAIN]]
    voc_pos_size = 21

    tasks = ["pos"]

    model = get_bert_token_classification(vocab_size=vocab_size, voc_pos_size=voc_pos_size,
                                          tasks=["normalize"],
                                          initialize_bpe_layer=None,
                                          checkpoint_dir=checkpoint_dir)
    add_task_2 = True
    if add_task_2:
        model.classifier_task_2 = nn.Linear(model.bert.config.hidden_size, voc_pos_size)
        model.num_labels_2 =voc_pos_size
    #model.load_state_dict(torch.load(checkpoint_dir, map_location=lambda storage, loc: storage))
    # NB : AT TEST TIME :  null_token_index should be loaded not passed as argument
    pref_suffix = ""
    batch_size = 1
    lr = ""
    evalu = True
    if evalu:
        model = run(bert_with_classifier=model,
                    voc_tokenizer=voc_tokenizer, tasks=tasks, train_path=train_path, dev_path=dev_path,
                    auxilliary_task_norm_not_norm=True,
                    saving_every_epoch=10, lr=lr,
                    dict_path=os.path.join(model_location, "dictionaries"),
                    end_predictions=os.path.join(model_location, "predictions"),
                    batch_size=batch_size, n_iter_max_per_epoch=10, n_epoch=1,
                    test_path_ls=test_paths_ls, run_mode="test",
                    args=None,
                    description="", null_token_index=null_token_index, null_str=NULL_STR, model_location=model_location,
                    model_id="b5338-LOOK_THE_PREDICTIONS-2batch-0.0001lr",
                    model_suffix="{}-{}batch-{}lr".format(pref_suffix, batch_size, lr),
                    debug=True, report=True,
                    verbose="raw_data")

    # TO SEE TOKENIZATION IMPACT : verbose='raw_data'
    #interact_bert_wrap(tokenizer, model,
    #                   null_str=NULL_STR, null_token_index=null_token_index,
    #                   topk=5, verbose=3)




