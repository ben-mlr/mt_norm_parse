from env.models_dir import *
from model.bert_normalize import get_bert_token_classification
from training.bert_normalize.fine_tune_bert import run
from evaluate.interact import interact_bert_wrap
from model.bert_tools_from_core_code.tokenization import BertTokenizer
from predict.predict_string_bert import interact_bert
from toolbox.pred_tools.heuristics import get_letter_indexes
from io_.dat.constants import TOKEN_BPE_BERT_START, TOKEN_BPE_BERT_SEP, NULL_STR

PAD_ID_BERT = 0
PAD_BERT = "[PAD]"

train_path = [PERMUTATION_TRAIN_DIC[10000]]
dev_path = [PERMUTATION_TEST]
train_path = [DEMO]
dev_path = [DEMO]#[LIU_DEV]#[DEMO2]
#dev_path = None
test_paths_ls = [[DEV], [LIU_DEV], [TEST], [LIU_TRAIN]]#, [LIU_TRAIN], [LIU_DEV], [DEV], [LEX_TEST], [LEX_TRAIN], [LEX_LIU_TRAIN]]
test_paths_ls = [[TEST],
                 [DEV],
                 [EWT_DEV]]
test_paths_ls = [[DEMO]]

train_path = [GENERATED_DIC[100]]
dev_path = [GENERATED_DIC[100]]


train = True
playwith = False


if train:
    # TODO : WARNING : why the delis still
    #  loaded even in vocab size not consistent with what is suppose to be the vocabulary of the model loaded

    voc_tokenizer = BERT_MODEL_DIC["bert-cased"]["vocab"]
    model_dir = BERT_MODEL_DIC["bert-cased"]["model"]
    vocab_size = BERT_MODEL_DIC["bert-cased"]["vocab_size"]

    initialize_bpe_layer = True
    freeze_parameters = False
    freeze_layer_prefix_ls = ["cls", "bert.encoder", "bert.encoder.layer.1"]
    tasks = ["normalize"]
    layer_wise_attention = 0
    train_path = [LEX_TRAIN_SPLIT_2]#, DEMO]
    dev_path = [DEMO]#, DEMO]
    test_paths_ls = [[LEX_TEST]]#, [DEMO]]
    bert_module = "mlm"
    mask_n_predictor = False
    voc_pos_size = 16
    #["bert"]
    model = get_bert_token_classification(pretrained_model_dir=model_dir,
                                          vocab_size=vocab_size, dropout_classifier=0.5,
                                          freeze_parameters=freeze_parameters,
                                          voc_pos_size=voc_pos_size, tasks=tasks,layer_wise_attention=layer_wise_attention,
                                          freeze_layer_prefix_ls=freeze_layer_prefix_ls, bert_module=bert_module,
                                          mask_n_predictor=mask_n_predictor,
                                          dropout_bert=0.0, initialize_bpe_layer=initialize_bpe_layer)
    lr = 0.0001

    batch_size = 2
    null_token_index = BERT_MODEL_DIC["bert-cased"]["vocab_size"]  # based on bert cased vocabulary
    description = "DEBUGGING_LEAK-AS_BEFORE"
    print("{} lr batch_size initialize_bpe_layer training_data".format(REPORT_FLAG_VARIABLES_ENRICH_STR))
    print("{} tnr accuracy f1 tnr precision recall npvr".format(REPORT_FLAG_VARIABLES_EXPAND_STR))
    print("{} ".format(REPORT_FLAG_VARIABLES_FIXED_STR))
    print("{} lr batch_size initialize_bpe_layer training_data".format(REPORT_FLAG_VARIABLES_ANALYSED_STR))

    list_reference_heuristic_test = pickle.load(open(os.path.join(PROJECT_PATH, "./data/wiki-news-FAIR-SG-top50000.pkl"), "rb"))#list(json.load(open(os.path.join(PROJECT_PATH, "./data/words_dictionary.json"), "r"), object_pairs_hook=OrderedDict).keys())
    #index_alphabetical_order = json.load(open(os.path.join(PROJECT_PATH,"data/wiki-news-FAIR-SG-top50000-letter_to_index.json"), "r"))#json.load(open(os.path.join(PROJECT_PATH, "data/words_dictionary_letter_to_index.json"), "r"))

    slang_dic = json.load(open(os.path.join(PROJECT_PATH, "./data/urban_dic_abbreviations.json"), "r"))

    model = run(bert_with_classifier=model,
                voc_tokenizer=voc_tokenizer, tasks=tasks, train_path=train_path, dev_path=dev_path,
                auxilliary_task_norm_not_norm=True,
                saving_every_epoch=10,
                lr=0.00001, #lr=OrderedDict([("bert", 5e-5), ("classifier_task_1", 0.001), ("classifier_task_2", 0.001)]),
                batch_size=batch_size, n_iter_max_per_epoch=100,
                n_epoch=1,
                test_path_ls=test_paths_ls,
                description=description, null_token_index=null_token_index, null_str=NULL_STR,
                model_suffix="{}".format(description), debug=False,
                tokenize_and_bpe=False,
                fine_tuning_strategy="standart",
                masking_strategy=["mlm", "0"],
                freeze_parameters=freeze_parameters, freeze_layer_prefix_ls=freeze_layer_prefix_ls,
                initialize_bpe_layer=initialize_bpe_layer, args=None,
                skip_1_t_n=False, dropout_input_bpe=0.0,
                heuristic_ls=None, gold_error_detection=False,
                bucket_test=True, must_get_norm_test=True,
                list_reference_heuristic_test=list_reference_heuristic_test,
                slang_dic_test=slang_dic,bert_module=bert_module,
                norm_2_noise_eval=False, #norm_2_noise_training=,
                aggregating_bert_layer_mode=5, case="lower", #threshold_edit=2.9,
                report=True, verbose="alignement")


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
    #model_location = "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/checkpoints/bert/9320927-B-ed1e8-9320927-B-model_0"
    #model_name = "9320927-B-ed1e8-9320927-B-model_0-ep19-checkpoint.pt"
    model_location = "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/checkpoints/bert/9356861-B-36825-9356861-B-model_0/"
    model_name = "9356861-B-36825-9356861-B-model_0-epbest-checkpoint.pt"
    checkpoint_dir = os.path.join(model_location, model_name)
    test_paths_ls = [[TEST]]
    # TODO : predict with a norm2noise model
    #  can use tasks trick ..
    voc_pos_size = 21
    tasks = ["normalize"]
    layer_wise_attention = False
    model = get_bert_token_classification(vocab_size=vocab_size, voc_pos_size=voc_pos_size,
                                          tasks=["normalize"],
                                          initialize_bpe_layer=None,
                                          bert_module="mlm", layer_wise_attention=layer_wise_attention,
                                          checkpoint_dir=checkpoint_dir)

    model.normalization_module = True
    add_task_2 = False
    if add_task_2:
        model.classifier_task_2 = nn.Linear(model.bert.config.hidden_size, voc_pos_size)
        model.num_labels_2 = voc_pos_size
    #model.load_state_dict(torch.load(checkpoint_dir, map_location=lambda storage, loc: storage))
    # NB : AT TEST TIME :  null_token_index should be loaded not passed as argument
    pref_suffix = ""
    batch_size = 1
    lr = ""

    evalu = False

    list_reference_heuristic_test = [] #json.load(open("./data/words_dictionary.json", "r")).keys()
    slang_dic = json.load(open("./data/urban_dic_abbreviations.json","r"))
    if evalu:
        model = run(bert_with_classifier=model,
                    voc_tokenizer=voc_tokenizer, tasks=tasks, train_path=train_path, dev_path=dev_path,
                    auxilliary_task_norm_not_norm=True,
                    saving_every_epoch=10, lr=lr,
                    dict_path=os.path.join(model_location, "dictionaries"),
                    end_predictions=os.path.join(model_location, "predictions"),
                    batch_size=batch_size, n_iter_max_per_epoch=n_sent, n_epoch=1,
                    test_path_ls=test_paths_ls, run_mode="test",
                    args=None,bert_module="mlm",
                    description="", null_token_index=null_token_index, null_str=NULL_STR, model_location=model_location,
                    model_id=model_name[:-2],
                    model_suffix="{}-{}batch-{}lr".format(pref_suffix, batch_size, lr),
                    debug=False, report=True,
                    remove_mask_str_prediction=True, inverse_writing=False,
                    extra_label_for_prediction="RE_PREDICT",
                    heuristic_test_ls=[None],
                    bucket_test=False, must_get_norm_test=False,
                    slang_dic_test=slang_dic, list_reference_heuristic_test=list_reference_heuristic_test,
                    layer_wise_attention=layer_wise_attention,
                    verbose="raw_data")
        sentences = False
        if sentences:
            for n_sent in [50, 80, 120, 150, 250, 350]:
                model = run(bert_with_classifier=model,
                            voc_tokenizer=voc_tokenizer, tasks=tasks,
                            train_path=train_path, dev_path=dev_path,
                            auxilliary_task_norm_not_norm=True,
                            saving_every_epoch=10, lr=lr,
                            dict_path=os.path.join(model_location, "dictionaries"),
                            end_predictions=os.path.join(model_location, "predictions"),
                            batch_size=batch_size, n_iter_max_per_epoch=n_sent, n_epoch=1,
                            test_path_ls=test_paths_ls, run_mode="test",
                            args=None,
                            description="", null_token_index=null_token_index, null_str=NULL_STR, model_location=model_location,
                            model_id="9326829-B-fbbe9-9326829",
                            model_suffix="{}-{}batch-{}lr".format(pref_suffix, batch_size, lr),
                            debug=False, report=True,
                            remove_mask_str_prediction=True, inverse_writing=True,
                            extra_label_for_prediction="{}".format(n_sent),
                            bucket_test=False, must_get_norm_test=False,
                            verbose=1)
                print("DONE ", n_sent)

    # TO SEE TOKENIZATION IMPACT : verbose='raw_data'
    interact_bert_wrap(tokenizer, model,
                       tasks=tasks,
                       null_str=NULL_STR, null_token_index=null_token_index,
                       topk=5, verbose=3)


some_processing = False


if some_processing:

    with open("./data/wiki-news-300d-1M-subword-top50000.vec", "r") as f:
        ind = 0
        ls = []
        for line in f:
            ind += 1
            if ind == 1:
                continue
            word = line.strip().split(" ")[0].lower()
            if word.isalpha():
                ls.append(word)
        print(word, word.isalpha())
        ls.sort()
        #pickle.dump(ls, open("./data/wiki-news-FAIR-SG-top50000.pkl", "wb"))
        ls_2 = pickle.load(open("./data/wiki-news-FAIR-SG-top50000.pkl", "rb"))
        pdb.set_trace()

    data = json.load(open("./data/words_dictionary.json", "r"), object_pairs_hook=OrderedDict)
    new_letter = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", 'q', 'r', "s", "t","u","v", "w", "x", "y", "z", "-"]
    len_max = len(ls)

    dic_ind = get_letter_indexes(ls)
    pdb.set_trace()

    #json.dump(dic_ind, open("./data/wiki-news-FAIR-SG-top50000-letter_to_index.json", "w"))
    #json.dump(dic_ind, open("./data/wiki-news-FAIR-SG-top50000.pkl", "w"))
    if False:
        with open("./data/urban_dic_abbreviations.txt","r") as f:
            urban_dic = {}
            for line in f:
                if len(line.strip()) != 0:
                    reg = re.match("(.*):(.*)", line.strip())
                    original = reg.group(1).lower()
                    def_ = reg.group(2).lower().replace(" ","")
                    print(original, def_)
                    urban_dic[original] = def_

        #json.dump(urban_dic, open("./data/urban_dic_abbreviations.json","w"))
