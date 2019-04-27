from env.models_dir import *
from model.bert_normalize import get_bert_token_classification
from training.fine_tune_bert import run
from evaluate.interact import interact_bert_wrap
from model.bert_tools_from_core_code.tokenization import BertTokenizer
from predict.predict_string_bert import interact_bert
from io_.dat.constants import TOKEN_BPE_BERT_START, TOKEN_BPE_BERT_SEP,NULL_STR

PAD_ID_BERT = 0
PAD_BERT = "[PAD]"

train_path = [PERMUTATION_TRAIN_DIC[10000]]
dev_path = [PERMUTATION_TEST]
train_path = [DEMO]
dev_path = [DEMO2]
test_paths_ls = [[LIU_TRAIN], [LIU_DEV], [TEST], [DEV]]
tasks = ["normalize"]


train = True
playwith = False

if train:


    # TODO : WARNING : why the delis still loaded even in vocab size not consistent with what is suppose to be the vocabulary of the model loaded
    voc_tokenizer = BERT_MODEL_DIC["bert-cased"]["vocab"]
    model_dir = BERT_MODEL_DIC["bert-cased"]["model"]
    vocab_size = BERT_MODEL_DIC["bert-cased"]["vocab_size"]

    model = get_bert_token_classification(pretrained_model_dir=model_dir,
                                           vocab_size=vocab_size, initialize_bpe_layer=True)

    lr = 0.0001
    batch_size = 2
    null_token_index = BERT_MODEL_DIC["bert-cased"]["vocab_size"]  # based on bert cased vocabulary
    pref_suffix = "LOOK_THE_PREDICTIONS"
    description = "BERT_NORM:{}-{}batch-{}lr-trained:{}-LIUDEV".format(pref_suffix, batch_size, lr, REPO_DATASET[train_path[0]])

    model = run(bert_with_classifier=model,
                voc_tokenizer=voc_tokenizer, tasks=tasks, train_path=train_path, dev_path=dev_path,
                auxilliary_task_norm_not_norm=True,
                saving_every_epoch=10, lr=lr,
                batch_size=batch_size, n_iter_max_per_epoch=5, n_epoch=1,
                test_path_ls=test_paths_ls,
                description=description, null_token_index=null_token_index, null_str=NULL_STR,
                model_suffix="{}-{}batch-{}lr".format(pref_suffix, batch_size, lr), debug=False, report=True, verbose=1)


null_token_index = BERT_MODEL_DIC["bert-cased"]["vocab_size"]  # based on bert cased vocabulary


if playwith:

    vocab_size = BERT_MODEL_DIC["bert-cased"]["vocab_size"]
    voc_tokenizer = BERT_MODEL_DIC["bert-cased"]["vocab"]
    tokenizer = BertTokenizer.from_pretrained(voc_tokenizer)
    model_location = "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/./checkpoints/bert/b5338-LOOK_THE_PREDICTIONS-2batch-0.0001lr"
    checkpoint_dir = os.path.join(model_location,"b5338-LOOK_THE_PREDICTIONS-2batch-0.0001lr-ep24-checkpoint.pt")

    model = get_bert_token_classification(vocab_size=vocab_size,
                                           checkpoint_dir=checkpoint_dir)

    #model.load_state_dict(torch.load(checkpoint_dir, map_location=lambda storage, loc: storage))
    # NB : AT TEST TIME :  null_token_index should be loaded not passed as argument
    pref_suffix = ""
    batch_size = 2
    lr = ""
    model = run(bert_with_classifier=model,
                voc_tokenizer=voc_tokenizer, tasks=tasks, train_path=train_path, dev_path=dev_path,
                auxilliary_task_norm_not_norm=True,
                saving_every_epoch=10, lr=lr,
                dict_path=os.path.join(model_location, "dictionaries"),
                end_predictions=os.path.join(model_location, "predictions"),
                batch_size=batch_size, n_iter_max_per_epoch=5, n_epoch=1,
                test_path_ls=test_paths_ls, run_mode="test",
                description="", null_token_index=null_token_index, null_str=NULL_STR,
                model_suffix="{}-{}batch-{}lr".format(pref_suffix, batch_size, lr), debug=False, report=True, verbose=1)

    #interact_bert_wrap(tokenizer, model,
    #                   null_str=NULL_STR, null_token_index=null_token_index,
    #                   topk=5, verbose=2)




