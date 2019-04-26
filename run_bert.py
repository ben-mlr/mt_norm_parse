from env.models_dir import *
from model.bert_normalize import run
from evaluate.interact import interact_bert_wrap
from model.bert_tools_from_core_code.tokenization import BertTokenizer
from predict.predict_string_bert import interact_bert
TOKEN_BPE_BERT_START = "[CLS]"
TOKEN_BPE_BERT_SEP = "[SEP]"
PAD_ID_BERT = 0
PAD_BERT = "[PAD]"


train_path = [PERMUTATION_TRAIN_DIC[10000]]
dev_path = [PERMUTATION_TEST]
train_path = [LIU_TRAIN]
dev_path = [TEST]
tasks = ["normalize"]

train = True
playwith = False

if train:

    voc_tokenizer = BERT_MODEL_DIC["bert-cased"]["vocab"]
    model_dir = BERT_MODEL_DIC["bert-cased"]["model"]
    vocab_size = BERT_MODEL_DIC["bert-cased"]["vocab_size"]
    # TODO : WARNING : why the delis still loaded even in vocab size not consistent with what is suppose to be the vocabulary of the model loaded
    config = BertConfig(vocab_size_or_config_json_file=vocab_size, hidden_size=768,
                        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    # QUESTION : WHERE IS THE MODEL ACTUALLY BEING LOADED ???
    num_labels = vocab_size+1
    NULL_TOKEN_INDEX = vocab_size
    initialize_bpe_layer = True 
    model = BertForTokenClassification(config, num_labels)

    model = model.from_pretrained(model_dir, num_labels=num_labels)

    if initialize_bpe_layer:
        output_layer = torch.cat((model.bert.embeddings.word_embeddings.weight.data, torch.rand((1, 768))), dim=0)
        model.classifier.weight = nn.Parameter(output_layer)
    lr = 0.0001
    batch_size = 10
    pref_suffix = "local_init"

    run(bert_with_classifier=model,
        voc_tokenizer=voc_tokenizer, tasks=tasks, train_path=train_path, dev_path=dev_path,
        auxilliary_task_norm_not_norm=True,
        saving_every_epoch=10, lr=lr,
        batch_size=batch_size, n_iter_max_per_epoch=2000, n_epoch=1,
        model_suffix="{}-{}batch-{}lr".format(pref_suffix, batch_size, lr), debug=False, report=True, verbose=1)

if playwith:
    vocab_size = BERT_MODEL_DIC["bert-cased"]["vocab_size"]
    voc_tokenizer = BERT_MODEL_DIC["bert-cased"]["vocab"]

    tokenizer = BertTokenizer.from_pretrained(voc_tokenizer)
    config = BertConfig(vocab_size_or_config_json_file=vocab_size, hidden_size=768,
                        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = vocab_size + 1
    initialize_bpe_layer = True
    model = BertForTokenClassification(config, num_labels)
    checkpoint_dir = "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/checkpoints/bert/6c95d-real-30batch-0.0001lr/6c95d-real-30batch-0.0001lr-ep49-checkpoint.pt"
    model.load_state_dict(torch.load(checkpoint_dir, map_location=lambda storage, loc: storage))

    interact_bert_wrap(tokenizer, model, topk=5, verbose=2)




