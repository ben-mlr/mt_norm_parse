from env.models_dir import *
from model.bert_normalize import run

TOKEN_BPE_BERT_START = "[CLS]"
TOKEN_BPE_BERT_SEP = "[SEP]"
PAD_ID_BERT = 0
PAD_BERT = "[PAD]"


train_path = [PERMUTATION_TRAIN_DIC[10000]]
dev_path = [PERMUTATION_TEST]
train_path = [LIU_TRAIN]
dev_path = [TEST]
tasks = ["normalize"]

if True:

    voc_tokenizer = BERT_MODEL_DIC["bert-cased"]["vocab"]
    model_dir = BERT_MODEL_DIC["bert-cased"]["model"]
    vocab_size = BERT_MODEL_DIC["bert-cased"]["vocab_size"]
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
    pdb.set_trace()
    lr = 0.002
    batch_size = 10
    run(bert_with_classifier=model,
        voc_tokenizer=voc_tokenizer, tasks=tasks, train_path=train_path, dev_path=dev_path,
        auxilliary_task_norm_not_norm=True,
        saving_every_epoch=10, lr=lr,
        batch_size=batch_size, n_iter_max_per_epoch=40, n_epoch=1,
        model_suffix="real-{}batch-{}lr".format(batch_size, lr), debug=True, report=True, verbose=1)