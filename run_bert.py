from env.models_dir import *
from model.bert_normalize import run

TOKEN_BPE_BERT_START = "[CLS]"
TOKEN_BPE_BERT_SEP = "[SEP]"
PAD_ID_BERT = 0
PAD_BERT = "[PAD]"


train_path = [LIU_TRAIN]
dev_path = [TEST]
tasks = ["normalize"]


if True:

    voc_tokenizer = BERT_CASED_DIR

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
                        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 32001
    NULL_TOKEN_INDEX = 32000
    initialize_bpe_layer = False
    model = BertForTokenClassification(config, num_labels)
    if initialize_bpe_layer:
        output_layer = torch.cat((model.bert.embeddings.word_embeddings.weight.data, torch.rand((1, 768))), dim=0)
        model.classifier.weight = nn.Parameter(output_layer)

    run(bert_with_classifier=model,
        voc_tokenizer=voc_tokenizer, tasks=tasks, train_path=train_path, dev_path=dev_path,
        auxilliary_task_norm_not_norm=True,
        saving_every_epoch=10,
        batch_size=3, n_iter_max_per_epoch=2, n_epoch=2,
        model_suffix="init", debug=False,
        report=True, verbose=1)