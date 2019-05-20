from env.models_dir import *
from io_.info_print import printing
from toolbox.deep_learning_toolbox import freeze_param
from model.bert_tools_from_core_code.modeling import BertForTokenClassification, BertConfig


def get_bert_token_classification(vocab_size, voc_pos_size=None,
                                  pretrained_model_dir=None, checkpoint_dir=None,
                                  freeze_parameters=False, freeze_layer_prefix_ls=None,
                                  dropout_classifier=None,dropout_bert=0.,tasks=None,
                                  initialize_bpe_layer=None, verbose=1):
    """
    two use case :
    - initialize bert based on pretrained_model_dir and add a token prediction module based or not on initialize_bpe_layer
    - reload from checkpoint bert+tokenclassification
    :param vocab_size:
    :param pretrained_model_dir:
    :param checkpoint_dir:
    :param initialize_bpe_layer:
    :param verbose:
    :return:
    """
    if tasks is None:
        tasks = ["normalize"]

    assert checkpoint_dir is not None or True, \
        "Neither checkpoint_dir or pretrained_model_dir was provided"
    assert pretrained_model_dir is None or checkpoint_dir is None, \
        "Only one of checkpoint_dir or pretrained_model_dir should be provided "
    config = BertConfig(vocab_size_or_config_json_file=vocab_size, hidden_size=768,
                        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    #config.hidden_dropout_prob = 0.2
    # QUESTION : WHERE IS THE MODEL ACTUALLY BEING LOADED ???
    # this line is useless apparently as it does it load it again
    if "normalize" in tasks:
        num_labels = vocab_size + 1
    else:
        num_labels = 1
        printing("MODEL : WE ARE STILL DEFINING a CLASSIFIER_1 but it should be left untrained ", verbose=verbose,verbose_level=1)
    if "pos" in tasks:
        assert voc_pos_size is not None

    model = BertForTokenClassification(config, num_labels, dropout_classifier=dropout_classifier, num_labels_2=voc_pos_size)

    if pretrained_model_dir is not None:

        assert initialize_bpe_layer is not None, "ERROR initialize_bpe_layer should not be None "

        model = model.from_pretrained(pretrained_model_dir, num_labels=num_labels, dropout_custom=dropout_bert)
        printing("MODEL : loading pretrained BERT and adding extra module for token classification based on {}",
                 var=[pretrained_model_dir],
                 verbose=verbose,
                 verbose_level=1)
        if dropout_classifier is not None:
            model.dropout = nn.Dropout(dropout_classifier)
            printing("MODEL : SETTING DROPOUT CLASSIFIER TO {}".format(dropout_classifier), verbose=verbose, verbose_level=1)

        #if (len(tasks) > 1 and "normalize" in tasks:
        if "pos" in tasks:
            #assert tasks[1] == "pos", "ONLY  POS and normalize supported so far"
            model.classifier_task_2 = nn.Linear(model.bert.config.hidden_size, voc_pos_size)
            model.num_labels_2 = voc_pos_size
            printing("MODEL : adding classifier_task_2 for POS with voc_pos_size {}",
                     var=[voc_pos_size], verbose=verbose, verbose_level=1)

        if initialize_bpe_layer and tasks[0] == "normalize":
            # it needs to be the first as we are setting classifier (1) to the normalizaiton classifier
            output_layer = torch.cat((model.bert.embeddings.word_embeddings.weight.data, torch.rand((1, 768))),
                                     dim=0)
            model.classifier_task_1.weight = nn.Parameter(output_layer)
            printing("MODEL : initializing output normalization layer with embedding layer + extra token ",
                     verbose=verbose,
                     verbose_level=1)
        if freeze_parameters:
            model = freeze_param(model, freeze_layer_prefix_ls=freeze_layer_prefix_ls, verbose=verbose)

    elif checkpoint_dir is not None:
        assert initialize_bpe_layer is None, \
            "ERROR initialize_bpe_layer should b None as loading from existing checkpoint"
        #model.load_state_dict(mode)
        model.load_state_dict(torch.load(checkpoint_dir, map_location=lambda storage, loc: storage))
        printing("MODEL : loading model BERT+token classification pretrained from checkpoint {}",
                 var=[checkpoint_dir],
                 verbose=verbose, verbose_level=1)

    return model


