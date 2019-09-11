import logging
import tarfile
import tempfile
from env.project_variables import CHECKPOINT_BERT_DIR
from env.models_dir import *
from io_.info_print import printing
from toolbox.deep_learning_toolbox import freeze_param
from toolbox.report_tools import get_init_args_dir
from model.bert_tools_from_core_code.modeling import BertForTokenClassification, BertConfig, BertForMaskedLM, BertMultiTask, BertConfig


logger = logging.getLogger(__name__)


def make_bert_multitask(pretrained_model_dir, tasks, num_labels_per_task, init_args_dir, mask_id):
    assert num_labels_per_task is not None and isinstance(num_labels_per_task, dict), \
        "ERROR : num_labels_per_task {} should be a dictionary".format(num_labels_per_task)
    assert isinstance(tasks, list) and len(tasks) >= 1, "ERROR tasks {} should be a list of len >=1".format(tasks)

    if pretrained_model_dir is not None and init_args_dir is None:
        model = BertMultiTask.from_pretrained(pretrained_model_dir, tasks=tasks,
                                              mask_id=mask_id,
                                              num_labels_per_task=num_labels_per_task,mapping_keys_state_dic={"cls": "head.mlm"})
        pdb.set_trace()
    elif init_args_dir is not None:
        init_args_dir = get_init_args_dir(init_args_dir)

        args_checkpoint = json.load(open(init_args_dir,"r"))
        assert "checkpoint_dir" in args_checkpoint, "ERROR checkpoint_dir not in {} ".format(args_checkpoint)
        checkpoint_dir = args_checkpoint["checkpoint_dir"]
        assert os.path.isfile(checkpoint_dir), "ERROR checkpoint {} not found ".format(checkpoint_dir)
        # redefining model and reloading

        def get_config_bert(bert_model, config_file_name="bert_config.json"):
            model_dir = BERT_MODEL_DIC[bert_model]["model"]
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(model_dir, tempdir))
            with tarfile.open(model_dir, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
            config_file = os.path.join(serialization_dir, config_file_name)
            assert os.path.isfile(config_file), "ERROR {} not a file ".format(config_file)
            return config_file

        config_file = get_config_bert(args_checkpoint["hyperparameters"]["bert_model"])
        config = BertConfig(config_file)

        model = BertMultiTask(config=config, tasks=[task for tasks in args_checkpoint["hyperparameters"]["tasks"] for task in tasks],
                              num_labels_per_task=args_checkpoint["info_checkpoint"]["num_labels_per_task"])
        model.load_state_dict(torch.load(checkpoint_dir, map_location=lambda storage, loc: storage))
        model.append_extra_heads_model(downstream_tasks=tasks, num_labels_dic_new=num_labels_per_task)
    else:
        raise(Exception("only one of pretrained_model_dir checkpoint_dir can be defined "))

    return model


def get_bert_token_classification(vocab_size, voc_pos_size=None,
                                  pretrained_model_dir=None, checkpoint_dir=None,
                                  freeze_parameters=False, freeze_layer_prefix_ls=None, dropout_classifier=None,
                                  dropout_bert=0.,tasks=None,
                                  bert_module="token_class",
                                  layer_wise_attention=False,
                                  mask_n_predictor=False, add_task_2_for_downstream=False,
                                  initialize_bpe_layer=None, debug=False, verbose=1):
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
    if not debug:
        pdb.set_trace = lambda: None
    if tasks is None:
        tasks = ["normalize"]
    AVAILABLE_BERT_MODE = ["mlm", "token_class"]
    assert bert_module in AVAILABLE_BERT_MODE, "ERROR bert_module should be in {} ".format(AVAILABLE_BERT_MODE)
    assert checkpoint_dir is not None or True, "Neither checkpoint_dir or pretrained_model_dir was provided"
    assert pretrained_model_dir is None or checkpoint_dir is None, \
        "Only one of checkpoint_dir or pretrained_model_dir should be provided "
    config = BertConfig(vocab_size_or_config_json_file=vocab_size, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072, layer_wise_attention=layer_wise_attention,
                        mask_n_predictor=False)
    # config.hidden_dropout_prob = 0.2
    # QUESTION : WHERE IS THE MODEL ACTUALLY BEING LOADED ???
    # this line is useless apparently as it does it load it again
    if "normalize" in tasks:
        num_labels = vocab_size + 1
    else:
        num_labels = 1
        printing("MODEL : WE ARE STILL DEFINING a CLASSIFIER_1 but it should be left untrained ", verbose=verbose,verbose_level=1)
    if "pos" in tasks:
        assert voc_pos_size is not None

    if bert_module == "token_class":
        model = BertForTokenClassification(config, num_labels, dropout_classifier=dropout_classifier,
                                           num_labels_2=voc_pos_size)
    elif bert_module == "mlm":
        model = BertForMaskedLM(config)

    if pretrained_model_dir is not None and checkpoint_dir is None:
        if bert_module == "token_class":
            assert initialize_bpe_layer is not None, "ERROR initialize_bpe_layer should not be None "
        if bert_module == "token_class":
            model = model.from_pretrained(pretrained_model_dir, num_labels=num_labels, dropout_custom=dropout_bert)
        elif bert_module == "mlm":
            model = model.from_pretrained(pretrained_model_dir, normalization_mode=True,
                                          layer_wise_attention=layer_wise_attention, mask_n_predictor=mask_n_predictor)
            space_vector = torch.normal(torch.mean(model.bert.embeddings.word_embeddings.weight.data, dim=0), std=torch.std(model.bert.embeddings.word_embeddings.weight.data,dim=0)).unsqueeze(0)#torch.rand((1, 768)

            output_layer = torch.cat((model.bert.embeddings.word_embeddings.weight.data, space_vector), dim=0)
            model.cls.predictions.decoder = nn.Linear(model.bert.config.hidden_size, vocab_size + 1, bias=False)
            model.cls.predictions.decoder.weight = nn.Parameter(output_layer)
            model.cls.predictions.bias = nn.Parameter(torch.zeros(vocab_size + 1))

        printing("MODEL {} mode loading pretrained BERT and adding extra module for token classification based on {}",
                 var=[bert_module, pretrained_model_dir],
                 verbose=verbose,
                 verbose_level=1)
        if dropout_classifier is not None and bert_module == "token_class":
            model.dropout = nn.Dropout(dropout_classifier)
            printing("MODEL : SETTING DROPOUT CLASSIFIER TO {}".format(dropout_classifier), verbose=verbose, verbose_level=1)

        #if (len(tasks) > 1 and "normalize" in tasks:
        if "pos" in tasks and bert_module in ["token_class", "mlm"]:
            #assert tasks[1] == "pos", "ONLY  POS and normalize supported so far"
            model.classifier_task_2 = nn.Linear(model.bert.config.hidden_size, voc_pos_size)
            model.num_labels_2 = voc_pos_size
            printing("MODEL : adding classifier_task_2 for POS with voc_pos_size {}",
                     var=[voc_pos_size], verbose=verbose, verbose_level=1)

        if initialize_bpe_layer and tasks[0] == "normalize" and bert_module == "token_class":
            # it needs to be the first as we are setting classifier (1) to the normalizaiton classifier
            output_layer = torch.cat((model.bert.embeddings.word_embeddings.weight.data, torch.rand((1, 768))),dim=0)
            model.classifier_task_1.weight = nn.Parameter(output_layer)
            printing("MODEL : initializing output normalization layer with embedding layer + extra token ",
                     verbose=verbose,
                     verbose_level=1)
        if freeze_parameters and freeze_layer_prefix_ls is not None:
            model = freeze_param(model, freeze_layer_prefix_ls=freeze_layer_prefix_ls, verbose=verbose)

    elif checkpoint_dir is not None:
        assert initialize_bpe_layer is None, \
            "ERROR initialize_bpe_layer should b None as loading from existing checkpoint"

        if bert_module == "mlm":
            model.normalization_module = True
            print("SETTING model.layer_wise_attention  to model.layer_wise_attention ", model.layer_wise_attention,
                  layer_wise_attention)
            model.cls.predictions.decoder = nn.Linear(model.bert.config.hidden_size, vocab_size + 1, bias=False)
            model.cls.predictions.bias = nn.Parameter(torch.zeros(vocab_size + 1))

        model.load_state_dict(torch.load(checkpoint_dir, map_location=lambda storage, loc: storage))
        if add_task_2_for_downstream:
            model.classifier_task_2 = nn.Linear(model.bert.config.hidden_size, voc_pos_size)
            model.num_labels_2 = voc_pos_size
            print("MODEL : adding module task 2")
        printing("MODEL : loading model BERT+token classification pretrained from checkpoint {} on tasks {}",
                 var=[checkpoint_dir, tasks],
                 verbose=verbose, verbose_level=1)
    else:
        raise(Exception("neither checkpoint dir nor pretrained_model_dir was provided"))

    return model


