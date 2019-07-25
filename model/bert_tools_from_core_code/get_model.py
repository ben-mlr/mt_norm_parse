from env.importing import nn
from model.bert_normalize import get_bert_token_classification, make_bert_multitask
from io_.info_print import printing


def get_multi_task_bert_model(args, model_dir, vocab_size, voc_pos_size, debug, verbose, num_labels_per_task=None):
    if args.checkpoint_dir is None:
        # TODO vocab_size should be loaded from args.json
        # TEMPORARY : should eventually keep only : model = make_bert_multitask()
        if args.multitask:
            model = make_bert_multitask(pretrained_model_dir=model_dir, tasks=args.tasks, num_labels_per_task=num_labels_per_task)
        else:
            model = get_bert_token_classification(pretrained_model_dir=model_dir,
                                                  vocab_size=vocab_size,
                                                  freeze_parameters=args.freeze_parameters,
                                                  freeze_layer_prefix_ls=args.freeze_layer_prefix_ls,
                                                  dropout_classifier=args.dropout_classifier,
                                                  dropout_bert=args.dropout_bert,
                                                  tasks=args.tasks,
                                                  voc_pos_size=voc_pos_size,
                                                  bert_module=args.bert_module,
                                                  layer_wise_attention=args.layer_wise_attention,
                                                  mask_n_predictor=args.append_n_mask,
                                                  initialize_bpe_layer=args.initialize_bpe_layer,
                                                  debug=debug)
    else:
        printing("MODEL : reloading from checkpoint {} all models parameters are "
                 "ignored except task bert module and layer_wise_attention", var=[args.checkpoint_dir],
                 verbose_level=1, verbose=verbose)
        # TODO args.original_task  , vocab_size is it necessary
        original_task = ["normalize"]
        print("WARNING : HARDCODED add_task_2_for_downstream : True ")
        model = get_bert_token_classification(vocab_size=vocab_size,
                                              voc_pos_size=voc_pos_size,
                                              tasks=original_task,
                                              initialize_bpe_layer=None, bert_module=args.bert_module,
                                              layer_wise_attention=args.layer_wise_attention,
                                              mask_n_predictor=args.append_n_mask,
                                              add_task_2_for_downstream=True,
                                              checkpoint_dir=args.checkpoint_dir, debug=debug)
        add_task_2 = False
        if add_task_2:
            printing("MODEL : adding extra classifer for task_2  with {} label", var=[voc_pos_size],
                     verbose=verbose, verbose_level=1)
            model.classifier_task_2 = nn.Linear(model.bert.config.hidden_size, voc_pos_size)
            model.num_labels_2 = voc_pos_size
    return model
