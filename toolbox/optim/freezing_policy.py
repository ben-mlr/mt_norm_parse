import toolbox.deep_learning_toolbox as dptx
from io_.info_print import printing
from env.project_variables import AVAILABLE_BERT_FINE_TUNING_STRATEGY


def apply_fine_tuning_strategy(fine_tuning_strategy, model, epoch, lr_init, verbose):
    assert fine_tuning_strategy in AVAILABLE_BERT_FINE_TUNING_STRATEGY

    if fine_tuning_strategy in ["standart", "bert_out_first"]:
        optimizer = dptx.get_optimizer(model.parameters(), lr=lr_init)
        printing("TRAINING : fine tuning strategy {} : learning rate constant {}", var=[fine_tuning_strategy,lr_init],
                 verbose_level=1, verbose=verbose)
    if fine_tuning_strategy == "bert_out_first":
        info_add = ""
        if epoch != 0:
            info_add = "not"
            freeze_layer_prefix_ls = "bert"
            model = dptx.freeze_param(model, freeze_layer_prefix_ls, verbose=verbose)
        printing("TRAINING : fine tuning strategy {} : {} freezing bert for epoch {}" \
                 .format(fine_tuning_strategy, info_add, epoch))

    return model, optimizer
