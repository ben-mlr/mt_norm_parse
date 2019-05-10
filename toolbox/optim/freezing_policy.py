from env.importing import pdb
import toolbox.deep_learning_toolbox as dptx
from io_.info_print import printing
from env.project_variables import AVAILABLE_BERT_FINE_TUNING_STRATEGY


def apply_fine_tuning_strategy(fine_tuning_strategy, model, epoch, lr_init, betas=None,verbose=1):
    assert fine_tuning_strategy in AVAILABLE_BERT_FINE_TUNING_STRATEGY, "{} not in {}".format(fine_tuning_strategy, AVAILABLE_BERT_FINE_TUNING_STRATEGY)

    if fine_tuning_strategy in ["standart", "bert_out_first", "only_first_and_last"]:
        assert isinstance(lr_init, float), "{} lr : type {}".format(lr_init, type(lr_init))
        optimizer = [dptx.get_optimizer(model.parameters(), lr=lr_init, betas=betas)]
        printing("TRAINING : fine tuning strategy {} : learning rate constant {} betas {}", var=[fine_tuning_strategy, lr_init, betas],
                 verbose_level=1, verbose=verbose)
    elif fine_tuning_strategy == "flexible_lr":

        assert isinstance(lr_init, dict), "lr_init should be dict in {}".format(fine_tuning_strategy)
        # sanity check
        optimizer = []
        n_all_layers = len([a for a,_ in model.named_parameters()])
        n_optim_layer = 0
        for pref, lr in lr_init.items():
            #param_group = filter(lambda p: p[0].startswith(pref), model.named_parameters())
            param_group = [param for name, param in model.named_parameters() if name.startswith(pref)]
            n_optim_layer += len(param_group)
            optimizer.append(dptx.get_optimizer(param_group, lr=lr, betas=betas))
        assert n_all_layers == n_optim_layer, \
            "ERROR : You are missing some layers in the optimization n_all {} n_optim {} ".format(n_all_layers,
                                                                                                  n_optim_layer)

        printing("TRAINING : fine tuning strategy {} : learning rate constant : {} betas {}", var=[fine_tuning_strategy, lr_init, betas],
                 verbose_level=1, verbose=verbose)

    if fine_tuning_strategy == "bert_out_first":
        info_add = ""
        if epoch == 0:
            info_add = "not"
            freeze_layer_prefix_ls = "bert"
            model = dptx.freeze_param(model, freeze_layer_prefix_ls, verbose=verbose)
        printing("TRAINING : fine tuning strategy {} : {} freezing bert for epoch {}" \
                 .format(fine_tuning_strategy, info_add, epoch), verbose_level=1, verbose=verbose)
    elif fine_tuning_strategy == "only_first_and_last":
        model = dptx.freeze_param(model, ["embeddings", "classifier"], verbose=verbose)

    return model, optimizer
