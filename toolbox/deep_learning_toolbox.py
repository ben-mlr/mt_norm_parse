from env.importing import *

from env.project_variables import AVAILABLE_OPTIMIZER
from io_.info_print import printing


def get_optimizer(parameters, lr, optimizer="adam", betas=None, verbose=1):

    assert optimizer in AVAILABLE_OPTIMIZER, "ERROR optimizers supported are {} ".format(AVAILABLE_OPTIMIZER)
    if optimizer == "adam":
        if betas is None:
            # betas = (0.9, 0.9)
            print("DEFAULT betas:", betas)
        opt = torch.optim.Adam(parameters, lr=lr, #betas=betas,
                               eps=1e-9)
    elif optimizer == "SGD":
        assert betas is None, "ERROR "
        opt = torch.optim.SGD(parameters, lr=lr)
    elif optimizer == "bahdanu-adadelta":
        assert betas is None, "ERROR betas not supported for optimizer {}".format(optimizer)
        opt = torch.optim.Adadelta(parameters, eps=10e-6, rho=0.95)
    printing("TRAINING : optimizer {} has been reloaded with lr {} betas {} ", var=[optimizer, lr, betas], verbose=verbose, verbose_level=1)

    return opt


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_cumulated_list(sent_len):
    sent_len_cumulated = [0]
    cumu = 0
    for len_sent in sent_len:
        cumu += int(len_sent)
        sent_len_cumulated.append(cumu)
    return sent_len_cumulated


def freeze_param(model, freeze_layer_prefix_ls=None, not_freeze_layer_prefix_ls=None,verbose=1):
    freezing_layer = 0

    if not_freeze_layer_prefix_ls is None:
        not_freeze_layer_prefix_ls = []
    if freeze_layer_prefix_ls is None:
        freeze_layer_prefix_ls = []
    for name, param in model.named_parameters():
        for prefix in freeze_layer_prefix_ls:
            if name.startswith(prefix):
                param.requires_grad = False
                freezing_layer += 1
                printing("TRAINING : freezing {} parameter ", var=[name], verbose=verbose, verbose_level=2)
        to_freeze = 0
        for prefix in not_freeze_layer_prefix_ls:
            if not name.startswith(prefix):
                to_freeze += 1
            if not to_freeze == len(not_freeze_layer_prefix_ls):
                param.requires_grad = False
                freezing_layer += 1
                printing("TRAINING :- freezing {} parameter ", var=[name], verbose=verbose, verbose_level=1)
    printing("TRAINING : freezing {} layers : {} prefix , not freezing {} ",
             var=[freezing_layer, freeze_layer_prefix_ls, not_freeze_layer_prefix_ls],
             verbose=verbose,
             verbose_level=1)
    assert freezing_layer > 0, "ERROR : did not fine any layers starting with {}".format(prefix)

    return model


def dropout_input_tensor(input_tokens_tensor, mask_token_index, sep_token_index, dropout, cls_token_index=None, pad_index=None,
                         apply_dropout=None, applied_dropout_rate=None):
    if apply_dropout is None:
        assert applied_dropout_rate is not None
        apply_dropout = np.random.random() < applied_dropout_rate
    droping_multiplier_input_tokens_tensor = torch.zeros_like(input_tokens_tensor).bernoulli_(1 - dropout)
    droping_multiplier_input_tokens_tensor[input_tokens_tensor == sep_token_index] = 1
    if cls_token_index is not None:
        droping_multiplier_input_tokens_tensor[input_tokens_tensor == cls_token_index] = 1
        droping_multiplier_input_tokens_tensor[input_tokens_tensor == pad_index] = 1
    # we mask all the tokens which got droping_multiplier_input_tokens_tensor 0
    if apply_dropout:
        input_tokens_tensor[droping_multiplier_input_tokens_tensor == 0] = mask_token_index
    return input_tokens_tensor, droping_multiplier_input_tokens_tensor, apply_dropout
