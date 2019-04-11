from env.importing import *

from env.project_variables import AVAILABLE_OPTIMIZER, SEED_TORCH
from io_.info_print import printing

torch.manual_seed(SEED_TORCH)


def get_optimizer(parameters, lr, optimizer="adam", betas=None, verbose=1):

    assert optimizer in AVAILABLE_OPTIMIZER, "ERROR optimizers supported are {} ".format(AVAILABLE_OPTIMIZER)
    if optimizer == "adam":
        if betas is None:
            betas = (0.9, 0.9)
            print("DOZAT INIT ADAM betas:", betas)
        opt = torch.optim.Adam(parameters, lr=lr, betas=betas, eps=1e-9)
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