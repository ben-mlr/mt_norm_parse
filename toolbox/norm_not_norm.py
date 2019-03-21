from env.project_variables import AVAILABLE_TASKS, MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE
from io_.info_print import printing
from toolbox.sanity_check import sanity_check_loss_poneration


def get_label_norm(norm):
    if norm == 0:
        return "NEED_NORM"
    elif norm == 1:
        return "NORMED"


def schedule_training(multi_task_loss_ponderation, tasks=None):
    # TODO : add consistency assertion between multi_task_loss_ponderation and tasks
    # TWO MODES : 1 multi_task_loss_ponderation string to point to predefine loss ponderation else it has to be a dictionary that respect a template
    #AVAILABLE_TASKS
    if multi_task_loss_ponderation == "all":
        norm_not_norm, normalize, pos = 1, 1, 0.01
    elif multi_task_loss_ponderation == "pos":
        norm_not_norm, normalize, pos = None, None, 1
    elif multi_task_loss_ponderation == "normalize":
        norm_not_norm, normalize, pos = 0, 1, None
    elif multi_task_loss_ponderation == "norm_not_norm":
        norm_not_norm, normalize, pos = 1, 0, None
    elif multi_task_loss_ponderation == "uniform":
        norm_not_norm, normalize, pos = 1, 1, 1
    elif multi_task_loss_ponderation == "normalization_100":
        norm_not_norm, normalize, pos = 1, 1, 0.01
    elif multi_task_loss_ponderation == "pos_100":
        norm_not_norm, normalize, pos = 1, 1, 0.01
    elif multi_task_loss_ponderation == "norm_not_norm":
        norm_not_norm, normalize, pos = 1, 0, None
    elif isinstance(multi_task_loss_ponderation, dict):
        return multi_task_loss_ponderation
    else:
        raise(Exception("multi_task_loss_ponderation {} not found as predefined mode neither a dict ".format(multi_task_loss_ponderation)))
    return {"pos": pos, "normalize": normalize, "norm_not_norm": norm_not_norm}


def scheduling_policy(phases_ls, epoch, tasks, verbose=1):

    if phases_ls is None:
        ponderation = 1
        weight_binary_loss = 1
        weight_pos_loss = 1
        if len(tasks) == 1:
            mode = tasks[0]
        else:
            mode = "all"
        printing("WARNING : default policy scheduling (no scheduling {} ponderation_normalize_loss,  {}"
                 " weight_binary_loss and pos {} : LOSS MODE SET TO {} ", var=[ponderation, weight_binary_loss, weight_pos_loss, mode],
                 verbose_level=1, verbose=verbose)
        return mode, ponderation, weight_binary_loss, weight_pos_loss
    for phase in phases_ls:

        assert phase.get("epoch_start") is not None
        assert phase.get("epoch_stop") is not None
        assert phase.get("epoch_stop") > phase.get("epoch_start")
        assert phase.get("weight_binary_loss") is not None
        assert phase.get("ponderation_normalize_loss") is not None
        assert phase.get("multi_task_mode") in AVAILABLE_TASKS

        if phase["epoch_start"] <= epoch < phase["epoch_stop"]:
            return phase["multi_task_mode"], phase["ponderation_normalize_loss"], phase["weight_binary_loss"]