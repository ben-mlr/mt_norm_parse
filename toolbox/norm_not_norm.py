from env.project_variables import AVAILABLE_TASKS
from io_.info_print import printing


def get_label_norm(norm):
    if norm == 0:
        return "NEED_NORM"
    elif norm == 1:
        return "NORMED"


def schedule_training(multi_task_mode):
    assert multi_task_mode in AVAILABLE_TASKS
    if multi_task_mode == "all":
        return 1, 1, 0.01
    elif multi_task_mode == "pos":
        return None, None, 1
    elif multi_task_mode == "normalize":
        return 0, 1, None
    elif multi_task_mode == "norm_not_norm":
        return 1, 0, None


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