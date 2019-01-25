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
        return 1, 1
    elif multi_task_mode == "normalize":
        return 0, 1
    elif multi_task_mode == "norm_not_norm":
        return 1, 0


def scheduling_policy(phases_ls, epoch, verbose=1):

    if phases_ls is None:
        printing("WARNING : default policy scheduling (no scheduling", verbose_level=1, verbose=verbose)
        return "all", 1, 0.01
    for phase in phases_ls:
        assert phase.get("epoch_start") is not None
        assert phase.get("epoch_stop") is not None
        assert phase.get("epoch_stop") > phase.get("epoch_start")
        assert phase.get("weight_binary_loss") is not None
        assert phase.get("ponderation_normalize_loss") is not None
        assert phase.get("multi_task_mode") in AVAILABLE_TASKS

        if phase["epoch_start"] <= epoch < phase["epoch_stop"]:
            return phase["multi_task_mode"], phase["ponderation_normalize_loss"], phase["weight_binary_loss"]