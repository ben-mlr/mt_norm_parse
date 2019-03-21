from env.project_variables import AVAILABLE_TASKS, MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE
import time
from io_.info_print import printing
def sanity_check_info_checkpoint(info_checkpoint, template):
    for key in template.keys():
        # git id is added on the fly as updated
        if key not in ["git_id", "other"]:
            assert key in info_checkpoint, "ERROR {} key is not in info_checkpoint".format(key)


def get_timing(former):
    if former is not None:
        return time.time() - former, time.time()
    else:
        return None, None


def sanity_check_loss_poneration(ponderation_dic, verbose=1):
    if isinstance(ponderation_dic, dict):
        for task in AVAILABLE_TASKS:
            if task != "all": # Still some ambiguity in 'all' setting
                assert task in ponderation_dic, "ERROR : task {} is not related to a ponderation while it should ".format(task)
    elif isinstance(ponderation_dic,str):
        assert ponderation_dic in MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE, "ERROR ponderation should be in {}".format(ponderation_dic,MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE)
        printing("WARNING : COULD NOT SANITY CHECK ponderation_dic {} ", var=[ponderation_dic], verbose=verbose,
                 verbose_level=1)
    else:
        raise(Exception("ponderation_dic is neither string or dict {}".format(ponderation_dic)))