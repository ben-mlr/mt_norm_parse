from env.tasks_settings import TASKS_PARAMETER
from io_.info_print import printing
from test_._test_early_stoping import sanity_check_early_stop_metric


def get_early_stopping_metric(tasks, verbose, main_task=None, early_stoppin_metric=None):
    if main_task is None:
        printing("INFO : default main task provided is the first of the list {} ", var=[tasks],
                 verbose=verbose, verbose_level=1)
        main_task = tasks[0]

    if early_stoppin_metric is None:
        early_stoppin_metric = TASKS_PARAMETER[main_task]["eval_metrics"][0][0]
        printing("INFO : default early_stoppin_metric is early_stoppin_metric  {} first one of "
                 "the first possible in TASK_PARAMETER", var=[early_stoppin_metric],
                 verbose=verbose, verbose_level=1)
    sanity_check_early_stop_metric(early_stoppin_metric, TASKS_PARAMETER, tasks)

    return early_stoppin_metric
