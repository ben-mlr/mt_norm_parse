
def sanity_check_early_stop_metric(early_stoppin_metric, task_parameter_setting, tasks):
    for task in tasks:
        for metrics in task_parameter_setting[task]["eval_metrics"]:
            test = (early_stoppin_metric in metrics)
            if test:
                break
    assert test, "ERROR : early_stoppin_metric {} not in task_parameter_setting".format(early_stoppin_metric)
