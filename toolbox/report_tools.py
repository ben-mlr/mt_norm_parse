


def get_score(scores, metric, info_score, task, data):
    for report in scores:
        if report["metric"] == metric and report["info_score"] == info_score and report["task"] == task and report[
            "data"] == data:
            return report
    raise (Exception(
        "REPORT with {} metric {} info_score {} task and {} data not found in {} ".format(metric, info_score, task, data,
                                                                                         scores)))