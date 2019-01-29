
AVAILABLE_SCHEDULING_POLICIES = [None, "policy_1"]


def policy_1(epoch):

    return  [{"epoch_start": 0, "epoch_stop": 20,
             "weight_binary_loss": 0.05,
             "ponderation_normalize_loss": 0,
              "multi_task_mode": "norm_not_norm"},
             {"epoch_start": 20, "epoch_stop": 80,
              "weight_binary_loss": 0.05*(80-epoch)/60,
              "ponderation_normalize_loss": 0.1*(epoch+20)/60,
              "multi_task_mode": "all"},
             ]