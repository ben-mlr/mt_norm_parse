# train_path = LIU_TRAIN
# dev_path = LIU_DEV
# test_path = TEST#[TEST, CP_WR_PASTE_TEST_269]
from io_.info_print import printing
import os

from training.train_eval import train_eval
from training.fine_tune import fine_tune
from toolbox.grid_tool import grid_param_label_generate
from env.project_variables import PROJECT_PATH, TRAINING,LIU_TRAIN, DEMO_SENT, CP_WR_PASTE_TEST_269, \
    LIU_DEV, DEV, DIR_TWEET_W2V, TEST, DIR_TWEET_W2V, CHECKPOINT_DIR, DEMO, DEMO2, CP_PASTE_WR_TRAIN, \
    CP_WR_PASTE_DEV, CP_WR_PASTE_TEST, CP_PASTE_DEV, CP_PASTE_TRAIN, CP_PASTE_TEST, EWT_DEV, EWT_TEST, \
    LIU_DEV_SENT, LIU_TRAIN_SENT, DEV_SENT, TEST_SENT, DEMO_SENT, TRAINING_DEMO, EN_LINES_EWT_TRAIN, EN_LINES_DEV, EN_LINES_EWT_TRAIN, \
    MTNT_TOK_TRAIN, MTNT_TOK_DEV, MTNT_EN_FR_TRAIN, MTNT_EN_FR_DEV, MTNT_EN_FR_TEST, RUN_SCRIPTS_DIR, GPU_AVAILABLE_DEFAULT_LS

from toolbox.git_related import get_commit_id
from tracking.reporting_google_sheet import update_status, append_reporting_sheet


def fine_tune_run(fine_tune_label, model_full_name, n_epochs):
    description = "Fine tuning {}".format(model_full_name)
    OAR = os.environ.get('OAR_JOB_ID') + "_rioc-" if os.environ.get('OAR_JOB_ID', None) is not None else "local"
    print("OAR=", OAR)
    fine_tune_label = OAR + "-" + fine_tune_label

    row, col = append_reporting_sheet(git_id=get_commit_id(), rioc_job=OAR, description=description,
                                      log_dir="", target_dir="",
                                      env="", status="running fine tune",
                                      verbose=1)
    try:
        fine_tune(train_path=LIU_TRAIN, dev_path=LIU_DEV, evaluation=True, batch_size=100,
                  test_path=[LIU_TRAIN, LIU_DEV, TEST, CP_WR_PASTE_TEST_269], n_epochs=n_epochs,
                  fine_tune_label=fine_tune_label + "fine_tune_for_real-BACK_NORMALIZE-tenth",
                  model_full_name=model_full_name,
                  learning_rate=0.00005, freeze_ls_param_prefix=["char_embedding", "encoder", "bridge"],
                  tasks=["normalize"],
                  debug=False, verbose=1)
        update_status(row=row, new_status="done fine tune", verbose=1)
    except Exception as e:
        update_status(row=row, new_status="failed ERROR {} ".format(e),
                  verbose=1)

    to_enrich = "lr  char_decoding char_src_attention "
    to_analysed = to_enrich
    to_keep_only = ""
    print("GRID_INFO enrch vars= batch_size lr ", to_enrich)
    print("GRID_INFO analy vars=  ", to_analysed)
    print("GRID_INFO fixed vals=  batch_size,2 lr,0.0001 ", to_keep_only)


if __name__ == "__main__":
    n_epochs=1
    fine_tune_label = "fine_tuning"
    model_full_name = "99428_rioc--DEBUG_NO_LOSS_PADDING-0-model_1-model_1_8fb8"
    fine_tune_run(fine_tune_label=fine_tune_label, model_full_name=model_full_name,n_epochs=n_epochs)
