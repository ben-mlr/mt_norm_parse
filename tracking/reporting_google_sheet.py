import sys
sys.path.insert(0, ".")
from env.importing import *
from io_.info_print import printing
from env.project_variables import CLIENT_GOOGLE_CLOUD, SHEET_NAME_DEFAULT, TAB_NAME_DEFAULT, PROJECT_PATH

# use creds to create a client to interact with the Google Drive API

SCOPES_GOOGLE_SHEET = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

REPORTED_VARIABLE_PER_SHEET = {"experiments_tracking": {"git_id": 1,
                                                        "job_id": 2,
                                                        "tasks": 3,
                                                        "description": 4,
                                                        "logs_dir": 5,
                                                        "target_dir": 6,
                                                        "env": 7,
                                                        "completion": 8,
                                                        "evaluation_dir": 9,
                                                        "tensorboard_dir": 10}
                               }

try:
    CLIENT_GOOGLE_CLOUD = os.path.join(PROJECT_PATH, "tracking/google_api")
    creds = ServiceAccountCredentials.from_json_keyfile_name(os.path.join(CLIENT_GOOGLE_CLOUD, 'client.json'), SCOPES_GOOGLE_SHEET)
    # Extract and print all of the values
    SHEET_NAME_DEFAULT, TAB_NAME_DEFAULT = "model_evaluation", "experiments_tracking"
except Exception as e:
    print(e)

def open_client(credientials=creds, sheet_name=SHEET_NAME_DEFAULT, tab_name=TAB_NAME_DEFAULT):
    client = gspread.authorize(credientials)
    sheet = client.open(sheet_name)
    sheet = sheet.worksheet(tab_name)
    return sheet, sheet_name, tab_name


def append_reporting_sheet(git_id, tasks, rioc_job, description, log_dir, target_dir, env, status,
                           verbose=1):

    sheet, sheet_name, tab_name = open_client()
    # Find a workbook by name and open the first sheet
    # Make sure you use the right name here.
    #worksheet_list = sheet.worksheets()
    if not rioc_job.startswith("local"):
        sheet.append_row([git_id,  rioc_job, tasks, description, log_dir, target_dir, env, status, None, None, None, None,"-"])
        list_of_hashes = sheet.get_all_records()
        printing("REPORT : Appending report to page {} in sheet {} of {} rows and {} columns ",
                 var=[tab_name, sheet_name, len(list_of_hashes)+1, len(list_of_hashes[0])],
                 verbose=verbose,
                 verbose_level=1)
    else:
        list_of_hashes=["NOTHING"]
    return len(list_of_hashes)+1, len(list_of_hashes[0])


def update_status(row, value, col_number=8, sheet=None, verbose=1):
    if sheet is None:
        sheet, sheet_name, tab_name = open_client()
    if value is not None:
        sheet.update_cell(row, col_number, value)
        printing("REPORT : col {} updated in sheet with {} ", var=[col_number, value], verbose=verbose, verbose_level=1)


git_id = "aaa"
rioc= "XXX"
description = "test"
log_dir = "--"
target_dir = "--"
env = "rioc"
completion = "running"
#row, col = append_reporting_sheet(git_id, rioc, description, log_dir, target_dir, env, completion)
#print(col, row)
#update_status(row, "algright")


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--new_report", type=int, required=True, help="1 means new 0 means will append it")
    args.add_argument("--git_id", required=False)
    args.add_argument("--job_id", type=str, default="UNK")
    args.add_argument("--description", type=str)
    args.add_argument("--log_dir", type=str)
    args.add_argument("--target_dir", type=str)
    args.add_argument("--env", type=str)
    args.add_argument("--completion", type=str)
    args.add_argument("--tasks", type=str, required=False)
    args.add_argument("--evaluation_dir", type=str)
    args.add_argument("--tensorboard_dir", type=str)
    args.add_argument("--row_sheet", type=int)
    args = args.parse_args()

    if args.new_report:
        # create with job id; log dir and completion : 'starting'  and env
        append_reporting_sheet(rioc_job=args.job_id,
                               git_id=args.git_id,
                               tasks=args.tasks,
                               description=args.description, log_dir=args.log_dir,
                               target_dir=args.target_dir,
                               env=args.env,
                               status="starting")
    else:
        assert args.row_sheet is not None, "ERROR : --row_sheet missing "
        sheet, sheet_name, tab_name = open_client(sheet_name=SHEET_NAME_DEFAULT, tab_name=TAB_NAME_DEFAULT)
        update_status(row=args.row_sheet, value=args.git_id, col_number=REPORTED_VARIABLE_PER_SHEET["experiments_tracking"]["git_id"], sheet=sheet)
        update_status(row=args.row_sheet, value=args.description, col_number=REPORTED_VARIABLE_PER_SHEET["experiments_tracking"]["description"], sheet=sheet)
        update_status(row=args.row_sheet, value=args.target_dir, col_number=REPORTED_VARIABLE_PER_SHEET["experiments_tracking"]["target_dir"], sheet=sheet)
        update_status(row=args.row_sheet, value=args.completion, col_number=REPORTED_VARIABLE_PER_SHEET["experiments_tracking"]["completion"], sheet=sheet)
        update_status(row=args.row_sheet, value=args.tasks, col_number=REPORTED_VARIABLE_PER_SHEET["experiments_tracking"]["tasks"], sheet=sheet)
        update_status(row=args.row_sheet, value=args.evaluation_dir, col_number=REPORTED_VARIABLE_PER_SHEET["experiments_tracking"]["evaluation_dir"], sheet=sheet)
        update_status(row=args.row_sheet, value=args.tensorboard_dir, col_number=REPORTED_VARIABLE_PER_SHEET["experiments_tracking"]["tensorboard_dir"], sheet=sheet)
