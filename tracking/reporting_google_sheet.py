from io_.info_print import printing
from env.project_variables import CLIENT_GOOGLE_CLOUD, SHEET_NAME_DEFAULT, TAB_NAME_DEFAULT
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
# use creds to create a client to interact with the Google Drive API

SCOPES_GOOGLE_SHEET = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']


creds = ServiceAccountCredentials.from_json_keyfile_name(os.path.join(CLIENT_GOOGLE_CLOUD ,'client.json'), SCOPES_GOOGLE_SHEET)
# Extract and print all of the values


def open_client(credientials=creds,sheet_name=SHEET_NAME_DEFAULT, tab_name=TAB_NAME_DEFAULT):
    client = gspread.authorize(credientials)
    sheet = client.open(sheet_name)
    sheet = sheet.worksheet(tab_name)
    return sheet, sheet_name, tab_name


def append_reporting_sheet(git_id,rioc_job, description, log_dir, target_dir, env, status,
                           verbose=1):
    sheet, sheet_name, tab_name = open_client()
    # Find a workbook by name and open the first sheet
    # Make sure you use the right name here.
    #worksheet_list = sheet.worksheets()
    sheet.append_row([git_id, rioc_job, description, log_dir, target_dir, env, status, None, None, None, None,"-"])
    list_of_hashes = sheet.get_all_records()
    printing("REPORT : Appending report to page {} in sheet {} of {} rows and {} columns ",
             var=[tab_name, sheet_name, len(list_of_hashes)+1, len(list_of_hashes[0])],
             verbose=verbose,
             verbose_level=1)
    return len(list_of_hashes)+1, len(list_of_hashes[0])


def update_status(row, new_status, col_number=7, verbose=1):
    sheet, sheet_name, tab_name = open_client()
    sheet.update_cell(row, col_number, new_status)
    printing("REPORT : job status updated in sheet with {} ", var=[new_status], verbose=verbose, verbose_level=1)

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


# TODO
# A
## - add this in a try except to grid run so that it rights each grid analysis run
###  --> except should be modifying cell from running to failed, add target dir and report dir to what it should
## - add in reporting script update cell to point to report
# B :  with taskfarm
## - add a write update_row before the task farm script
## add an update one at the end with report
