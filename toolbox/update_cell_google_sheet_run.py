
from tracking.reporting_google_sheet import update_status

if __name__=="__main__":
    import argparse

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--row", required=True, type=int)
    arg_parse.add_argument("--new_cell", required=True, type=str)
    arg_parse.add_argument("--col", default=8, type=int)
    args = arg_parse.parse_args()
    update_status(args.row, args.new_cell, col_number=args.col)
