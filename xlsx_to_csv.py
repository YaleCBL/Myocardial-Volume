#!/usr/bin/env python3

import pandas as pd
import os

def xlsx_to_csvs(excel_path, output_dir=None):
    # Create output directory if not specified
    if output_dir is None:
        output_dir = os.path.dirname(excel_path)
    os.makedirs(output_dir, exist_ok=True)

    # Load all sheets
    xls = pd.ExcelFile(excel_path)
    print(f"Found sheets: {xls.sheet_names}")

    # Iterate through each sheet
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        # Sanitize sheet name for filename
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in sheet_name)
        csv_path = os.path.join(output_dir, f"{safe_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Exported '{sheet_name}' to '{csv_path}'")

# export csv's
in_folder = "/Users/pfaller/Documents/Lab/people/john_stendahl/data_xlsx"
fnames = ["DSEA08 Data for Martin cleaned.xlsx",]
# fnames = ["DSEA16 Data for Martin_multicycle cleaned.xlsx"]

for fname in fnames:
    xlsx_to_csvs(os.path.join(in_folder, fname), "data")
