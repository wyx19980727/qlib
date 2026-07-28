import sys, os
sys.path.insert(0, "/home/albus/Python_Codes/qlib")
os.makedirs("/tmp/debug_csv", exist_ok=True)
# clean old file
for f in os.listdir("/tmp/debug_csv"):
    os.remove(os.path.join("/tmp/debug_csv", f))

from scripts.albus_produce_data import process_symbol_interval
process_symbol_interval("2026-05-22", "00085", "/tmp/debug_csv", 1, "1min")

import pandas as pd
df = pd.read_csv("/tmp/debug_csv/00085.csv")
print("NaN per column:")
print(df.isna().sum())
print(f"\nTotal rows: {len(df)}")
print("\nFirst 20 rows:")
print(df[["date","open","close","volume","paused"]].head(20))
print("\nLast 20 rows:")
print(df[["date","open","close","volume","paused"]].tail(20))
