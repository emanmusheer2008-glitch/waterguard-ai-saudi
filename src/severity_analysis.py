import pandas as pd
import numpy as np

leaks = pd.read_csv(
    "data/2018_Leakages.csv",
    sep=";",
    decimal=",",
    parse_dates=["Timestamp"]
)

leak_cols = [c for c in leaks.columns if c != "Timestamp"]

leaks["total_leak"] = leaks[leak_cols].sum(axis=1)
leaks["max_leak"] = leaks[leak_cols].max(axis=1)
leaks["active_count"] = (leaks[leak_cols] > 0).sum(axis=1)

print("=" * 70)
print("WATERGUARD — LEAK SEVERITY ANALYSIS")
print("=" * 70)

print("\nTOTAL LEAK DISTRIBUTION")
print(leaks["total_leak"].describe(
    percentiles=[.25,.50,.75,.80,.90,.95,.97,.99]
))

print("\nMAX INDIVIDUAL LEAK DISTRIBUTION")
print(leaks["max_leak"].describe(
    percentiles=[.50,.75,.80,.90,.95,.97,.99]
))

positive = leaks.loc[leaks["total_leak"] > 0, "total_leak"]

print("\nPOSITIVE LEAK VALUES ONLY")
print(positive.describe(
    percentiles=[.10,.25,.50,.75,.90,.95,.99]
))

# Change in total leakage over 5 minutes
leaks["leak_change"] = leaks["total_leak"].diff().fillna(0)

print("\nLARGEST LEAK INCREASES")
print(
    leaks.nlargest(20, "leak_change")[
        ["Timestamp","total_leak","leak_change","active_count"]
    ].to_string(index=False)
)

print("\nMONTHLY AVERAGE TOTAL LEAK")
leaks["month"] = leaks["Timestamp"].dt.month
print(leaks.groupby("month")["total_leak"].agg(["mean","max"]))

print("\nDone.")
