from pathlib import Path
import pandas as pd

path = Path("data/2018_Leakages.csv")

# Correct BattLeDIM CSV format:
# ; separates columns
# , is the decimal separator
df = pd.read_csv(
    path,
    sep=";",
    decimal=",",
    parse_dates=["Timestamp"]
)

print("=" * 70)
print("WATERGUARD AI — LEAK ANALYSIS")
print("=" * 70)

print("\nDataset shape:", df.shape)
print("Time range:", df["Timestamp"].min(), "to", df["Timestamp"].max())

leak_cols = [c for c in df.columns if c != "Timestamp"]

print("\nLeak locations:", len(leak_cols))
print(leak_cols)

# Any positive leakage at a timestamp = leak active
df["leak_active"] = (df[leak_cols] > 0).any(axis=1)

print("\n--- LEAK ACTIVITY ---")
print("Normal timestamps:", (~df["leak_active"]).sum())
print("Leak timestamps:", df["leak_active"].sum())
print("Leak percentage:", round(df["leak_active"].mean() * 100, 2), "%")

print("\n--- EACH LEAK LOCATION ---")
for col in leak_cols:
    active = df[col] > 0

    if active.any():
        print(
            f"{col:6} | "
            f"start: {df.loc[active, 'Timestamp'].min()} | "
            f"end: {df.loc[active, 'Timestamp'].max()} | "
            f"active rows: {active.sum():6} | "
            f"max leak: {df[col].max():.3f}"
        )

print("\n--- NUMBER OF SIMULTANEOUS LEAKS ---")
df["active_leak_count"] = (df[leak_cols] > 0).sum(axis=1)
print(df["active_leak_count"].value_counts().sort_index())

print("\n--- SAMPLE ACTIVE LEAK ROWS ---")
print(df[df["leak_active"]].head(10).to_string(index=False))

print("\n" + "=" * 70)
print("LEAK ANALYSIS COMPLETE")
print("=" * 70)
