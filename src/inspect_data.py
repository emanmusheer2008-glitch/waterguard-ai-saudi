from pathlib import Path
import pandas as pd

DATA = Path("data")
scada_path = DATA / "2018_SCADA.xlsx"
leak_path = DATA / "2018_Leakages.csv"

print("=" * 70)
print("WATERGUARD AI SAUDI — DATA AUDIT")
print("=" * 70)

print("\n[1] FILE CHECK")
for path in [scada_path, leak_path]:
    print(f"{path.name}: {'FOUND' if path.exists() else 'MISSING'}")
    if path.exists():
        print(f"  Size: {path.stat().st_size / 1024**2:.2f} MB")

if not scada_path.exists() or not leak_path.exists():
    raise FileNotFoundError("Required dataset files are missing from data/")

print("\n[2] SCADA WORKBOOK SHEETS")
xls = pd.ExcelFile(scada_path)
print(xls.sheet_names)

print("\n[3] INSPECTING EACH SCADA SHEET")
for sheet in xls.sheet_names:
    df = pd.read_excel(scada_path, sheet_name=sheet, nrows=5)

    print("\n" + "-" * 70)
    print("SHEET:", sheet)
    print("Columns:", list(df.columns))
    print("Preview:")
    print(df.to_string(index=False))

print("\n[4] LEAKAGE FILE")
# Fixed: the file uses ";" separators and "," decimals. The original call
# pd.read_csv(leak_path) failed with a ParserError at line 2312.
leaks = pd.read_csv(leak_path, sep=";", decimal=",", parse_dates=["Timestamp"])

print("Shape:", leaks.shape)
print("Columns:", list(leaks.columns))

print("\nFirst 10 leakage records:")
print(leaks.head(10).to_string(index=False))

print("\nMissing values:")
print(leaks.isna().sum())

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
