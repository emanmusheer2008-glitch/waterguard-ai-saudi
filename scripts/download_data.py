"""Download the two BattLeDIM 2018 files used by WaterGuard from Zenodo (CC BY 4.0)."""
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
RECORD = "https://zenodo.org/records/4017659/files/{name}?download=1"
FILES = ["2018_Leakages.csv", "2018_SCADA.xlsx"]

DATA.mkdir(exist_ok=True)
for name in FILES:
    target = DATA / name
    if target.exists():
        print(f"{name}: already present, skipping")
        continue
    print(f"Downloading {name} ...")
    try:
        urllib.request.urlretrieve(RECORD.format(name=name), target)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Failed ({exc}). Download manually from https://doi.org/10.5281/zenodo.4017659 into data/.")
    print(f"  saved {target.stat().st_size / 1e6:.1f} MB")
print("Done. Please cite the dataset (see data/README.md).")
