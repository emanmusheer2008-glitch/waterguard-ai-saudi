"""Download the BattLeDIM files used by WaterGuard from Zenodo (CC BY 4.0).

  2018_SCADA.xlsx    model inputs (92 MB)
  2018_Leakages.csv  ground truth, used for the target and evaluation only (6 MB)
  L-TOWN.inp         EPANET network model, used only to draw the network map and to
                     check the inspection guidance against the true leak locations (0.4 MB)

Usage:  python scripts/download_data.py
"""
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
RECORD = "https://zenodo.org/records/4017659/files/{name}?download=1"
FILES = {  # name -> approximate size in bytes (sanity check against truncated downloads)
    "2018_Leakages.csv": 6_234_468,
    "2018_SCADA.xlsx": 92_174_099,
    "L-TOWN.inp": 412_200,
}


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for name, size in FILES.items():
        target = DATA / name
        if target.exists():
            print(f"{name}: already present, skipping")
            continue
        print(f"Downloading {name} ...")
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            urllib.request.urlretrieve(RECORD.format(name=name), tmp)
        except Exception as exc:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            sys.exit(f"Failed ({exc}). Download manually from https://doi.org/10.5281/zenodo.4017659 into data/.")
        if abs(tmp.stat().st_size - size) > 0.01 * size:
            sys.exit(f"{name}: unexpected size {tmp.stat().st_size} bytes (expected ~{size}); kept as {tmp.name}")
        tmp.rename(target)
        print(f"  saved {target.stat().st_size / 1e6:.1f} MB")
    print("Done. Please cite the dataset (see data/README.md).")


if __name__ == "__main__":
    main()
