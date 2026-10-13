"""Download the Favorita files into DATA_DIR if they are not already there.

The official source is the Kaggle competition "Store Sales - Time Series
Forecasting". Kaggle requires a personal API token, so this script uses the
public Hugging Face copy of those same CSVs. holidays_events.csv is copied
from the repository root when that file is already present.
"""

import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA_DIR

MIRROR = "https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting/resolve/main"
FILES = ["train.csv", "test.csv", "stores.csv", "oil.csv", "transactions.csv", "holidays_events.csv"]


def download(url, dest):
    print(f"downloading {dest.name}")
    urllib.request.urlretrieve(url, dest)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    root_holidays = ROOT / "holidays_events.csv"
    for name in FILES:
        dest = DATA_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"already present: {dest}")
            continue
        if name == "holidays_events.csv" and root_holidays.exists():
            shutil.copy(root_holidays, dest)
            print(f"copied {root_holidays} -> {dest}")
            continue
        download(f"{MIRROR}/{name}", dest)
        print(f"saved {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
