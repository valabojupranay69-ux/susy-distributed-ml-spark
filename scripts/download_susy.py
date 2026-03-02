

from pathlib import Path
from urllib.request import Request, urlopen
import zipfile
import sys

SUSY_ZIP_URL = "https://archive.ics.uci.edu/static/public/279/susy.zip"

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ZIP_PATH = RAW / "susy.zip"

CHUNK = 1024 * 1024 


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r, open(dest, "wb") as f:
        total = 0
        while True:
            b = r.read(CHUNK)
            if not b:
                break
            f.write(b)
            total += len(b)
            print(f"\rDownloaded: {total/1024/1024:.1f} MB", end="", flush=True)
    print("\nSaved to:", dest)


def unzip(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    print("Unzipped to:", out_dir)


def main() -> int:
    print("Project root:", ROOT)
    print("Downloading:", SUSY_ZIP_URL)

    download(SUSY_ZIP_URL, ZIP_PATH)
    unzip(ZIP_PATH, RAW)

    gz = RAW / "SUSY.csv.gz"
    if gz.exists():
        print(" Found:", gz)
        print("Size (MB):", gz.stat().st_size / 1024 / 1024)
        return 0

    print(" ERROR: SUSY.csv.gz not found. Contents of data/raw:")
    for p in RAW.iterdir():
        print(" -", p.name)
    return 1


if __name__ == "__main__":
    sys.exit(main())
