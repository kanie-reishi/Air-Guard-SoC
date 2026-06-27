"""Download and cache UCI Air Quality dataset (De Vito et al.)."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/360/"
    "air+quality.zip"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_NAME = "AirQualityUCI.csv"


def _extract_csv_from_zip(zip_path: Path, dest_csv: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise RuntimeError(f"No CSV found inside {zip_path}")
        # Prefer AirQualityUCI.csv if present
        target = next(
            (m for m in members if "airqualityuci" in m.lower().replace(" ", "")),
            members[0],
        )
        with zf.open(target) as src, open(dest_csv, "wb") as dst:
            shutil.copyfileobj(src, dst)


def download_air_quality(data_dir: Path | None = None, force: bool = False) -> Path:
    """Download UCI Air Quality dataset; return path to CSV."""
    data_dir = data_dir or DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / CSV_NAME

    if csv_path.exists() and not force:
        print(f"Dataset already cached: {csv_path}")
        return csv_path

    zip_path = data_dir / "air_quality.zip"
    print(f"Downloading from {UCI_ZIP_URL} ...")
    try:
        urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)
        _extract_csv_from_zip(zip_path, csv_path)
        print(f"Saved: {csv_path}")
    except Exception as exc:
        if csv_path.exists():
            print(f"Download failed ({exc}); using existing {csv_path}")
            return csv_path
        raise RuntimeError(
            "Could not download dataset. Place AirQualityUCI.csv manually in "
            f"{data_dir} (from https://archive.ics.uci.edu/dataset/360/air+quality)"
        ) from exc
    finally:
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)

    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UCI Air Quality dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory to store CSV",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()
    path = download_air_quality(args.data_dir, force=args.force)
    print(f"Ready: {path}")


if __name__ == "__main__":
    main()
