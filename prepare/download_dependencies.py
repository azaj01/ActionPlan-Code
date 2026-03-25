"""
Download actionplan_deps.zip from Google Drive and extract at project root.

This script:
1. Downloads actionplan_deps.zip from Google Drive (via gdown or manual fallback)
2. Extracts it at the root of the ActionPlan-Code directory

Usage:
    python prepare/download_dependencies.py

Manual download (if gdown fails):
    https://drive.google.com/file/d/1q5xd3EARWCoel3iiUKo6YXRJUN34OvYQ/view?usp=drive_link
Requirements:
    - gdown: pip install gdown (for automatic download)
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path

# Project root = parent of prepare/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GDRIVE_ID = "1q5xd3EARWCoel3iiUKo6YXRJUN34OvYQ"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"
MANUAL_URL = f"https://drive.google.com/file/d/{GDRIVE_ID}/view?usp=drive_link"
ZIP_NAME = "actionplan_deps.zip"


def download_with_gdown(zip_path: Path) -> bool:
    """Download using gdown if available."""
    try:
        import gdown
        print(f"Downloading {ZIP_NAME} from Google Drive...")
        gdown.download(GDRIVE_URL, str(zip_path), quiet=False, fuzzy=True)
        return zip_path.exists()
    except ImportError:
        print("gdown not installed. Install with: pip install gdown")
        return False
    except Exception as e:
        print(f"gdown download failed: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract zip at project root preserving folder structure."""
    if not zip_path.exists():
        return False
    print(f"Extracting {zip_path.name} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("Extraction complete.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download and extract actionplan_deps.zip")
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download; only extract if zip already exists",
    )
    parser.add_argument(
        "--keep_zip",
        action="store_true",
        help="Keep the zip file after extraction",
    )
    args = parser.parse_args()

    zip_path = PROJECT_ROOT / ZIP_NAME

    if not args.skip_download:
        if zip_path.exists():
            print(f"{ZIP_NAME} already exists. Use --skip_download to extract only.")
            resp = input("Re-download? [y/N]: ").strip().lower()
            if resp != "y":
                args.skip_download = True
            else:
                zip_path.unlink()

        if not args.skip_download:
            if not download_with_gdown(zip_path):
                print("\nManual download:")
                print(f"  1. Download from: {MANUAL_URL}")
                print(f"  2. Save as: {zip_path}")
                print(f"  3. Run: python prepare/download_dependencies.py --skip_download")
                sys.exit(1)

    if not zip_path.exists():
        print(f"Error: {ZIP_NAME} not found.")
        print(f"Download manually from: {MANUAL_URL}")
        print(f"Save to: {zip_path}")
        sys.exit(1)

    if not extract_zip(zip_path, PROJECT_ROOT):
        sys.exit(1)

    if not args.keep_zip:
        zip_path.unlink()
        print(f"Removed {ZIP_NAME}")


if __name__ == "__main__":
    main()
