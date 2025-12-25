"""
Script to download and extract PKLot dataset
"""

import urllib.request
import os
from pathlib import Path
import tarfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for download"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True,
                            miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path,
                                  reporthook=t.update_to)


def extract_tar_gz(tar_path, extract_path):
    """Extract tar.gz file with progress bar"""
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    print(f"Extraction complete!")


def main():
    print("=" * 70)
    print("PKLot Dataset Downloader")
    print("=" * 70)

    # PKLot dataset URL
    url = "https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz"

    # Paths
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    tar_path = data_dir / "PKLot.tar.gz"

    # Check if already downloaded
    if tar_path.exists():
        print(f"\nPKLot.tar.gz already exists at: {tar_path}")
        response = input("Re-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
        else:
            print(f"\nDownloading PKLot dataset from: {url}")
            print(f"This is a large file (~700 MB), it may take several minutes...")
            download_url(url, str(tar_path))
            print(f"\nDownload complete: {tar_path}")
    else:
        print(f"\nDownloading PKLot dataset from: {url}")
        print(f"This is a large file (~700 MB), it may take several minutes...")
        try:
            download_url(url, str(tar_path))
            print(f"\nDownload complete: {tar_path}")
        except Exception as e:
            print(f"\nError downloading: {e}")
            print("\nAlternative: Manual download")
            print(f"1. Visit: {url}")
            print(f"2. Download PKLot.tar.gz")
            print(f"3. Place it in: {data_dir}")
            return

    # Extract
    pklot_dir = data_dir / "PKLot"
    if pklot_dir.exists():
        print(f"\nPKLot directory already exists at: {pklot_dir}")
        response = input("Re-extract? (y/n): ")
        if response.lower() != 'y':
            print("Skipping extraction.")
            print(f"\n✓ Dataset ready at: {pklot_dir}")
            return

    print(f"\nExtracting to: {data_dir}")
    extract_tar_gz(tar_path, data_dir)

    print("\n" + "=" * 70)
    print("✓ PKLot dataset downloaded and extracted successfully!")
    print("=" * 70)
    print(f"\nDataset location: {pklot_dir}")
    print("\nNext step: Process the dataset")
    print("  python process_pklot.py")


if __name__ == "__main__":
    main()
