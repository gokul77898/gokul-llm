"""
AWS Indian Supreme Court Judgments Dataset Downloader

Downloads parquet files from the AWS Open Data Registry.
Dataset: https://registry.opendata.aws/indian-supreme-court-judgments

Usage:
    python3 src/ingest/download.py --years 2018 2019 2020
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from typing import List

# AWS S3 base URL for Indian Supreme Court judgments
BASE_URL = "https://indian-supreme-court.s3.amazonaws.com/parquet"

def download_parquet(year: int, output_dir: str = "data/parquet") -> bool:
    """
    Download a single parquet file for a given year.
    
    Args:
        year: Year of judgments to download
        output_dir: Directory to save the parquet file
        
    Returns:
        bool: True if download successful, False otherwise
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"{year}.parquet"
    url = f"{BASE_URL}/{filename}"
    output_file = output_path / filename
    
    # Skip if file already exists
    if output_file.exists():
        print(f"✅ {filename} already exists, skipping download")
        return True
    
    print(f"📥 Downloading {filename} from {url}...")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_file, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                chunk_size = 8192
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Show progress
                        percent = (downloaded / total_size) * 100
                        print(f"\r   Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
                print()  # New line after progress
        
        print(f"✅ Downloaded {filename} ({os.path.getsize(output_file)} bytes)")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download {filename}: {e}")
        if output_file.exists():
            output_file.unlink()  # Clean up partial download
        return False
    except Exception as e:
        print(f"❌ Unexpected error downloading {filename}: {e}")
        return False


def download_multiple_years(years: List[int], output_dir: str = "data/parquet") -> int:
    """
    Download parquet files for multiple years.
    
    Args:
        years: List of years to download
        output_dir: Directory to save parquet files
        
    Returns:
        int: Number of successfully downloaded files
    """
    print(f"Starting download of {len(years)} parquet files...")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    success_count = 0
    failed_years = []
    
    for year in years:
        if download_parquet(year, output_dir):
            success_count += 1
        else:
            failed_years.append(year)
    
    print("-" * 60)
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Successful: {success_count}/{len(years)}")
    if failed_years:
        print(f"   ❌ Failed: {failed_years}")
    
    return success_count


def main():
    """Main entry point for downloading parquet files."""
    parser = argparse.ArgumentParser(
        description="Download Indian Supreme Court judgment parquet files from AWS"
    )
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        default=[2018, 2019, 2020],
        help='Years to download (default: 2018 2019 2020)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/parquet',
        help='Output directory (default: data/parquet)'
    )
    
    args = parser.parse_args()
    
    # Download files
    success_count = download_multiple_years(args.years, args.output)
    
    if success_count == 0:
        print("\n⚠️  No files downloaded successfully")
        sys.exit(1)
    
    print(f"\n✅ Download complete! {success_count} file(s) ready for ingestion")
    sys.exit(0)


if __name__ == "__main__":
    main()
