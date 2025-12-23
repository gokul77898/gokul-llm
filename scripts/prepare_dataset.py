#!/usr/bin/env python3
"""
PHASE B: Dataset Preparation
Prepare Indian Kanoon Supreme Court judgments dataset (~5GB)
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple
import re


def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    for entry in path.rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def create_raw_directory():
    """Create raw data directory."""
    raw_dir = Path("data/rag/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {raw_dir}")
    return raw_dir


def check_dataset_size(raw_dir: Path, target_gb: float = 5.0, tolerance: float = 0.5) -> Tuple[bool, float]:
    """Check if dataset size is within target range.
    
    Args:
        raw_dir: Path to raw data directory
        target_gb: Target size in GB
        tolerance: Tolerance in GB
    
    Returns:
        Tuple of (is_valid, actual_size_gb)
    """
    size_bytes = get_directory_size(raw_dir)
    size_gb = size_bytes / (1024 ** 3)
    
    min_gb = target_gb - tolerance
    max_gb = target_gb + tolerance
    
    is_valid = min_gb <= size_gb <= max_gb
    
    return is_valid, size_gb


def validate_file_format(file_path: Path) -> Tuple[bool, str]:
    """Validate file format and metadata.
    
    Args:
        file_path: Path to file
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check extension
    if file_path.suffix.lower() != '.txt':
        return False, f"Invalid extension: {file_path.suffix} (must be .txt)"
    
    # Check encoding and content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return False, "Not UTF-8 encoded"
    except Exception as e:
        return False, f"Read error: {e}"
    
    # Check if empty
    if not content.strip():
        return False, "Empty file"
    
    # Check for required metadata headers
    required_headers = ['ACT:', 'SECTION:', 'TYPE:', 'COURT:', 'YEAR:']
    lines = content.split('\n')
    
    header_found = {header: False for header in required_headers}
    
    for i, line in enumerate(lines[:10]):  # Check first 10 lines
        for header in required_headers:
            if line.strip().startswith(header):
                header_found[header] = True
    
    missing_headers = [h for h, found in header_found.items() if not found]
    if missing_headers:
        return False, f"Missing metadata headers: {', '.join(missing_headers)}"
    
    # Check for HTML tags
    if '<html' in content.lower() or '<body' in content.lower():
        return False, "Contains HTML tags"
    
    return True, ""


def normalize_files(raw_dir: Path) -> Tuple[int, int, List[str]]:
    """Normalize all files in raw directory.
    
    Args:
        raw_dir: Path to raw data directory
    
    Returns:
        Tuple of (valid_count, invalid_count, errors)
    """
    valid_count = 0
    invalid_count = 0
    errors = []
    
    print("\nValidating and normalizing files...")
    
    for file_path in raw_dir.rglob('*.txt'):
        is_valid, error_msg = validate_file_format(file_path)
        
        if is_valid:
            valid_count += 1
            if valid_count % 100 == 0:
                print(f"  Validated {valid_count} files...")
        else:
            invalid_count += 1
            errors.append(f"{file_path.name}: {error_msg}")
            
            # Remove invalid file
            file_path.unlink()
            print(f"  ✗ Removed invalid file: {file_path.name} ({error_msg})")
    
    return valid_count, invalid_count, errors


def main():
    print("=" * 70)
    print("PHASE B: DATASET PREPARATION")
    print("=" * 70)
    print()
    
    # Step 1: Create directory
    print("Step 1: Creating raw data directory...")
    raw_dir = create_raw_directory()
    print()
    
    # Step 2: Check if dataset exists
    print("Step 2: Checking for dataset...")
    file_count = len(list(raw_dir.glob('*.txt')))
    
    if file_count == 0:
        print("✗ No dataset found in data/rag/raw/")
        print()
        print("INSTRUCTIONS:")
        print("1. Download Indian Kanoon Supreme Court judgments (text format)")
        print("2. Place .txt files in: data/rag/raw/")
        print("3. Ensure files have required metadata headers:")
        print("   - ACT: <act name>")
        print("   - SECTION: <section number or NA>")
        print("   - TYPE: case_law")
        print("   - COURT: Supreme Court of India")
        print("   - YEAR: <YYYY>")
        print("4. Target size: ~5 GB (±0.5 GB)")
        print()
        print("Then re-run this script.")
        sys.exit(1)
    
    print(f"✓ Found {file_count} files")
    print()
    
    # Step 3: Check dataset size
    print("Step 3: Checking dataset size...")
    size_bytes = get_directory_size(raw_dir)
    size_gb = size_bytes / (1024 ** 3)
    
    print(f"  Current size: {format_size(size_bytes)} ({size_gb:.2f} GB)")
    
    is_valid_size, _ = check_dataset_size(raw_dir, target_gb=5.0, tolerance=0.5)
    
    if not is_valid_size:
        print(f"  ⚠ Size outside target range (4.5 GB - 5.5 GB)")
        print(f"  Current: {size_gb:.2f} GB")
        
        if size_gb > 5.5:
            print(f"  Action: Remove {size_gb - 5.0:.2f} GB of files")
        else:
            print(f"  Action: Add {5.0 - size_gb:.2f} GB of files")
        
        print()
        print("Adjust dataset size and re-run this script.")
        sys.exit(1)
    
    print(f"  ✓ Size within target range (4.5 GB - 5.5 GB)")
    print()
    
    # Step 4: Normalize files
    print("Step 4: Normalizing files...")
    valid_count, invalid_count, errors = normalize_files(raw_dir)
    
    print()
    print(f"✓ Validation complete:")
    print(f"  Valid files: {valid_count}")
    print(f"  Invalid files removed: {invalid_count}")
    
    if errors and len(errors) <= 10:
        print()
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
    elif errors:
        print()
        print(f"Total errors: {len(errors)} (showing first 10)")
        for error in errors[:10]:
            print(f"  - {error}")
    
    print()
    
    # Final verification
    if valid_count == 0:
        print("✗ ERROR: No valid files remaining")
        sys.exit(1)
    
    # Final size check
    final_size_bytes = get_directory_size(raw_dir)
    final_size_gb = final_size_bytes / (1024 ** 3)
    
    print("=" * 70)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 70)
    print(f"Valid files: {valid_count}")
    print(f"Total size: {format_size(final_size_bytes)} ({final_size_gb:.2f} GB)")
    print(f"Average file size: {format_size(final_size_bytes // valid_count)}")
    print()
    print("✓ Dataset ready for ingestion")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
