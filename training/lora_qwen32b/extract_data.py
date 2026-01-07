#!/usr/bin/env python3
"""
Production Data Extraction for LoRA Training
Extracts clean text from 300GB legal corpus into train.jsonl

Usage:
    python extract_data.py --input_dir /path/to/300gb/data --output data/train.jsonl
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Generator, Optional
import hashlib


def log(msg: str) -> None:
    """Timestamped logging."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def clean_text(text: str, min_length: int = 200) -> Optional[str]:
    """Clean and validate text for training."""
    if not text or not isinstance(text, str):
        return None
    
    # Remove excessive whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Check minimum length
    if len(text) < min_length:
        return None
    
    # Check for binary/garbage content
    non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text)
    if non_ascii_ratio > 0.3:  # More than 30% non-ASCII is likely garbage
        return None
    
    return text


def extract_from_txt(file_path: Path) -> Optional[str]:
    """Extract text from .txt files."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Skip metadata headers if present
        lines = content.split('\n')
        if lines and lines[0].startswith(('ACT:', 'SECTION:', 'DOCUMENT:')):
            # Find first empty line and skip header
            for i, line in enumerate(lines):
                if not line.strip():
                    content = '\n'.join(lines[i+1:])
                    break
        
        return clean_text(content)
    except Exception as e:
        return None


def extract_from_json(file_path: Path) -> Optional[str]:
    """Extract text from JSON files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            text = data.get('text') or data.get('content') or data.get('body', '')
        elif isinstance(data, list) and len(data) > 0:
            text = data[0].get('text', '') if isinstance(data[0], dict) else str(data[0])
        else:
            return None
        
        return clean_text(text)
    except Exception:
        return None


def extract_from_pdf_text(file_path: Path) -> Optional[str]:
    """Extract text from PDF text exports."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return clean_text(content)
    except Exception:
        return None


def process_directory(input_dir: Path, extensions: set) -> Generator[str, None, None]:
    """Process all files in directory and yield clean texts."""
    file_count = 0
    success_count = 0
    
    for file_path in input_dir.rglob('*'):
        if not file_path.is_file():
            continue
        
        suffix = file_path.suffix.lower()
        if suffix not in extensions:
            continue
        
        file_count += 1
        
        # Extract based on file type
        text = None
        if suffix == '.txt':
            text = extract_from_txt(file_path)
        elif suffix == '.json':
            text = extract_from_json(file_path)
        elif suffix in ('.md', '.rst'):
            text = extract_from_txt(file_path)
        
        if text:
            success_count += 1
            yield text
        
        # Progress logging
        if file_count % 10000 == 0:
            log(f"  Processed {file_count} files, extracted {success_count} texts")
    
    log(f"  Final: {file_count} files processed, {success_count} texts extracted")


def create_train_val_split(
    texts: list,
    output_dir: Path,
    val_ratio: float = 0.1,
) -> tuple:
    """Create train/validation split and write to JSONL files."""
    import random
    random.seed(42)
    random.shuffle(texts)
    
    val_size = int(len(texts) * val_ratio)
    val_texts = texts[:val_size]
    train_texts = texts[val_size:]
    
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    
    # Write train file
    with open(train_path, 'w', encoding='utf-8') as f:
        for text in train_texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
    
    # Write validation file
    with open(val_path, 'w', encoding='utf-8') as f:
        for text in val_texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
    
    return train_path, val_path, len(train_texts), len(val_texts)


def main():
    parser = argparse.ArgumentParser(description="Extract training data from legal corpus")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing legal documents",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for train.jsonl and val.jsonl",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=200,
        help="Minimum text length (default: 200 chars)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to extract (default: all)",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        log(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log("=" * 60)
    log("PRODUCTION DATA EXTRACTION")
    log("=" * 60)
    log(f"Input directory: {input_dir}")
    log(f"Output directory: {output_dir}")
    log(f"Validation ratio: {args.val_ratio}")
    
    # Supported extensions
    extensions = {'.txt', '.json', '.md', '.rst'}
    
    # Extract texts
    log("Extracting texts...")
    texts = []
    for text in process_directory(input_dir, extensions):
        texts.append(text)
        if args.max_samples and len(texts) >= args.max_samples:
            log(f"  Reached max samples: {args.max_samples}")
            break
    
    if len(texts) == 0:
        log("❌ No valid texts extracted!")
        sys.exit(1)
    
    log(f"✓ Extracted {len(texts)} texts")
    
    # Calculate stats
    total_chars = sum(len(t) for t in texts)
    avg_chars = total_chars / len(texts)
    estimated_tokens = total_chars / 4  # Rough estimate
    
    log(f"  Total characters: {total_chars:,}")
    log(f"  Average chars/sample: {avg_chars:.0f}")
    log(f"  Estimated tokens: {estimated_tokens:,.0f}")
    
    # Create train/val split
    log("Creating train/validation split...")
    train_path, val_path, train_count, val_count = create_train_val_split(
        texts, output_dir, args.val_ratio
    )
    
    log(f"✓ Train samples: {train_count}")
    log(f"✓ Validation samples: {val_count}")
    log(f"✓ Train file: {train_path} ({train_path.stat().st_size / (1024*1024):.1f} MB)")
    log(f"✓ Validation file: {val_path} ({val_path.stat().st_size / (1024*1024):.1f} MB)")
    
    log("=" * 60)
    log("DATA EXTRACTION COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    main()
