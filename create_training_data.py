#!/usr/bin/env python3
"""
Production Data Extraction for LoRA Training

Creates train.jsonl and val.jsonl from legal corpus.
NO SAMPLE LIMITS - processes entire corpus.
Deterministic split with fixed seed for reproducibility.

Usage:
    python create_training_data.py
    python create_training_data.py --input_dir /path/to/data --val_ratio 0.1
"""

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# =============================================================================
# FIXED SEED FOR REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = 42


def log(msg: str) -> None:
    """Timestamped logging."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def extract_text_from_txt(file_path: Path, min_length: int = 200) -> Optional[str]:
    """Extract text from .txt files, skipping metadata headers."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Skip metadata header if present
        if lines and lines[0].startswith(('ACT:', 'SECTION:', 'DOCUMENT:')):
            for i, line in enumerate(lines):
                if not line.strip():
                    lines = lines[i+1:]
                    break
        
        text = ''.join(lines).strip()
        
        # Clean up whitespace
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Validate: minimum length and not binary garbage
        if len(text) < min_length:
            return None
        
        # Check for binary content (high non-ASCII ratio = garbage)
        non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text) if text else 0
        if non_ascii_ratio > 0.3:
            return None
        
        return text
    except Exception:
        return None


def extract_text_from_json(file_path: Path, min_length: int = 200) -> Optional[str]:
    """Extract text from JSON chunk files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            text = data.get('text') or data.get('content') or data.get('body', '')
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            text = data[0].get('text', '')
        else:
            return None
        
        if not text:
            return None
        
        text = text.strip()
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        if len(text) < min_length:
            return None
        
        # Check for binary content
        non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text) if text else 0
        if non_ascii_ratio > 0.3:
            return None
        
        return text
    except Exception:
        return None


def process_directory(source_dir: Path, min_length: int = 200) -> List[str]:
    """
    Process all files in directory and return clean texts.
    NO SAMPLE LIMITS - processes everything.
    """
    texts = []
    file_count = 0
    
    if not source_dir.exists():
        log(f"  Directory not found: {source_dir}")
        return texts
    
    log(f"Processing: {source_dir}")
    
    for file_path in source_dir.rglob('*'):
        if not file_path.is_file():
            continue
        
        # Skip index files and hidden files
        if file_path.name.startswith('.') or file_path.name == 'index.json':
            continue
        
        suffix = file_path.suffix.lower()
        text = None
        
        if suffix == '.txt':
            text = extract_text_from_txt(file_path, min_length)
        elif suffix == '.json':
            text = extract_text_from_json(file_path, min_length)
        
        if text:
            texts.append(text)
        
        file_count += 1
        if file_count % 10000 == 0:
            log(f"  Processed {file_count} files, extracted {len(texts)} texts")
    
    log(f"  Finished: {file_count} files processed, {len(texts)} texts extracted")
    return texts


def create_train_val_split(
    texts: List[str],
    output_dir: Path,
    val_ratio: float = 0.1,
) -> tuple:
    """
    Create deterministic train/validation split.
    Uses fixed seed for reproducibility.
    Val data is NEVER mixed into training.
    """
    # Set seed for deterministic split
    random.seed(RANDOM_SEED)
    
    # Shuffle deterministically
    shuffled = texts.copy()
    random.shuffle(shuffled)
    
    # Split
    val_size = int(len(shuffled) * val_ratio)
    val_texts = shuffled[:val_size]
    train_texts = shuffled[val_size:]
    
    # Write files
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    
    log(f"Writing train.jsonl ({len(train_texts)} samples)...")
    with open(train_path, 'w', encoding='utf-8') as f:
        for text in train_texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
    
    log(f"Writing val.jsonl ({len(val_texts)} samples)...")
    with open(val_path, 'w', encoding='utf-8') as f:
        for text in val_texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
    
    return train_path, val_path, len(train_texts), len(val_texts)


def main():
    parser = argparse.ArgumentParser(description="Create training data from legal corpus")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Input directory (default: data/rag/)",
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
        help="Validation split ratio (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=200,
        help="Minimum text length in characters (default: 200)",
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent
    output_dir = project_root / args.output_dir
    
    # Determine input directories
    if args.input_dir:
        source_dirs = [Path(args.input_dir)]
    else:
        source_dirs = [
            project_root / "data" / "rag" / "raw",
            project_root / "data" / "rag" / "chunks",
        ]
    
    log("=" * 60)
    log("PRODUCTION DATA EXTRACTION")
    log("=" * 60)
    log(f"Random seed: {RANDOM_SEED} (deterministic)")
    log(f"Validation ratio: {args.val_ratio}")
    log(f"Min text length: {args.min_length} chars")
    log(f"Output directory: {output_dir}")
    
    # Process all source directories - NO LIMITS
    all_texts = []
    for source_dir in source_dirs:
        texts = process_directory(source_dir, args.min_length)
        all_texts.extend(texts)
    
    if len(all_texts) == 0:
        log("❌ ERROR: No valid texts extracted!")
        log("Check that input directories contain .txt or .json files with 'text' field.")
        sys.exit(1)
    
    log(f"✓ Total texts extracted: {len(all_texts)}")
    
    # Calculate stats
    total_chars = sum(len(t) for t in all_texts)
    avg_chars = total_chars / len(all_texts)
    estimated_tokens = total_chars / 4  # Rough estimate
    
    log(f"  Total characters: {total_chars:,}")
    log(f"  Average chars/sample: {avg_chars:.0f}")
    log(f"  Estimated tokens: {estimated_tokens:,.0f}")
    
    # Create train/val split
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path, train_count, val_count = create_train_val_split(
        all_texts, output_dir, args.val_ratio
    )
    
    log("=" * 60)
    log("DATA EXTRACTION COMPLETE")
    log("=" * 60)
    log(f"✓ Train: {train_path} ({train_count} samples, {train_path.stat().st_size / (1024*1024):.1f} MB)")
    log(f"✓ Val:   {val_path} ({val_count} samples, {val_path.stat().st_size / (1024*1024):.1f} MB)")
    log(f"✓ Validation data is SEPARATE from training data")


if __name__ == "__main__":
    main()
