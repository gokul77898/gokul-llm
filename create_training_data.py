#!/usr/bin/env python3
"""
Create training dataset from existing legal data
"""

import json
import os
import re
from pathlib import Path

def extract_text_from_txt(file_path):
    """Extract text from .txt files, skipping metadata headers."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Skip metadata header (first 6 lines)
        if lines and lines[0].startswith('ACT:'):
            lines = lines[7:]  # Skip header + blank line
        
        text = ''.join(lines).strip()
        
        # Clean up
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        if len(text) >= 200:  # Minimum length
            return text
    except Exception:
        pass
    return None

def extract_text_from_json(file_path):
    """Extract text from JSON chunk files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'text' in data:
            text = data['text'].strip()
            text = re.sub(r'\n+', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            
            if len(text) >= 200:
                return text
    except Exception:
        pass
    return None

def main():
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    output_file = data_dir / "train.jsonl"
    
    print("Creating training dataset...")
    
    # Process raw files and chunks
    source_dirs = [
        data_dir / "rag" / "raw",
        data_dir / "rag" / "chunks"
    ]
    
    texts = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
            
        print(f"Processing: {source_dir}")
        
        for file_path in source_dir.rglob('*'):
            if file_path.is_file():
                if file_path.suffix == '.txt':
                    text = extract_text_from_txt(file_path)
                    if text:
                        texts.append(text)
                elif file_path.suffix == '.json' and file_path.name != 'index.json':
                    text = extract_text_from_json(file_path)
                    if text:
                        texts.append(text)
    
    # Limit to first 1000 samples for Colab testing
    texts = texts[:1000]
    
    # Write JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
    
    print(f"✅ Created {output_file}")
    print(f"📊 Samples: {len(texts)}")
    print(f"📁 Size: {output_file.stat().st_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    main()
