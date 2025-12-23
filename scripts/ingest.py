#!/usr/bin/env python3
"""
Phase-1 RAG: Document Ingestion Script

Ingests raw documents from data/rag/raw/ into data/rag/documents/
Each document is validated, normalized, and stored as canonical JSON.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --config configs/phase1_rag.yaml

This script is stateless and can be run independently.
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


# Inline document types to avoid import chain
class DocumentType(str, Enum):
    BARE_ACT = "bare_act"
    CASE_LAW = "case_law"
    AMENDMENT = "amendment"
    NOTIFICATION = "notification"


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def log(msg: str) -> None:
    """Simple logging with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def parse_raw_document(file_path: Path) -> dict:
    """
    Parse a raw document file with embedded metadata.
    
    Expected format:
    ACT: IPC
    SECTION: 420
    YEAR: 1860
    TYPE: bare_act
    
    <content>
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    metadata = {}
    content_start = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            content_start = i + 1
            break
        
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            metadata[key] = value
    
    text_content = '\n'.join(lines[content_start:]).strip()
    metadata['content'] = text_content
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Phase-1 RAG: Document Ingestion")
    parser.add_argument(
        "--config",
        default="configs/phase1_rag.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    # Load configuration
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        log(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Print header
    print("=" * 60)
    print("Phase-1 RAG: Document Ingestion")
    print("=" * 60)
    log(f"Config: {args.config}")
    log(f"Version: {config.get('version', 'unknown')}")
    
    # Resolve paths
    raw_dir = PROJECT_ROOT / config['paths']['raw_dir']
    documents_dir = PROJECT_ROOT / config['paths']['documents_dir']
    
    log(f"Raw directory: {raw_dir}")
    log(f"Documents directory: {documents_dir}")
    
    # Ensure directories exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)
    
    # Helper functions (inline to avoid import chain)
    def clean_text(text: str) -> str:
        """Clean raw text."""
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def generate_doc_id(text: str, source: str) -> str:
        """Generate deterministic document ID."""
        content = f"{source}:{text[:1000]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def save_document(doc_path: Path, doc_data: dict) -> None:
        """Save document as JSON."""
        temp_path = doc_path.with_suffix('.json.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, indent=2, ensure_ascii=False)
        temp_path.rename(doc_path)
    
    # Track results
    results = {
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "documents": [],
        "errors": [],
    }
    
    # Process raw files
    raw_files = list(raw_dir.glob("*.txt"))
    log(f"Found {len(raw_files)} raw files")
    
    if not raw_files:
        log("WARNING: No raw files found in data/rag/raw/")
        log("Place .txt files with metadata headers in data/rag/raw/")
        print()
        print("Expected format:")
        print("  ACT: Minimum Wages Act")
        print("  YEAR: 1948")
        print("  TYPE: bare_act")
        print("  ")
        print("  <document content>")
        sys.exit(0)
    
    print()
    for file_path in sorted(raw_files):
        results["processed"] += 1
        
        try:
            # Parse metadata from file
            meta = parse_raw_document(file_path)
            
            # Determine document type
            doc_type_str = meta.get('type', 'bare_act').lower()
            doc_type_map = {
                'bare_act': DocumentType.BARE_ACT,
                'case_law': DocumentType.CASE_LAW,
                'amendment': DocumentType.AMENDMENT,
                'notification': DocumentType.NOTIFICATION,
            }
            doc_type = doc_type_map.get(doc_type_str, DocumentType.BARE_ACT)
            
            # Extract year
            year = None
            if 'year' in meta:
                try:
                    year = int(meta['year'])
                except ValueError:
                    pass
            
            # Process document
            raw_text = meta.get('content', '')
            cleaned_text = clean_text(raw_text)
            title = file_path.stem.replace('_', ' ').title()
            source = file_path.name
            act = meta.get('act')
            court = meta.get('court')
            citation = meta.get('citation')
            
            # Generate deterministic doc_id
            doc_id = generate_doc_id(cleaned_text, source)
            
            # Check if already exists
            doc_path = documents_dir / f"{doc_id}.json"
            if doc_path.exists():
                results["skipped"] += 1
                log(f"○ Skipped (exists): {file_path.name}")
                continue
            
            # Create document data
            doc_data = {
                "doc_id": doc_id,
                "title": title,
                "doc_type": doc_type.value,
                "act": act,
                "court": court,
                "year": year,
                "citation": citation,
                "raw_text": cleaned_text,
                "source": source,
                "version": 1,
                "created_at": datetime.utcnow().isoformat(),
            }
            
            # Save document
            save_document(doc_path, doc_data)
            
            results["succeeded"] += 1
            results["documents"].append({
                "doc_id": doc_id,
                "title": title,
                "source": source,
                "act": act,
                "year": year,
            })
            log(f"✓ Ingested: {file_path.name} -> {doc_id[:12]}...")
            
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({
                "file": str(file_path.name),
                "error": str(e),
            })
            log(f"✗ Failed: {file_path.name} - {e}")
    
    # Print summary
    print()
    print("=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    log(f"Processed: {results['processed']}")
    log(f"Succeeded: {results['succeeded']}")
    log(f"Skipped:   {results['skipped']}")
    log(f"Failed:    {results['failed']}")
    
    # Storage stats
    doc_count = len(list(documents_dir.glob("*.json")))
    total_size = sum(f.stat().st_size for f in documents_dir.glob("*.json"))
    print()
    log(f"Total documents in store: {doc_count}")
    log(f"Storage size: {round(total_size / (1024 * 1024), 2)} MB")
    log(f"Storage path: {documents_dir}")
    
    if results['errors']:
        print()
        log("ERRORS:")
        for err in results['errors']:
            print(f"  - {err['file']}: {err['error']}")
    
    print()
    sys.exit(0 if results['failed'] == 0 else 1)


if __name__ == "__main__":
    main()
