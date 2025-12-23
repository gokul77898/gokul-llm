#!/usr/bin/env python3
"""
Phase-1 RAG: Document Chunking Script

Chunks documents from data/rag/documents/ into data/rag/chunks/
Each chunk represents ONE legal unit (section, subsection, or paragraph).

Usage:
    python scripts/chunk.py
    python scripts/chunk.py --config configs/phase1_rag.yaml

This script is stateless and can be run independently.
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def log(msg: str) -> None:
    """Simple logging with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def main():
    parser = argparse.ArgumentParser(description="Phase-1 RAG: Document Chunking")
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
    print("Phase-1 RAG: Document Chunking")
    print("=" * 60)
    log(f"Config: {args.config}")
    log(f"Version: {config.get('version', 'unknown')}")
    
    # Resolve paths
    documents_dir = PROJECT_ROOT / config['paths']['documents_dir']
    chunks_dir = PROJECT_ROOT / config['paths']['chunks_dir']
    
    log(f"Documents directory: {documents_dir}")
    log(f"Chunks directory: {chunks_dir}")
    
    # Ensure directories exist
    documents_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # Get chunking config
    min_chunk_length = config.get('chunking', {}).get('min_chunk_length', 20)
    
    # Helper functions (inline to avoid import chain)
    def generate_chunk_id(doc_id: str, section: str, offset: int) -> str:
        """Generate deterministic chunk ID."""
        content = f"{doc_id}:{section}:{offset}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def generate_semantic_id(act: Optional[str], section: str, chunk_index: int, year: Optional[int] = None, court: Optional[str] = None) -> str:
        """Generate human-readable semantic ID.
        
        Format: <ACT>_<SECTION>_<INDEX>
        Examples:
            - IPC_420_0
            - MinimumWagesAct_2_1
            - SC_420_2015_0
        """
        # Sanitize act name
        if act:
            act_clean = re.sub(r'[^a-zA-Z0-9]', '', act)
        elif court:
            act_clean = re.sub(r'[^a-zA-Z0-9]', '', court)
        else:
            act_clean = "Unknown"
        
        # Add year if it's a case law
        if year and court:
            return f"{act_clean}_{section}_{year}_{chunk_index}"
        else:
            return f"{act_clean}_{section}_{chunk_index}"
    
    def parse_sections(text: str, doc_type: str) -> List[Dict]:
        """Parse document into sections based on legal structure."""
        sections = []
        
        if doc_type in ('bare_act', 'amendment'):
            # Parse by Section pattern
            pattern = r'(Section\s+\d+[A-Z]?\.?[^\n]*(?:\n(?!Section\s+\d)[^\n]*)*)'  
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            if matches:
                for i, match in enumerate(matches):
                    section_text = match.group(1).strip()
                    # Extract section number
                    sec_match = re.match(r'Section\s+(\d+[A-Z]?)', section_text, re.IGNORECASE)
                    section_num = sec_match.group(1) if sec_match else str(i + 1)
                    
                    sections.append({
                        'section': section_num,
                        'text': section_text,
                        'start_offset': match.start(),
                        'end_offset': match.end(),
                    })
            else:
                # Fallback: split by paragraphs
                sections = parse_paragraphs(text)
        else:
            # For case_law, notification: split by paragraphs
            sections = parse_paragraphs(text)
        
        return sections
    
    def parse_paragraphs(text: str) -> List[Dict]:
        """Parse text into paragraphs."""
        paragraphs = text.split('\n\n')
        sections = []
        offset = 0
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) >= min_chunk_length:
                sections.append({
                    'section': str(i + 1),
                    'text': para,
                    'start_offset': offset,
                    'end_offset': offset + len(para),
                })
            offset += len(para) + 2
        
        return sections
    
    def load_document(doc_path: Path) -> Optional[Dict]:
        """Load document from JSON file."""
        if not doc_path.exists():
            return None
        with open(doc_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_chunk(chunk_path: Path, chunk_data: Dict) -> bool:
        """Save chunk as JSON. Returns False if already exists."""
        if chunk_path.exists():
            return False
        temp_path = chunk_path.with_suffix('.json.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        temp_path.rename(chunk_path)
        return True
    
    # Track results
    results = {
        "documents_processed": 0,
        "documents_succeeded": 0,
        "documents_failed": 0,
        "chunks_created": 0,
        "chunks_skipped": 0,
        "errors": [],
    }
    
    # Get all documents
    doc_files = list(documents_dir.glob("*.json"))
    log(f"Found {len(doc_files)} documents")
    
    if not doc_files:
        log("WARNING: No documents found in data/rag/documents/")
        log("Run 'python scripts/ingest.py' first")
        sys.exit(0)
    
    print()
    for doc_path in sorted(doc_files):
        doc_id = doc_path.stem
        results["documents_processed"] += 1
        
        try:
            # Load document
            doc = load_document(doc_path)
            if not doc:
                results["documents_failed"] += 1
                results["errors"].append({
                    "doc_id": doc_id,
                    "error": "Document not found",
                })
                continue
            
            # Parse sections
            doc_type = doc.get('doc_type', 'bare_act')
            raw_text = doc.get('raw_text', '')
            sections = parse_sections(raw_text, doc_type)
            
            # Create and save chunks
            chunks_saved = 0
            chunks_skipped = 0
            
            for i, section in enumerate(sections):
                chunk_id = generate_chunk_id(doc_id, section['section'], section['start_offset'])
                chunk_path = chunks_dir / f"{chunk_id}.json"
                
                # Generate semantic ID
                semantic_id = generate_semantic_id(
                    act=doc.get('act'),
                    section=section['section'],
                    chunk_index=i,
                    year=doc.get('year'),
                    court=doc.get('court')
                )
                
                chunk_data = {
                    "chunk_id": chunk_id,
                    "semantic_id": semantic_id,
                    "doc_id": doc_id,
                    "act": doc.get('act'),
                    "section": section['section'],
                    "subsection": None,
                    "doc_type": doc_type,
                    "text": section['text'],
                    "citation": doc.get('citation'),
                    "court": doc.get('court'),
                    "year": doc.get('year'),
                    "start_offset": section['start_offset'],
                    "end_offset": section['end_offset'],
                    "chunk_index": i,
                    "version": 1,
                    "created_at": datetime.utcnow().isoformat(),
                }
                
                if save_chunk(chunk_path, chunk_data):
                    chunks_saved += 1
                else:
                    chunks_skipped += 1
            
            results["documents_succeeded"] += 1
            results["chunks_created"] += chunks_saved
            results["chunks_skipped"] += chunks_skipped
            
            title = doc.get('title', doc_id)[:40]
            log(f"✓ Chunked: {title} -> {len(sections)} chunks ({chunks_saved} new, {chunks_skipped} skipped)")
            
        except Exception as e:
            results["documents_failed"] += 1
            results["errors"].append({
                "doc_id": doc_id[:12],
                "error": str(e),
            })
            log(f"✗ Failed: {doc_id[:12]}... - {e}")
    
    # Print summary
    print()
    print("=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)
    log(f"Documents processed: {results['documents_processed']}")
    log(f"Documents succeeded: {results['documents_succeeded']}")
    log(f"Documents failed:    {results['documents_failed']}")
    print()
    log(f"Chunks created: {results['chunks_created']}")
    log(f"Chunks skipped: {results['chunks_skipped']}")
    
    # Storage stats
    chunk_files = [f for f in chunks_dir.glob("*.json") if f.name != "index.json"]
    chunk_count = len(chunk_files)
    total_size = sum(f.stat().st_size for f in chunk_files)
    
    # Count by act
    acts = {}
    doc_ids_seen = set()
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r') as f:
                data = json.load(f)
                act = data.get('act')
                if act:
                    acts[act] = acts.get(act, 0) + 1
                doc_ids_seen.add(data.get('doc_id', 'unknown'))
        except:
            pass
    
    print()
    log(f"Total chunks in store: {chunk_count}")
    log(f"Total documents: {len(doc_ids_seen)}")
    log(f"Storage size: {round(total_size / (1024 * 1024), 2)} MB")
    log(f"Storage path: {chunks_dir}")
    
    if acts:
        print()
        log("Chunks by Act:")
        for act, count in sorted(acts.items()):
            print(f"  - {act}: {count} chunks")
    
    if results['errors']:
        print()
        log("ERRORS:")
        for err in results['errors']:
            print(f"  - {err['doc_id']}: {err['error']}")
    
    print()
    sys.exit(0 if results['documents_failed'] == 0 else 1)


if __name__ == "__main__":
    main()
