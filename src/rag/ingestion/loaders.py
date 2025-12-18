"""
Document Loaders

Phase R0: RAG Foundations

Load legal documents from various sources.
Applies cleaning, normalization, and validation.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..schemas.document import DocumentType, LegalDocument
from ..storage.filesystem import FilesystemStorage
from ..utils.text_cleaning import clean_text
from .normalizer import normalize_text, extract_act_name, extract_year
from .validator import validate_document, ValidationError


def load_document(
    source_path: str,
    doc_type: DocumentType,
    title: Optional[str] = None,
    act: Optional[str] = None,
    court: Optional[str] = None,
    year: Optional[int] = None,
    citation: Optional[str] = None,
    storage: Optional[FilesystemStorage] = None,
) -> LegalDocument:
    """
    Load a document from a file path.
    
    Flow:
    1. Read raw text from file
    2. Clean text (remove headers/footers, normalize whitespace)
    3. Normalize legal references
    4. Validate document
    5. Generate deterministic doc_id
    6. Optionally persist to filesystem
    
    Args:
        source_path: Path to source file (txt, pdf text extract)
        doc_type: Type of legal document
        title: Document title (extracted from filename if not provided)
        act: Act name (for bare_act, amendment)
        court: Court name (for case_law)
        year: Document year
        citation: Legal citation
        storage: Optional storage to persist document
        
    Returns:
        Validated LegalDocument
        
    Raises:
        FileNotFoundError: If source file not found
        ValidationError: If document validation fails
    """
    path = Path(source_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    # Read raw text
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        raw_text = f.read()
    
    # Use filename as title if not provided
    if not title:
        title = path.stem.replace('_', ' ').replace('-', ' ').title()
    
    return load_from_text(
        raw_text=raw_text,
        source=str(path.name),
        doc_type=doc_type,
        title=title,
        act=act,
        court=court,
        year=year,
        citation=citation,
        storage=storage,
    )


def load_from_text(
    raw_text: str,
    source: str,
    doc_type: DocumentType,
    title: str,
    act: Optional[str] = None,
    court: Optional[str] = None,
    year: Optional[int] = None,
    citation: Optional[str] = None,
    storage: Optional[FilesystemStorage] = None,
) -> LegalDocument:
    """
    Load a document from raw text.
    
    Args:
        raw_text: Raw document text
        source: Source identifier (filename or URL)
        doc_type: Type of legal document
        title: Document title
        act: Act name (for bare_act, amendment)
        court: Court name (for case_law)
        year: Document year
        citation: Legal citation
        storage: Optional storage to persist document
        
    Returns:
        Validated LegalDocument
        
    Raises:
        ValidationError: If document validation fails
    """
    # Step 1: Clean text
    cleaned_text = clean_text(raw_text)
    
    # Step 2: Normalize legal references
    normalized_text = normalize_text(cleaned_text)
    
    # Step 3: Auto-extract metadata if not provided
    if not act and doc_type in (DocumentType.BARE_ACT, DocumentType.AMENDMENT):
        act = extract_act_name(normalized_text)
    
    if not year:
        year = extract_year(normalized_text)
    
    # Step 4: Generate deterministic doc_id
    doc_id = LegalDocument.generate_doc_id(normalized_text, source)
    
    # Step 5: Create document
    doc = LegalDocument(
        doc_id=doc_id,
        title=title,
        doc_type=doc_type,
        act=act,
        court=court,
        year=year,
        citation=citation,
        raw_text=normalized_text,
        source=source,
        version=1,
        created_at=datetime.utcnow().isoformat(),
    )
    
    # Step 6: Validate
    validate_document(doc)
    
    # Step 7: Persist if storage provided
    if storage:
        storage.save(doc)
    
    return doc


def ingest_directory(
    directory: str,
    doc_type: DocumentType,
    storage: Optional[FilesystemStorage] = None,
    extensions: tuple = ('.txt',),
) -> dict:
    """
    Ingest all documents from a directory.
    
    Args:
        directory: Path to directory
        doc_type: Document type for all files
        storage: Optional storage to persist documents
        extensions: File extensions to process
        
    Returns:
        Dictionary with ingestion results
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    results = {
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "documents": [],
        "errors": [],
    }
    
    for ext in extensions:
        for file_path in dir_path.glob(f"*{ext}"):
            results["processed"] += 1
            
            try:
                doc = load_document(
                    source_path=str(file_path),
                    doc_type=doc_type,
                    storage=storage,
                )
                results["succeeded"] += 1
                results["documents"].append({
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "source": doc.source,
                })
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "file": str(file_path.name),
                    "error": str(e),
                })
    
    return results


def parse_raw_document(file_path: str) -> dict:
    """
    Parse a raw document file with embedded metadata.
    
    Expected format:
    ACT: IPC
    SECTION: 420
    YEAR: 1860
    TYPE: bare_act
    
    <content>
    
    Args:
        file_path: Path to raw document file
        
    Returns:
        Dictionary with metadata and content
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
    
    # Extract content after metadata
    text_content = '\n'.join(lines[content_start:]).strip()
    metadata['content'] = text_content
    
    return metadata


def ingest_raw_directory(
    raw_dir: str = "data/raw",
    output_dir: str = "data/rag/documents",
) -> dict:
    """
    Ingest all raw documents from data/raw/ directory.
    
    Parses embedded metadata and creates canonical documents.
    
    Args:
        raw_dir: Path to raw documents directory
        output_dir: Path to output documents directory
        
    Returns:
        Ingestion results
    """
    raw_path = Path(raw_dir)
    storage = FilesystemStorage(base_path=output_dir)
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    
    results = {
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "documents": [],
        "errors": [],
    }
    
    for file_path in raw_path.glob("*.txt"):
        results["processed"] += 1
        
        try:
            # Parse metadata from file
            meta = parse_raw_document(str(file_path))
            
            # Determine document type
            doc_type_str = meta.get('type', 'bare_act').lower()
            if doc_type_str == 'bare_act':
                doc_type = DocumentType.BARE_ACT
            elif doc_type_str == 'case_law':
                doc_type = DocumentType.CASE_LAW
            elif doc_type_str == 'amendment':
                doc_type = DocumentType.AMENDMENT
            elif doc_type_str == 'notification':
                doc_type = DocumentType.NOTIFICATION
            else:
                doc_type = DocumentType.BARE_ACT
            
            # Extract year
            year = None
            if 'year' in meta:
                try:
                    year = int(meta['year'])
                except ValueError:
                    pass
            
            # Load document
            doc = load_from_text(
                raw_text=meta.get('content', ''),
                source=file_path.name,
                doc_type=doc_type,
                title=file_path.stem.replace('_', ' ').title(),
                act=meta.get('act'),
                court=meta.get('court'),
                year=year,
                citation=meta.get('citation'),
                storage=storage,
            )
            
            results["succeeded"] += 1
            results["documents"].append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "source": doc.source,
                "act": doc.act,
                "court": doc.court,
                "year": doc.year,
            })
            print(f"✓ Ingested: {file_path.name} -> {doc.doc_id[:8]}...")
            
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({
                "file": str(file_path.name),
                "error": str(e),
            })
            print(f"✗ Failed: {file_path.name} - {e}")
    
    return results


if __name__ == "__main__":
    """CLI entry point for ingestion."""
    import sys
    
    print("=" * 50)
    print("RAG Ingestion Pipeline")
    print("=" * 50)
    
    # Ingest from data/raw/
    try:
        results = ingest_raw_directory()
        
        print()
        print(f"Processed: {results['processed']}")
        print(f"Succeeded: {results['succeeded']}")
        print(f"Failed: {results['failed']}")
        
        if results['errors']:
            print("\nErrors:")
            for err in results['errors']:
                print(f"  - {err['file']}: {err['error']}")
        
        print()
        print("Documents created:")
        for doc in results['documents']:
            print(f"  - {doc['title']} ({doc['doc_id'][:8]}...)")
        
        sys.exit(0 if results['failed'] == 0 else 1)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
