"""
Filesystem Storage for Legal Documents

Phase R0: RAG Foundations

Stores documents as JSON files with versioning.
No overwrites without version bump.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from ..schemas.document import LegalDocument


class FilesystemStorage:
    """
    Filesystem-based document storage.
    
    Stores documents as JSON files:
    - data/rag/documents/{doc_id}.json
    
    Features:
    - Versioned writes only
    - No overwrites without version bump
    - Atomic writes
    """
    
    def __init__(self, base_path: str = "data/rag/documents"):
        """
        Initialize filesystem storage.
        
        Args:
            base_path: Base directory for document storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, doc: LegalDocument, allow_overwrite: bool = False) -> Path:
        """
        Save a document to filesystem.
        
        Args:
            doc: LegalDocument to save
            allow_overwrite: If True, allow overwriting existing version
            
        Returns:
            Path to saved document
            
        Raises:
            FileExistsError: If document exists and version not bumped
        """
        doc_path = self._get_doc_path(doc.doc_id)
        
        # Check for existing document
        if doc_path.exists() and not allow_overwrite:
            existing = self.load(doc.doc_id)
            if existing and existing.version >= doc.version:
                raise FileExistsError(
                    f"Document {doc.doc_id} already exists with version {existing.version}. "
                    f"Bump version to {existing.version + 1} to update."
                )
        
        # Atomic write: write to temp file then rename
        temp_path = doc_path.with_suffix('.json.tmp')
        
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(doc.model_dump(), f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_path.rename(doc_path)
            
        except Exception:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise
        
        return doc_path
    
    def load(self, doc_id: str) -> Optional[LegalDocument]:
        """
        Load a document by ID.
        
        Args:
            doc_id: Document ID to load
            
        Returns:
            LegalDocument or None if not found
        """
        doc_path = self._get_doc_path(doc_id)
        
        if not doc_path.exists():
            return None
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return LegalDocument(**data)
    
    def exists(self, doc_id: str) -> bool:
        """
        Check if a document exists.
        
        Args:
            doc_id: Document ID to check
            
        Returns:
            True if document exists
        """
        return self._get_doc_path(doc_id).exists()
    
    def delete(self, doc_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        doc_path = self._get_doc_path(doc_id)
        
        if not doc_path.exists():
            return False
        
        doc_path.unlink()
        return True
    
    def list_all(self) -> List[str]:
        """
        List all document IDs.
        
        Returns:
            List of document IDs
        """
        doc_ids = []
        for path in self.base_path.glob("*.json"):
            doc_ids.append(path.stem)
        return sorted(doc_ids)
    
    def count(self) -> int:
        """
        Count total documents.
        
        Returns:
            Number of documents
        """
        return len(list(self.base_path.glob("*.json")))
    
    def get_stats(self) -> dict:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage stats
        """
        doc_count = 0
        total_size = 0
        doc_types = {}
        
        for path in self.base_path.glob("*.json"):
            doc_count += 1
            total_size += path.stat().st_size
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    doc_type = data.get('doc_type', 'unknown')
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            except Exception:
                pass
        
        return {
            "document_count": doc_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "doc_types": doc_types,
            "storage_path": str(self.base_path.absolute()),
        }
    
    def _get_doc_path(self, doc_id: str) -> Path:
        """Get the file path for a document ID."""
        return self.base_path / f"{doc_id}.json"
