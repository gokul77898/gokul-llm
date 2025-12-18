"""
Chunk Storage for Legal Documents

Phase R1: Legal Chunking & Indexing

Stores chunks as JSON files with versioning and index management.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from ..schemas.chunk import LegalChunk, ChunkIndexEntry


class ChunkStorage:
    """
    Filesystem-based chunk storage.
    
    Stores chunks as JSON files:
    - data/rag/chunks/{chunk_id}.json
    
    Maintains an index file:
    - data/rag/chunks/index.json
    
    Features:
    - Versioned writes only
    - No overwrites without version bump
    - Atomic writes
    - Index manifest for chunk lookup
    """
    
    def __init__(self, base_path: str = "data/rag/chunks"):
        """
        Initialize chunk storage.
        
        Args:
            base_path: Base directory for chunk storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_path / "index.json"
        
        # Load or initialize index
        self._index: Dict[str, dict] = self._load_index()
    
    def save(self, chunk: LegalChunk, allow_overwrite: bool = False) -> Path:
        """
        Save a chunk to filesystem.
        
        Args:
            chunk: LegalChunk to save
            allow_overwrite: If True, allow overwriting existing version
            
        Returns:
            Path to saved chunk
            
        Raises:
            FileExistsError: If chunk exists and version not bumped
        """
        chunk_path = self._get_chunk_path(chunk.chunk_id)
        
        # Check for existing chunk
        if chunk_path.exists() and not allow_overwrite:
            existing = self.load(chunk.chunk_id)
            if existing and existing.version >= chunk.version:
                raise FileExistsError(
                    f"Chunk {chunk.chunk_id} already exists with version {existing.version}. "
                    f"Bump version to {existing.version + 1} to update."
                )
        
        # Atomic write: write to temp file then rename
        temp_path = chunk_path.with_suffix('.json.tmp')
        
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(chunk.model_dump(), f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_path.rename(chunk_path)
            
            # Update index
            self._update_index(chunk)
            
        except Exception:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise
        
        return chunk_path
    
    def save_many(self, chunks: List[LegalChunk], allow_overwrite: bool = False) -> int:
        """
        Save multiple chunks.
        
        Args:
            chunks: List of chunks to save
            allow_overwrite: If True, allow overwriting
            
        Returns:
            Number of chunks saved
        """
        saved = 0
        for chunk in chunks:
            try:
                self.save(chunk, allow_overwrite=allow_overwrite)
                saved += 1
            except FileExistsError:
                # Skip existing chunks
                pass
        
        return saved
    
    def load(self, chunk_id: str) -> Optional[LegalChunk]:
        """
        Load a chunk by ID.
        
        Args:
            chunk_id: Chunk ID to load
            
        Returns:
            LegalChunk or None if not found
        """
        chunk_path = self._get_chunk_path(chunk_id)
        
        if not chunk_path.exists():
            return None
        
        with open(chunk_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return LegalChunk(**data)
    
    def exists(self, chunk_id: str) -> bool:
        """Check if a chunk exists."""
        return self._get_chunk_path(chunk_id).exists()
    
    def delete(self, chunk_id: str) -> bool:
        """
        Delete a chunk.
        
        Args:
            chunk_id: Chunk ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        chunk_path = self._get_chunk_path(chunk_id)
        
        if not chunk_path.exists():
            return False
        
        chunk_path.unlink()
        
        # Remove from index
        if chunk_id in self._index:
            del self._index[chunk_id]
            self._save_index()
        
        return True
    
    def list_all(self) -> List[str]:
        """List all chunk IDs."""
        chunk_ids = []
        for path in self.base_path.glob("*.json"):
            if path.name != "index.json":
                chunk_ids.append(path.stem)
        return sorted(chunk_ids)
    
    def list_by_doc(self, doc_id: str) -> List[str]:
        """
        List chunk IDs for a specific document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of chunk IDs
        """
        return [
            chunk_id for chunk_id, entry in self._index.items()
            if entry.get("doc_id") == doc_id
        ]
    
    def count(self) -> int:
        """Count total chunks."""
        return len([p for p in self.base_path.glob("*.json") if p.name != "index.json"])
    
    def get_index(self) -> Dict[str, dict]:
        """
        Get the chunk index.
        
        Returns:
            Dictionary mapping chunk_id to index entry
        """
        return self._index.copy()
    
    def get_stats(self) -> dict:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage stats
        """
        chunk_count = 0
        total_size = 0
        doc_ids = set()
        acts = {}
        
        for path in self.base_path.glob("*.json"):
            if path.name == "index.json":
                continue
            
            chunk_count += 1
            total_size += path.stat().st_size
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    doc_ids.add(data.get('doc_id', 'unknown'))
                    act = data.get('act')
                    if act:
                        acts[act] = acts.get(act, 0) + 1
            except Exception:
                pass
        
        return {
            "chunk_count": chunk_count,
            "document_count": len(doc_ids),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "acts": acts,
            "storage_path": str(self.base_path.absolute()),
        }
    
    def rebuild_index(self) -> int:
        """
        Rebuild the index from stored chunks.
        
        Returns:
            Number of chunks indexed
        """
        self._index = {}
        
        for path in self.base_path.glob("*.json"):
            if path.name == "index.json":
                continue
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                chunk = LegalChunk(**data)
                entry = ChunkIndexEntry.from_chunk(chunk)
                self._index[chunk.chunk_id] = entry.model_dump()
            except Exception:
                pass
        
        self._save_index()
        return len(self._index)
    
    def _get_chunk_path(self, chunk_id: str) -> Path:
        """Get the file path for a chunk ID."""
        return self.base_path / f"{chunk_id}.json"
    
    def _load_index(self) -> Dict[str, dict]:
        """Load the index from disk."""
        if not self.index_path.exists():
            return {}
        
        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("chunks", {})
        except Exception:
            return {}
    
    def _save_index(self) -> None:
        """Save the index to disk."""
        index_data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "chunk_count": len(self._index),
            "chunks": self._index,
        }
        
        # Atomic write
        temp_path = self.index_path.with_suffix('.json.tmp')
        
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            
            temp_path.rename(self.index_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def _update_index(self, chunk: LegalChunk) -> None:
        """Update the index with a chunk."""
        entry = ChunkIndexEntry.from_chunk(chunk)
        self._index[chunk.chunk_id] = entry.model_dump()
        self._save_index()
