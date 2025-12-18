"""
BM25 Sparse Retrieval

Phase R2: Dual Retrieval Engine

Implements BM25-based sparse retrieval over legal chunks.
Prioritizes exact section numbers and statute names.

NO LLMs used in this module.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


@dataclass
class BM25Result:
    """Result from BM25 retrieval."""
    chunk_id: str
    text: str
    score: float
    act: Optional[str]
    section: Optional[str]
    doc_type: str
    year: Optional[int]
    citation: Optional[str]
    court: Optional[str]


class BM25Retriever:
    """
    BM25-based sparse retrieval for legal chunks.
    
    Features:
    - Loads chunks from filesystem
    - Builds BM25 index over chunk text
    - Prioritizes exact section/statute matches
    - Returns scored results with metadata
    """
    
    def __init__(self, chunks_dir: str = "data/rag/chunks"):
        """
        Initialize BM25 retriever.
        
        Args:
            chunks_dir: Directory containing chunk JSON files
        """
        self.chunks_dir = Path(chunks_dir)
        self.chunks: List[Dict] = []
        self.chunk_texts: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._loaded = False
    
    def load(self) -> int:
        """
        Load chunks and build BM25 index.
        
        Returns:
            Number of chunks loaded
        """
        self.chunks = []
        self.chunk_texts = []
        
        # Load all chunk files
        for chunk_path in self.chunks_dir.glob("*.json"):
            if chunk_path.name == "index.json":
                continue
            
            try:
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    chunk = json.load(f)
                
                self.chunks.append(chunk)
                
                # Tokenize text for BM25
                text = chunk.get('text', '')
                tokens = self._tokenize(text)
                self.chunk_texts.append(tokens)
                
            except Exception as e:
                print(f"Warning: Failed to load {chunk_path}: {e}")
        
        # Build BM25 index
        if self.chunk_texts:
            self.bm25 = BM25Okapi(self.chunk_texts)
        
        self._loaded = True
        return len(self.chunks)
    
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        boost_section: bool = True,
        boost_act: bool = True,
    ) -> List[BM25Result]:
        """
        Query the BM25 index.
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            boost_section: Boost chunks matching section numbers in query
            boost_act: Boost chunks matching act names in query
            
        Returns:
            List of BM25Result sorted by score
        """
        if not self._loaded:
            self.load()
        
        if not self.bm25 or not self.chunks:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query_text)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Apply boosts
        if boost_section or boost_act:
            scores = self._apply_boosts(
                scores,
                query_text,
                boost_section=boost_section,
                boost_act=boost_act,
            )
        
        # Get top-k results
        scored_chunks = list(zip(range(len(self.chunks)), scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_chunks = scored_chunks[:top_k]
        
        # Build results
        results = []
        for idx, score in top_chunks:
            if score <= 0:
                continue
            
            chunk = self.chunks[idx]
            results.append(BM25Result(
                chunk_id=chunk.get('chunk_id', ''),
                text=chunk.get('text', ''),
                score=float(score),
                act=chunk.get('act'),
                section=chunk.get('section'),
                doc_type=chunk.get('doc_type', 'unknown'),
                year=chunk.get('year'),
                citation=chunk.get('citation'),
                court=chunk.get('court'),
            ))
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Preserves legal terms like section numbers.
        """
        # Lowercase
        text = text.lower()
        
        # Preserve section numbers: "section 420" -> "section_420"
        text = re.sub(r'section\s+(\d+)', r'section_\1', text)
        
        # Split on non-alphanumeric (but keep underscores)
        tokens = re.findall(r'[a-z0-9_]+', text)
        
        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def _apply_boosts(
        self,
        scores: List[float],
        query_text: str,
        boost_section: bool = True,
        boost_act: bool = True,
    ) -> List[float]:
        """
        Apply boosts for section and act matches.
        
        Args:
            scores: Original BM25 scores
            query_text: Query string
            boost_section: Boost section matches
            boost_act: Boost act matches
            
        Returns:
            Boosted scores
        """
        scores = list(scores)
        query_lower = query_text.lower()
        
        # Extract section number from query
        section_match = re.search(r'section\s+(\d+[a-z]?)', query_lower)
        query_section = section_match.group(1) if section_match else None
        
        # Extract act name from query
        query_acts = self._extract_acts(query_lower)
        
        for i, chunk in enumerate(self.chunks):
            boost = 1.0
            
            # Boost for section match
            if boost_section and query_section:
                chunk_section = chunk.get('section', '')
                if chunk_section and query_section in str(chunk_section).lower():
                    boost *= 2.0  # Double score for section match
            
            # Boost for act match
            if boost_act and query_acts:
                chunk_act = (chunk.get('act') or '').lower()
                for act in query_acts:
                    if act in chunk_act or chunk_act in act:
                        boost *= 1.5  # 1.5x for act match
                        break
            
            scores[i] *= boost
        
        return scores
    
    def _extract_acts(self, text: str) -> List[str]:
        """Extract act names from text."""
        acts = []
        
        # Common act patterns
        patterns = [
            r'\bipc\b',
            r'\bcrpc\b',
            r'\bcpc\b',
            r'\biea\b',
            r'indian penal code',
            r'criminal procedure',
            r'civil procedure',
            r'evidence act',
            r'minimum wages',
            r'contract act',
            r'companies act',
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                acts.append(pattern.replace(r'\b', '').strip())
        
        return acts
    
    def get_chunk_count(self) -> int:
        """Get number of indexed chunks."""
        return len(self.chunks)
    
    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        if not self._loaded:
            self.load()
        
        acts = {}
        doc_types = {}
        
        for chunk in self.chunks:
            act = chunk.get('act', 'unknown')
            doc_type = chunk.get('doc_type', 'unknown')
            
            acts[act] = acts.get(act, 0) + 1
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        return {
            "chunk_count": len(self.chunks),
            "acts": acts,
            "doc_types": doc_types,
            "indexed": self._loaded,
        }
