"""
Fusion Logic for Dual Retrieval

Phase R2: Dual Retrieval Engine

Combines BM25 (sparse) and Dense (ChromaDB) retrieval results
using deterministic score fusion with legal-aware boosting.

NO LLMs used in this module.
"""

import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

from .bm25 import BM25Result
from .dense import DenseResult


@dataclass
class FusedResult:
    """Result from fused retrieval."""
    chunk_id: str
    text: str
    score: float  # Fused score
    bm25_score: float
    dense_score: float
    act: Optional[str]
    section: Optional[str]
    doc_type: str
    year: Optional[int]
    citation: Optional[str]
    court: Optional[str]
    source: str  # "bm25", "dense", or "fused"


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores to 0-1 range.
    
    Args:
        scores: List of raw scores
        
    Returns:
        Normalized scores
    """
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    return [(s - min_score) / (max_score - min_score) for s in scores]


def fuse_results(
    bm25_results: List[BM25Result],
    dense_results: List[DenseResult],
    query_text: str,
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6,
    boost_section_match: float = 1.5,
    boost_act_match: float = 1.3,
    penalize_wrong_statute: float = 0.7,
) -> List[FusedResult]:
    """
    Fuse BM25 and dense retrieval results.
    
    Fusion strategy:
    1. Normalize scores from both sources
    2. Combine with weighted average
    3. Apply boosts for section/act matches
    4. Apply penalties for mismatches
    
    Args:
        bm25_results: Results from BM25 retrieval
        dense_results: Results from dense retrieval
        query_text: Original query for boost calculation
        bm25_weight: Weight for BM25 scores (default 0.4)
        dense_weight: Weight for dense scores (default 0.6)
        boost_section_match: Boost multiplier for section match
        boost_act_match: Boost multiplier for act match
        penalize_wrong_statute: Penalty multiplier for wrong statute
        
    Returns:
        Fused results sorted by score
    """
    # Extract query metadata
    query_section = _extract_section(query_text)
    query_acts = _extract_acts(query_text)
    query_year = _extract_year(query_text)
    
    # Build chunk map with scores from both sources
    chunk_map: Dict[str, Dict] = {}
    
    # Add BM25 results
    bm25_scores = [r.score for r in bm25_results]
    norm_bm25 = normalize_scores(bm25_scores)
    
    for i, result in enumerate(bm25_results):
        chunk_map[result.chunk_id] = {
            "text": result.text,
            "bm25_score": norm_bm25[i] if i < len(norm_bm25) else 0,
            "dense_score": 0,
            "act": result.act,
            "section": result.section,
            "doc_type": result.doc_type,
            "year": result.year,
            "citation": result.citation,
            "court": result.court,
            "source": "bm25",
        }
    
    # Add/merge dense results
    dense_scores = [r.score for r in dense_results]
    norm_dense = normalize_scores(dense_scores)
    
    for i, result in enumerate(dense_results):
        if result.chunk_id in chunk_map:
            # Merge with existing
            chunk_map[result.chunk_id]["dense_score"] = norm_dense[i] if i < len(norm_dense) else 0
            chunk_map[result.chunk_id]["source"] = "fused"
        else:
            # Add new
            chunk_map[result.chunk_id] = {
                "text": result.text,
                "bm25_score": 0,
                "dense_score": norm_dense[i] if i < len(norm_dense) else 0,
                "act": result.act,
                "section": result.section,
                "doc_type": result.doc_type,
                "year": result.year,
                "citation": result.citation,
                "court": result.court,
                "source": "dense",
            }
    
    # Calculate fused scores with boosts/penalties
    fused_results = []
    
    for chunk_id, data in chunk_map.items():
        # Base fused score
        base_score = (
            bm25_weight * data["bm25_score"] +
            dense_weight * data["dense_score"]
        )
        
        # Apply boosts
        boost = 1.0
        
        # Boost for section match
        if query_section and data["section"]:
            if query_section.lower() in str(data["section"]).lower():
                boost *= boost_section_match
        
        # Boost for act match
        if query_acts and data["act"]:
            chunk_act = data["act"].lower()
            for act in query_acts:
                if act in chunk_act or chunk_act in act:
                    boost *= boost_act_match
                    break
            else:
                # Penalize if query mentions specific act but chunk is different
                if len(query_acts) == 1:
                    boost *= penalize_wrong_statute
        
        # Penalize year mismatch (if query specifies year)
        if query_year and data["year"]:
            if abs(query_year - data["year"]) > 10:
                boost *= 0.9  # Slight penalty for distant years
        
        final_score = base_score * boost
        
        fused_results.append(FusedResult(
            chunk_id=chunk_id,
            text=data["text"],
            score=final_score,
            bm25_score=data["bm25_score"],
            dense_score=data["dense_score"],
            act=data["act"],
            section=data["section"],
            doc_type=data["doc_type"],
            year=data["year"],
            citation=data["citation"],
            court=data["court"],
            source=data["source"],
        ))
    
    # Sort by fused score
    fused_results.sort(key=lambda x: x.score, reverse=True)
    
    return fused_results


def _extract_section(text: str) -> Optional[str]:
    """Extract section number from text."""
    match = re.search(r'section\s+(\d+[a-z]?)', text.lower())
    return match.group(1) if match else None


def _extract_acts(text: str) -> Set[str]:
    """Extract act names from text."""
    acts = set()
    text_lower = text.lower()
    
    patterns = [
        (r'\bipc\b', 'ipc'),
        (r'\bcrpc\b', 'crpc'),
        (r'\bcpc\b', 'cpc'),
        (r'\biea\b', 'iea'),
        (r'indian penal code', 'ipc'),
        (r'criminal procedure', 'crpc'),
        (r'civil procedure', 'cpc'),
        (r'evidence act', 'iea'),
        (r'minimum wages', 'minimum wages act'),
        (r'contract act', 'contract act'),
        (r'companies act', 'companies act'),
    ]
    
    for pattern, act_name in patterns:
        if re.search(pattern, text_lower):
            acts.add(act_name)
    
    return acts


def _extract_year(text: str) -> Optional[int]:
    """Extract year from text."""
    match = re.search(r'\b(19|20)\d{2}\b', text)
    return int(match.group(0)) if match else None


class FusionRetriever:
    """
    Fusion retriever combining BM25 and Dense retrieval.
    
    Provides a unified interface for dual retrieval with
    deterministic fusion logic.
    """
    
    def __init__(
        self,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
    ):
        """
        Initialize fusion retriever.
        
        Args:
            bm25_weight: Weight for BM25 scores
            dense_weight: Weight for dense scores
        """
        from .bm25 import BM25Retriever
        from .dense import DenseRetriever
        
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self._initialized = False
    
    def initialize(self, index_dense: bool = True) -> Dict:
        """
        Initialize both retrievers.
        
        Args:
            index_dense: Whether to index chunks into ChromaDB
            
        Returns:
            Initialization statistics
        """
        # Load BM25 index
        bm25_count = self.bm25.load()
        
        # Index into ChromaDB if requested
        dense_indexed = 0
        if index_dense:
            dense_indexed = self.dense.index_chunks()
        
        dense_count = self.dense.get_chunk_count()
        
        self._initialized = True
        
        return {
            "bm25_chunks": bm25_count,
            "dense_chunks": dense_count,
            "dense_indexed": dense_indexed,
        }
    
    def query(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> List[FusedResult]:
        """
        Query with fusion of BM25 and dense retrieval.
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            
        Returns:
            Fused results sorted by score
        """
        if not self._initialized:
            self.initialize()
        
        # Get results from both sources
        # Fetch more than top_k to allow for fusion
        fetch_k = min(top_k * 2, 50)
        
        bm25_results = self.bm25.query(query_text, top_k=fetch_k)
        dense_results = self.dense.query(query_text, top_k=fetch_k)
        
        # Fuse results
        fused = fuse_results(
            bm25_results=bm25_results,
            dense_results=dense_results,
            query_text=query_text,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
        )
        
        return fused[:top_k]
    
    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        return {
            "bm25": self.bm25.get_stats(),
            "dense": self.dense.get_stats(),
            "bm25_weight": self.bm25_weight,
            "dense_weight": self.dense_weight,
            "initialized": self._initialized,
        }
