"""
Legal Edge Extractor - Phase 2: Legal Edge Extraction

Rule-based extraction of legal relationships from document text.
NO LLMs, NO embeddings, NO ML inference - pure pattern matching.

Edge Types Extracted:

Case → Section:
  - INTERPRETS_SECTION
  - APPLIES_SECTION
  - DISTINGUISHES_SECTION

Case → Case:
  - CITES_CASE
  - OVERRULES_CASE

Every edge stores:
  - source_node_id
  - target_node_id
  - relation_type
  - source_chunk_id
  - sentence_text (provenance)
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator

from .legal_graph_builder import (
    LegalGraphBuilder,
    NodeType,
    EdgeType as BaseEdgeType,
)

logger = logging.getLogger(__name__)


class RelationType(str, Enum):
    """Extended edge/relation types for Phase 2."""
    # Case → Section relations
    INTERPRETS_SECTION = "INTERPRETS_SECTION"
    APPLIES_SECTION = "APPLIES_SECTION"
    DISTINGUISHES_SECTION = "DISTINGUISHES_SECTION"
    
    # Case → Case relations
    CITES_CASE = "CITES_CASE"
    OVERRULES_CASE = "OVERRULES_CASE"
    
    # Phase 1 relations (for completeness)
    HAS_SECTION = "HAS_SECTION"
    MENTIONS_SECTION = "MENTIONS_SECTION"
    BELONGS_TO_ACT = "BELONGS_TO_ACT"


@dataclass
class ExtractedEdge:
    """
    Represents an extracted legal relationship.
    
    Every edge has full provenance for explainability.
    """
    source_node_id: str
    target_node_id: str
    relation_type: str
    source_chunk_id: str
    sentence_text: str
    confidence: float = 1.0  # Rule-based = 1.0 (deterministic)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def edge_key(self) -> str:
        """Unique key for deduplication."""
        return f"{self.source_node_id}|{self.target_node_id}|{self.relation_type}"


@dataclass
class ExtractionStats:
    """Statistics from edge extraction."""
    total_chunks_processed: int = 0
    total_sentences_processed: int = 0
    total_edges_extracted: int = 0
    edges_by_type: Dict[str, int] = field(default_factory=dict)
    chunks_with_edges: int = 0
    duplicate_edges_skipped: int = 0
    extraction_time_seconds: float = 0.0
    extraction_timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LegalPatterns:
    """
    Deterministic regex patterns for legal relationship extraction.
    
    All patterns are explicit and require textual proof.
    NO inference, NO guessing.
    """
    
    # ─────────────────────────────────────────────
    # SECTION REFERENCE PATTERNS
    # ─────────────────────────────────────────────
    
    # Pattern to extract section numbers (e.g., "Section 420", "S. 302", "Sec. 376")
    SECTION_PATTERN = re.compile(
        r'(?:Section|Sec\.|S\.)\s*(\d+(?:\s*\(\s*\d+\s*\))?(?:\s*\(\s*[a-z]\s*\))?)',
        re.IGNORECASE
    )
    
    # Pattern to extract act names near section references
    ACT_PATTERN = re.compile(
        r'(?:of\s+(?:the\s+)?)?'
        r'(IPC|CrPC|CPC|Indian\s+Penal\s+Code|'
        r'Code\s+of\s+Criminal\s+Procedure|'
        r'Code\s+of\s+Civil\s+Procedure|'
        r'Evidence\s+Act|Indian\s+Evidence\s+Act|'
        r'Contract\s+Act|Indian\s+Contract\s+Act|'
        r'Companies\s+Act|Constitution|'
        r'Limitation\s+Act|Arbitration\s+Act|'
        r'NDPS\s+Act|IT\s+Act|'
        r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Act(?:\s*,?\s*\d{4})?)',
        re.IGNORECASE
    )
    
    # ─────────────────────────────────────────────
    # INTERPRETS_SECTION PATTERNS
    # ─────────────────────────────────────────────
    
    INTERPRETS_PATTERNS = [
        # "interpreting Section X"
        re.compile(
            r'interpret(?:ing|ed|s)?\s+(?:the\s+)?(?:provision(?:s)?\s+(?:of\s+)?)?'
            r'(?:Section|Sec\.|S\.)\s*(\d+(?:\s*\(\s*\d+\s*\))?)',
            re.IGNORECASE
        ),
        # "interpretation of Section X"
        re.compile(
            r'interpretation\s+of\s+(?:the\s+)?(?:Section|Sec\.|S\.)\s*(\d+(?:\s*\(\s*\d+\s*\))?)',
            re.IGNORECASE
        ),
        # "construing Section X"
        re.compile(
            r'constru(?:ing|ed|es)?\s+(?:the\s+)?(?:Section|Sec\.|S\.)\s*(\d+(?:\s*\(\s*\d+\s*\))?)',
            re.IGNORECASE
        ),
        # "meaning of Section X"
        re.compile(
            r'meaning\s+of\s+(?:the\s+)?(?:Section|Sec\.|S\.)\s*(\d+(?:\s*\(\s*\d+\s*\))?)',
            re.IGNORECASE
        ),
        # "scope of Section X"
        re.compile(
            r'scope\s+of\s+(?:the\s+)?(?:Section|Sec\.|S\.)\s*(\d+(?:\s*\(\s*\d+\s*\))?)',
            re.IGNORECASE
        ),
    ]
    
    # ─────────────────────────────────────────────
    # APPLIES_SECTION PATTERNS
    # ─────────────────────────────────────────────
    
    APPLIES_PATTERNS = [
        # "applies Section X"
        re.compile(
            r'appl(?:y|ies|ied|ying)\s+(?:the\s+)?(?:Section|Sec\.|S\.)\s*(\d+(?:\s*\(\s*\d+\s*\))?)',
            re.IGNORECASE
        ),
        # "application of Section X"
        re.compile(
            r'application\s+of\s+(?:the\s+)?(?:Section|Sec\.|S\.)\s*(\d+(?:\s*\(\s*\d+\s*\))?)',
            re.IGNORECASE
        ),
        # "under Section X"
        re.compile(
            r'(?:convicted|charged|punishable|liable|offence)\s+under\s+(?:the\s+)?'
            r'(?:Section|Sec\.|S\.)\s*(\d+(?:\s*\(\s*\d+\s*\))?)',
            re.IGNORECASE
        ),
        # "Section X is applicable"
        re.compile(
            r'(?:Section|Sec\.|S\.)\s*(\d+(?:\s*\(\s*\d+\s*\))?)\s+(?:is\s+)?applicable',
            re.IGNORECASE
        ),
        # "invoking Section X"
        re.compile(
            r'invok(?:ing|ed|es)?\s+(?:the\s+)?(?:Section|Sec\.|S\.)\s*(\d+(?:\s*\(\s*\d+\s*\))?)',
            re.IGNORECASE
        ),
    ]
    
    # ─────────────────────────────────────────────
    # DISTINGUISHES_SECTION PATTERNS
    # ─────────────────────────────────────────────
    
    DISTINGUISHES_PATTERNS = [
        # "distinguished from Section X"
        re.compile(
            r'distinguish(?:ed|ing|es)?\s+(?:from\s+)?(?:the\s+)?'
            r'(?:Section|Sec\.|S\.)\s*(\d+(?:\s*\(\s*\d+\s*\))?)',
            re.IGNORECASE
        ),
        # "Section X is distinguishable"
        re.compile(
            r'(?:Section|Sec\.|S\.)\s*(\d+(?:\s*\(\s*\d+\s*\))?)\s+'
            r'(?:is\s+)?distinguishable',
            re.IGNORECASE
        ),
        # "not applicable... Section X"
        re.compile(
            r'(?:Section|Sec\.|S\.)\s*(\d+(?:\s*\(\s*\d+\s*\))?)\s+'
            r'(?:is\s+)?not\s+applicable',
            re.IGNORECASE
        ),
        # "inapplicable... Section X"
        re.compile(
            r'(?:Section|Sec\.|S\.)\s*(\d+(?:\s*\(\s*\d+\s*\))?)\s+'
            r'(?:is\s+)?inapplicable',
            re.IGNORECASE
        ),
    ]
    
    # ─────────────────────────────────────────────
    # CITES_CASE PATTERNS
    # ─────────────────────────────────────────────
    
    # Case citation patterns (Indian legal citations)
    CASE_CITATION_PATTERN = re.compile(
        r'(?:'
        # AIR citations: AIR 2020 SC 123
        r'AIR\s+\d{4}\s+(?:SC|SCC?|Del|Bom|Mad|Cal|Kar|All|Pat|Ori|Ker|Raj|Guj|MP|HP|J&K|Chh|Jhar|Utt)\s+\d+'
        r'|'
        # SCC citations: (2020) 5 SCC 123
        r'\(\d{4}\)\s+\d+\s+SCC\s+\d+'
        r'|'
        # SCR citations: 2020 SCR 123
        r'\d{4}\s+(?:\d+\s+)?SCR\s+\d+'
        r'|'
        # Generic: State v. Name or Name v. State
        r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+v\.?\s+(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        r')',
        re.IGNORECASE
    )
    
    CITES_PATTERNS = [
        # "relied upon in X v. Y"
        re.compile(
            r'relied\s+upon\s+(?:in\s+)?(.+?(?:v\.?\s+.+?)?(?:\(\d{4}\)|\d{4}))',
            re.IGNORECASE
        ),
        # "as held in X v. Y"
        re.compile(
            r'as\s+held\s+in\s+(.+?(?:v\.?\s+.+?)?(?:\(\d{4}\)|\d{4}))',
            re.IGNORECASE
        ),
        # "following X v. Y"
        re.compile(
            r'following\s+(?:the\s+decision\s+in\s+)?(.+?(?:v\.?\s+.+?)?(?:\(\d{4}\)|\d{4}))',
            re.IGNORECASE
        ),
        # "referred to X v. Y"
        re.compile(
            r'referr(?:ed|ing)\s+to\s+(.+?(?:v\.?\s+.+?)?(?:\(\d{4}\)|\d{4}))',
            re.IGNORECASE
        ),
        # "cited X v. Y"
        re.compile(
            r'cit(?:ed|ing|es)\s+(.+?(?:v\.?\s+.+?)?(?:\(\d{4}\)|\d{4}))',
            re.IGNORECASE
        ),
        # "in X v. Y, it was held"
        re.compile(
            r'in\s+(.+?(?:v\.?\s+.+?))\s*,?\s*(?:it\s+was\s+)?held',
            re.IGNORECASE
        ),
    ]
    
    # ─────────────────────────────────────────────
    # OVERRULES_CASE PATTERNS
    # ─────────────────────────────────────────────
    
    OVERRULES_PATTERNS = [
        # "overruled X v. Y"
        re.compile(
            r'overrul(?:ed|ing|es)\s+(?:the\s+decision\s+in\s+)?(.+?(?:v\.?\s+.+?)?(?:\(\d{4}\)|\d{4}))',
            re.IGNORECASE
        ),
        # "X v. Y is overruled"
        re.compile(
            r'(.+?(?:v\.?\s+.+?))\s+(?:is\s+)?(?:hereby\s+)?overruled',
            re.IGNORECASE
        ),
        # "no longer good law"
        re.compile(
            r'(.+?(?:v\.?\s+.+?))\s+(?:is\s+)?no\s+longer\s+good\s+law',
            re.IGNORECASE
        ),
        # "departed from X v. Y"
        re.compile(
            r'depart(?:ed|ing|s)\s+from\s+(.+?(?:v\.?\s+.+?)?(?:\(\d{4}\)|\d{4}))',
            re.IGNORECASE
        ),
        # "disapproved X v. Y"
        re.compile(
            r'disapprov(?:ed|ing|es)\s+(?:the\s+decision\s+in\s+)?(.+?(?:v\.?\s+.+?)?(?:\(\d{4}\)|\d{4}))',
            re.IGNORECASE
        ),
    ]


class LegalEdgeExtractor:
    """
    Extracts legal relationships from document text using rule-based patterns.
    
    NO LLMs, NO embeddings, NO ML inference.
    Pure deterministic pattern matching with full provenance.
    """
    
    def __init__(
        self,
        documents_dir: str = "data/rag/documents",
        chunks_dir: str = "data/rag/chunks",
        graph_path: str = "data/graph/legal_graph_v1.pkl",
        output_dir: str = "data/graph",
    ):
        """
        Initialize the edge extractor.
        
        Args:
            documents_dir: Path to canonical documents
            chunks_dir: Path to chunk storage
            graph_path: Path to existing graph (Phase 1)
            output_dir: Path for output
        """
        self.documents_dir = Path(documents_dir)
        self.chunks_dir = Path(chunks_dir)
        self.graph_path = Path(graph_path)
        self.output_dir = Path(output_dir)
        
        # Load existing graph or create new
        if self.graph_path.exists():
            logger.info(f"Loading existing graph from {self.graph_path}")
            self.builder = LegalGraphBuilder.load(str(self.graph_path))
        else:
            logger.warning(f"Graph not found at {self.graph_path}, creating new")
            self.builder = LegalGraphBuilder(
                documents_dir=str(self.documents_dir),
                chunks_dir=str(self.chunks_dir),
                output_dir=str(self.output_dir),
            )
        
        # Track extracted edges for deduplication
        self._extracted_edges: List[ExtractedEdge] = []
        self._seen_edge_keys: Set[str] = set()
        
        # Statistics
        self._stats = ExtractionStats()
    
    # ─────────────────────────────────────────────
    # SENTENCE SPLITTING
    # ─────────────────────────────────────────────
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Split text into sentences for pattern matching.
        
        Simple rule-based splitting (no ML).
        """
        if not text:
            return []
        
        # Split on sentence-ending punctuation
        # But be careful with abbreviations like "v." and "Sec."
        sentences = re.split(
            r'(?<=[.!?])\s+(?=[A-Z])',
            text
        )
        
        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    # ─────────────────────────────────────────────
    # SECTION EXTRACTION
    # ─────────────────────────────────────────────
    
    def extract_section_from_sentence(
        self,
        sentence: str
    ) -> List[Tuple[str, str]]:
        """
        Extract (act, section) pairs from a sentence.
        
        Returns:
            List of (act_name, section_number) tuples
        """
        results = []
        
        # Find all section references
        section_matches = LegalPatterns.SECTION_PATTERN.finditer(sentence)
        
        for match in section_matches:
            section = match.group(1).strip()
            
            # Try to find the act name near this section reference
            # Look in a window around the match
            start = max(0, match.start() - 100)
            end = min(len(sentence), match.end() + 100)
            context = sentence[start:end]
            
            act_match = LegalPatterns.ACT_PATTERN.search(context)
            if act_match:
                act = act_match.group(1).strip()
            else:
                # Default to IPC if no act specified (common in Indian legal text)
                act = "IPC"
            
            results.append((act, section))
        
        return results
    
    # ─────────────────────────────────────────────
    # CASE CITATION EXTRACTION
    # ─────────────────────────────────────────────
    
    def extract_case_citation(self, text: str) -> Optional[str]:
        """
        Extract a case citation from text.
        
        Returns:
            Case citation string or None
        """
        match = LegalPatterns.CASE_CITATION_PATTERN.search(text)
        if match:
            return match.group(0).strip()
        return None
    
    # ─────────────────────────────────────────────
    # RELATIONSHIP EXTRACTION
    # ─────────────────────────────────────────────
    
    def extract_interprets_section(
        self,
        sentence: str,
        source_case_id: str,
        source_chunk_id: str,
        default_act: str = "IPC"
    ) -> List[ExtractedEdge]:
        """Extract INTERPRETS_SECTION relationships."""
        edges = []
        
        for pattern in LegalPatterns.INTERPRETS_PATTERNS:
            matches = pattern.finditer(sentence)
            for match in matches:
                section = match.group(1).strip()
                
                # Find act name
                act_match = LegalPatterns.ACT_PATTERN.search(sentence)
                act = act_match.group(1) if act_match else default_act
                
                # Create edge
                target_id = self.builder.make_section_id(act, section)
                source_id = self.builder.make_case_id(source_case_id)
                
                edge = ExtractedEdge(
                    source_node_id=source_id,
                    target_node_id=target_id,
                    relation_type=RelationType.INTERPRETS_SECTION.value,
                    source_chunk_id=source_chunk_id,
                    sentence_text=sentence[:500],  # Truncate for storage
                    metadata={"act": act, "section": section}
                )
                edges.append(edge)
        
        return edges
    
    def extract_applies_section(
        self,
        sentence: str,
        source_case_id: str,
        source_chunk_id: str,
        default_act: str = "IPC"
    ) -> List[ExtractedEdge]:
        """Extract APPLIES_SECTION relationships."""
        edges = []
        
        for pattern in LegalPatterns.APPLIES_PATTERNS:
            matches = pattern.finditer(sentence)
            for match in matches:
                section = match.group(1).strip()
                
                # Find act name
                act_match = LegalPatterns.ACT_PATTERN.search(sentence)
                act = act_match.group(1) if act_match else default_act
                
                # Create edge
                target_id = self.builder.make_section_id(act, section)
                source_id = self.builder.make_case_id(source_case_id)
                
                edge = ExtractedEdge(
                    source_node_id=source_id,
                    target_node_id=target_id,
                    relation_type=RelationType.APPLIES_SECTION.value,
                    source_chunk_id=source_chunk_id,
                    sentence_text=sentence[:500],
                    metadata={"act": act, "section": section}
                )
                edges.append(edge)
        
        return edges
    
    def extract_distinguishes_section(
        self,
        sentence: str,
        source_case_id: str,
        source_chunk_id: str,
        default_act: str = "IPC"
    ) -> List[ExtractedEdge]:
        """Extract DISTINGUISHES_SECTION relationships."""
        edges = []
        
        for pattern in LegalPatterns.DISTINGUISHES_PATTERNS:
            matches = pattern.finditer(sentence)
            for match in matches:
                section = match.group(1).strip()
                
                # Find act name
                act_match = LegalPatterns.ACT_PATTERN.search(sentence)
                act = act_match.group(1) if act_match else default_act
                
                # Create edge
                target_id = self.builder.make_section_id(act, section)
                source_id = self.builder.make_case_id(source_case_id)
                
                edge = ExtractedEdge(
                    source_node_id=source_id,
                    target_node_id=target_id,
                    relation_type=RelationType.DISTINGUISHES_SECTION.value,
                    source_chunk_id=source_chunk_id,
                    sentence_text=sentence[:500],
                    metadata={"act": act, "section": section}
                )
                edges.append(edge)
        
        return edges
    
    def extract_cites_case(
        self,
        sentence: str,
        source_case_id: str,
        source_chunk_id: str
    ) -> List[ExtractedEdge]:
        """Extract CITES_CASE relationships."""
        edges = []
        
        for pattern in LegalPatterns.CITES_PATTERNS:
            matches = pattern.finditer(sentence)
            for match in matches:
                cited_case = match.group(1).strip()
                
                # Clean up the citation
                cited_case = re.sub(r'\s+', ' ', cited_case)
                cited_case = cited_case.strip('.,;:')
                
                if len(cited_case) < 5:  # Too short to be valid
                    continue
                
                # Create edge
                target_id = self.builder.make_case_id(cited_case)
                source_id = self.builder.make_case_id(source_case_id)
                
                # Don't create self-references
                if source_id == target_id:
                    continue
                
                edge = ExtractedEdge(
                    source_node_id=source_id,
                    target_node_id=target_id,
                    relation_type=RelationType.CITES_CASE.value,
                    source_chunk_id=source_chunk_id,
                    sentence_text=sentence[:500],
                    metadata={"cited_case": cited_case}
                )
                edges.append(edge)
        
        return edges
    
    def extract_overrules_case(
        self,
        sentence: str,
        source_case_id: str,
        source_chunk_id: str
    ) -> List[ExtractedEdge]:
        """Extract OVERRULES_CASE relationships."""
        edges = []
        
        for pattern in LegalPatterns.OVERRULES_PATTERNS:
            matches = pattern.finditer(sentence)
            for match in matches:
                overruled_case = match.group(1).strip()
                
                # Clean up the citation
                overruled_case = re.sub(r'\s+', ' ', overruled_case)
                overruled_case = overruled_case.strip('.,;:')
                
                if len(overruled_case) < 5:  # Too short to be valid
                    continue
                
                # Create edge
                target_id = self.builder.make_case_id(overruled_case)
                source_id = self.builder.make_case_id(source_case_id)
                
                # Don't create self-references
                if source_id == target_id:
                    continue
                
                edge = ExtractedEdge(
                    source_node_id=source_id,
                    target_node_id=target_id,
                    relation_type=RelationType.OVERRULES_CASE.value,
                    source_chunk_id=source_chunk_id,
                    sentence_text=sentence[:500],
                    metadata={"overruled_case": overruled_case}
                )
                edges.append(edge)
        
        return edges
    
    def extract_from_sentence(
        self,
        sentence: str,
        source_case_id: str,
        source_chunk_id: str,
        default_act: str = "IPC"
    ) -> List[ExtractedEdge]:
        """
        Extract all relationships from a single sentence.
        
        Args:
            sentence: The sentence text
            source_case_id: ID of the source case
            source_chunk_id: ID of the source chunk
            default_act: Default act name if not specified
            
        Returns:
            List of extracted edges
        """
        edges = []
        
        # Extract Case → Section relationships
        edges.extend(self.extract_interprets_section(
            sentence, source_case_id, source_chunk_id, default_act
        ))
        edges.extend(self.extract_applies_section(
            sentence, source_case_id, source_chunk_id, default_act
        ))
        edges.extend(self.extract_distinguishes_section(
            sentence, source_case_id, source_chunk_id, default_act
        ))
        
        # Extract Case → Case relationships
        edges.extend(self.extract_cites_case(
            sentence, source_case_id, source_chunk_id
        ))
        edges.extend(self.extract_overrules_case(
            sentence, source_case_id, source_chunk_id
        ))
        
        return edges
    
    # ─────────────────────────────────────────────
    # CHUNK PROCESSING
    # ─────────────────────────────────────────────
    
    def process_chunk(
        self,
        chunk_id: str,
        chunk_data: Dict[str, Any]
    ) -> List[ExtractedEdge]:
        """
        Process a single chunk and extract edges.
        
        Args:
            chunk_id: Chunk identifier
            chunk_data: Chunk metadata and text
            
        Returns:
            List of extracted edges
        """
        edges = []
        
        doc_type = chunk_data.get("doc_type", "")
        
        # Only process case_law chunks for relationship extraction
        if doc_type != "case_law":
            return edges
        
        # Get case identifier
        case_id = chunk_data.get("citation") or chunk_data.get("doc_id", chunk_id)
        
        # Get default act from chunk metadata
        default_act = chunk_data.get("act", "IPC")
        
        # Get text content
        text = chunk_data.get("text", "")
        if not text:
            return edges
        
        # Split into sentences
        sentences = self.split_sentences(text)
        self._stats.total_sentences_processed += len(sentences)
        
        # Extract from each sentence
        for sentence in sentences:
            sentence_edges = self.extract_from_sentence(
                sentence, case_id, chunk_id, default_act
            )
            edges.extend(sentence_edges)
        
        return edges
    
    # ─────────────────────────────────────────────
    # MAIN EXTRACTION
    # ─────────────────────────────────────────────
    
    def extract_from_chunks(self) -> int:
        """
        Extract edges from all chunks.
        
        Returns:
            Number of edges extracted
        """
        if not self.chunks_dir.exists():
            logger.warning(f"Chunks directory not found: {self.chunks_dir}")
            return 0
        
        edge_count = 0
        chunks_with_edges = 0
        
        # Try to load from index first
        index_path = self.chunks_dir / "index.json"
        chunks_to_process = []
        
        if index_path.exists():
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                
                # Index only has metadata, need to load full chunks
                for chunk_id in index_data.get("chunks", {}).keys():
                    chunk_path = self.chunks_dir / f"{chunk_id}.json"
                    if chunk_path.exists():
                        chunks_to_process.append(chunk_path)
                
            except Exception as e:
                logger.warning(f"Could not load index: {e}")
        
        # Fallback: find all chunk files
        if not chunks_to_process:
            chunks_to_process = [
                p for p in self.chunks_dir.glob("*.json")
                if p.name != "index.json"
            ]
        
        # Process each chunk
        for chunk_path in chunks_to_process:
            try:
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                
                chunk_id = chunk_data.get("chunk_id", chunk_path.stem)
                edges = self.process_chunk(chunk_id, chunk_data)
                
                if edges:
                    chunks_with_edges += 1
                    for edge in edges:
                        if self._add_edge(edge):
                            edge_count += 1
                
                self._stats.total_chunks_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_path}: {e}")
        
        self._stats.chunks_with_edges = chunks_with_edges
        logger.info(f"Extracted {edge_count} edges from {self._stats.total_chunks_processed} chunks")
        
        return edge_count
    
    def extract_from_documents(self) -> int:
        """
        Extract edges from canonical documents.
        
        Returns:
            Number of edges extracted
        """
        if not self.documents_dir.exists():
            logger.warning(f"Documents directory not found: {self.documents_dir}")
            return 0
        
        edge_count = 0
        
        for doc_path in self.documents_dir.glob("*.json"):
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                
                doc_type = doc.get("doc_type", "")
                if doc_type != "case_law":
                    continue
                
                # Get case identifier
                case_id = doc.get("citation") or doc.get("doc_id", doc_path.stem)
                default_act = doc.get("act", "IPC")
                
                # Get text
                text = doc.get("raw_text", "")
                if not text:
                    continue
                
                # Split and extract
                sentences = self.split_sentences(text)
                self._stats.total_sentences_processed += len(sentences)
                
                for sentence in sentences:
                    edges = self.extract_from_sentence(
                        sentence, case_id, f"doc_{doc_path.stem}", default_act
                    )
                    for edge in edges:
                        if self._add_edge(edge):
                            edge_count += 1
                
            except Exception as e:
                logger.error(f"Error processing document {doc_path}: {e}")
        
        return edge_count
    
    def _add_edge(self, edge: ExtractedEdge) -> bool:
        """
        Add an edge if not duplicate.
        
        Returns:
            True if added, False if duplicate
        """
        key = edge.edge_key()
        
        if key in self._seen_edge_keys:
            self._stats.duplicate_edges_skipped += 1
            return False
        
        self._seen_edge_keys.add(key)
        self._extracted_edges.append(edge)
        
        # Update stats
        rel_type = edge.relation_type
        self._stats.edges_by_type[rel_type] = self._stats.edges_by_type.get(rel_type, 0) + 1
        
        return True
    
    # ─────────────────────────────────────────────
    # GRAPH INTEGRATION
    # ─────────────────────────────────────────────
    
    def add_edges_to_graph(self) -> int:
        """
        Add extracted edges to the graph.
        
        Returns:
            Number of edges added
        """
        added = 0
        
        for edge in self._extracted_edges:
            # Ensure source and target nodes exist
            source_type = self._get_node_type_from_id(edge.source_node_id)
            target_type = self._get_node_type_from_id(edge.target_node_id)
            
            # Add nodes if they don't exist
            if edge.source_node_id not in self.builder.graph:
                if source_type == NodeType.CASE:
                    self.builder.add_case_node(
                        edge.source_node_id.replace("CASE::", ""),
                        metadata={"source": "edge_extraction"}
                    )
            
            if edge.target_node_id not in self.builder.graph:
                if target_type == NodeType.SECTION:
                    # Parse section ID to get act and section
                    parts = edge.target_node_id.split("::")
                    if len(parts) >= 3:
                        act = parts[1]
                        section = parts[2]
                        self.builder.add_section_node(act, section)
                        # Also ensure Act->Section edge exists
                        self.builder.add_has_section_edge(act, section)
                elif target_type == NodeType.CASE:
                    self.builder.add_case_node(
                        edge.target_node_id.replace("CASE::", ""),
                        metadata={"source": "edge_extraction"}
                    )
            
            # Add the edge
            if not self.builder.graph.has_edge(edge.source_node_id, edge.target_node_id):
                self.builder.graph.add_edge(
                    edge.source_node_id,
                    edge.target_node_id,
                    edge_type=edge.relation_type,
                    source_chunk_id=edge.source_chunk_id,
                    sentence_text=edge.sentence_text,
                    confidence=edge.confidence,
                    **edge.metadata
                )
                added += 1
                logger.debug(f"Added edge: {edge.source_node_id} --{edge.relation_type}--> {edge.target_node_id}")
        
        return added
    
    def _get_node_type_from_id(self, node_id: str) -> Optional[NodeType]:
        """Determine node type from ID prefix."""
        if node_id.startswith("ACT::"):
            return NodeType.ACT
        elif node_id.startswith("SECTION::"):
            return NodeType.SECTION
        elif node_id.startswith("CASE::"):
            return NodeType.CASE
        return None
    
    # ─────────────────────────────────────────────
    # MAIN EXTRACTION PIPELINE
    # ─────────────────────────────────────────────
    
    def extract(self) -> ExtractionStats:
        """
        Run the full extraction pipeline.
        
        Returns:
            ExtractionStats with results
        """
        import time
        start_time = time.time()
        
        logger.info("Starting legal edge extraction...")
        
        # Extract from documents
        doc_edges = self.extract_from_documents()
        logger.info(f"Extracted {doc_edges} edges from documents")
        
        # Extract from chunks
        chunk_edges = self.extract_from_chunks()
        logger.info(f"Extracted {chunk_edges} edges from chunks")
        
        # Add to graph
        added = self.add_edges_to_graph()
        logger.info(f"Added {added} edges to graph")
        
        # Update stats
        self._stats.total_edges_extracted = len(self._extracted_edges)
        self._stats.extraction_time_seconds = round(time.time() - start_time, 2)
        self._stats.extraction_timestamp = datetime.utcnow().isoformat()
        
        logger.info(f"Extraction complete in {self._stats.extraction_time_seconds}s")
        
        return self._stats
    
    # ─────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────
    
    def save(self, version: str = "v2") -> Tuple[Path, Path]:
        """
        Save the updated graph.
        
        Args:
            version: Version string for filename
            
        Returns:
            Tuple of (pickle_path, json_path)
        """
        # Update builder stats
        self.builder._stats = self.builder._calculate_stats()
        
        return self.builder.save(version=version)
    
    # ─────────────────────────────────────────────
    # REPORTING
    # ─────────────────────────────────────────────
    
    def get_stats(self) -> ExtractionStats:
        """Get extraction statistics."""
        return self._stats
    
    def get_extracted_edges(self) -> List[ExtractedEdge]:
        """Get all extracted edges."""
        return self._extracted_edges.copy()
    
    def print_summary(self) -> None:
        """Print extraction summary."""
        stats = self._stats
        
        print("\n" + "=" * 60)
        print("EDGE EXTRACTION SUMMARY")
        print("=" * 60)
        
        print(f"\n📊 Processing Stats:")
        print(f"   Chunks processed: {stats.total_chunks_processed}")
        print(f"   Sentences processed: {stats.total_sentences_processed}")
        print(f"   Chunks with edges: {stats.chunks_with_edges}")
        
        print(f"\n🔗 Edges Extracted: {stats.total_edges_extracted}")
        print(f"   Duplicates skipped: {stats.duplicate_edges_skipped}")
        
        if stats.edges_by_type:
            print("\n📈 Edges by Type:")
            for edge_type, count in sorted(stats.edges_by_type.items()):
                print(f"   → {edge_type}: {count}")
        
        print(f"\n⏱️  Extraction Time: {stats.extraction_time_seconds}s")
        print(f"📅 Timestamp: {stats.extraction_timestamp}")
        
        # Graph stats
        graph_stats = self.builder.get_stats()
        print(f"\n📊 Updated Graph Stats:")
        print(f"   Total nodes: {graph_stats.total_nodes}")
        print(f"   Total edges: {graph_stats.total_edges}")
        
        print("=" * 60 + "\n")
