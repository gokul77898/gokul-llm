"""
Legal Graph Builder - Phase 1: Legal Graph Foundation

Builds a directed graph of legal entities using NetworkX.
NO LLMs, NO embeddings - pure deterministic graph construction.

Node Types:
- ACT::<ACT_NAME>
- SECTION::<ACT_NAME>::<SECTION>
- CASE::<CASE_ID>

Edge Types:
- Act --HAS_SECTION--> Section
- Case --MENTIONS_SECTION--> Section
- Case --BELONGS_TO_ACT--> Act
"""

import json
import logging
import pickle
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import networkx as nx

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Node types in the legal graph."""
    ACT = "act"
    SECTION = "section"
    CASE = "case"


class EdgeType(str, Enum):
    """Edge types in the legal graph."""
    HAS_SECTION = "HAS_SECTION"
    MENTIONS_SECTION = "MENTIONS_SECTION"
    BELONGS_TO_ACT = "BELONGS_TO_ACT"


@dataclass
class GraphStats:
    """Statistics about the legal graph."""
    total_nodes: int = 0
    total_edges: int = 0
    nodes_by_type: Dict[str, int] = field(default_factory=dict)
    edges_by_type: Dict[str, int] = field(default_factory=dict)
    acts: List[str] = field(default_factory=list)
    orphan_sections: int = 0
    orphan_cases: int = 0
    build_time_seconds: float = 0.0
    build_timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    """Result of graph validation."""
    is_valid: bool = True
    orphan_sections: List[str] = field(default_factory=list)
    cases_without_act: List[str] = field(default_factory=list)
    cases_without_section: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LegalGraphBuilder:
    """
    Builds a directed graph of legal entities.
    
    Reads from:
    - data/rag/documents/ (canonical documents)
    - data/rag/chunks/ (chunk metadata)
    
    Outputs:
    - data/graph/legal_graph_v1.pkl (pickle)
    - data/graph/legal_graph_v1.json (JSON for inspection)
    
    NO LLMs, NO embeddings used.
    """
    
    def __init__(
        self,
        documents_dir: str = "data/rag/documents",
        chunks_dir: str = "data/rag/chunks",
        output_dir: str = "data/graph",
    ):
        """
        Initialize the graph builder.
        
        Args:
            documents_dir: Path to canonical documents
            chunks_dir: Path to chunk storage
            output_dir: Path for graph output
        """
        self.documents_dir = Path(documents_dir)
        self.chunks_dir = Path(chunks_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize directed graph
        self.graph = nx.DiGraph()
        
        # Track seen entities for deduplication
        self._seen_acts: Set[str] = set()
        self._seen_sections: Set[str] = set()
        self._seen_cases: Set[str] = set()
        
        # Build statistics
        self._stats = GraphStats()
    
    # ─────────────────────────────────────────────
    # NODE ID GENERATION (Deterministic)
    # ─────────────────────────────────────────────
    
    @staticmethod
    def make_act_id(act_name: str) -> str:
        """
        Generate deterministic Act node ID.
        
        Format: ACT::<NORMALIZED_ACT_NAME>
        """
        normalized = LegalGraphBuilder.normalize_act_name(act_name)
        return f"ACT::{normalized}"
    
    @staticmethod
    def make_section_id(act_name: str, section: str) -> str:
        """
        Generate deterministic Section node ID.
        
        Format: SECTION::<NORMALIZED_ACT_NAME>::<NORMALIZED_SECTION>
        """
        normalized_act = LegalGraphBuilder.normalize_act_name(act_name)
        normalized_section = LegalGraphBuilder.normalize_section(section)
        return f"SECTION::{normalized_act}::{normalized_section}"
    
    @staticmethod
    def make_case_id(case_identifier: str) -> str:
        """
        Generate deterministic Case node ID.
        
        Format: CASE::<CASE_IDENTIFIER>
        """
        # Use doc_id or citation as case identifier
        normalized = case_identifier.strip().replace(" ", "_").upper()
        return f"CASE::{normalized}"
    
    # ─────────────────────────────────────────────
    # NORMALIZATION FUNCTIONS
    # ─────────────────────────────────────────────
    
    @staticmethod
    def normalize_act_name(act_name: str) -> str:
        """
        Normalize act name for consistent node IDs.
        
        Examples:
        - "IPC" -> "IPC"
        - "Indian Penal Code" -> "INDIAN_PENAL_CODE"
        - "ipc" -> "IPC"
        - "CrPC" -> "CRPC"
        """
        if not act_name:
            return "UNKNOWN_ACT"
        
        # Common abbreviation mappings
        abbreviations = {
            "indian penal code": "IPC",
            "code of criminal procedure": "CRPC",
            "criminal procedure code": "CRPC",
            "code of civil procedure": "CPC",
            "civil procedure code": "CPC",
            "indian evidence act": "IEA",
            "evidence act": "IEA",
            "constitution of india": "COI",
            "constitution": "COI",
            "companies act": "COMPANIES_ACT",
            "income tax act": "IT_ACT",
            "gst act": "GST_ACT",
            "arbitration act": "ARBITRATION_ACT",
            "contract act": "CONTRACT_ACT",
            "indian contract act": "CONTRACT_ACT",
            "negotiable instruments act": "NI_ACT",
            "limitation act": "LIMITATION_ACT",
            "specific relief act": "SR_ACT",
            "transfer of property act": "TPA",
            "motor vehicles act": "MV_ACT",
            "information technology act": "IT_ACT_2000",
            "prevention of corruption act": "PCA",
            "narcotic drugs act": "NDPS_ACT",
            "ndps act": "NDPS_ACT",
        }
        
        act_lower = act_name.lower().strip()
        
        # Check for known abbreviations
        if act_lower in abbreviations:
            return abbreviations[act_lower]
        
        # Check if already an abbreviation (all caps, short)
        if act_name.isupper() and len(act_name) <= 10:
            return act_name.strip()
        
        # Normalize: uppercase, replace spaces with underscores
        normalized = act_name.strip().upper().replace(" ", "_")
        # Remove special characters except underscores
        normalized = re.sub(r'[^A-Z0-9_]', '', normalized)
        
        return normalized or "UNKNOWN_ACT"
    
    @staticmethod
    def normalize_section(section: str) -> str:
        """
        Normalize section number for consistent node IDs.
        
        Examples:
        - "420" -> "420"
        - "Section 420" -> "420"
        - "420(1)" -> "420_1"
        - "420(1)(a)" -> "420_1_A"
        - "420-A" -> "420_A"
        """
        if not section:
            return "UNKNOWN"
        
        # Remove "Section" prefix (case-insensitive)
        section = re.sub(r'(?i)^section\s*', '', section.strip())
        
        # Normalize parentheses to underscores
        section = section.replace("(", "_").replace(")", "")
        
        # Normalize hyphens to underscores
        section = section.replace("-", "_")
        
        # Remove extra underscores
        section = re.sub(r'_+', '_', section)
        section = section.strip('_')
        
        # Uppercase for consistency
        return section.upper() or "UNKNOWN"
    
    # ─────────────────────────────────────────────
    # NODE CREATION
    # ─────────────────────────────────────────────
    
    def add_act_node(self, act_name: str, metadata: Optional[Dict] = None) -> str:
        """
        Add an Act node to the graph.
        
        Args:
            act_name: Name of the act
            metadata: Optional metadata dict
            
        Returns:
            Node ID
        """
        node_id = self.make_act_id(act_name)
        
        if node_id in self._seen_acts:
            return node_id
        
        self._seen_acts.add(node_id)
        
        attrs = {
            "node_type": NodeType.ACT.value,
            "name": act_name,
            "normalized_name": self.normalize_act_name(act_name),
        }
        if metadata:
            attrs.update(metadata)
        
        self.graph.add_node(node_id, **attrs)
        logger.debug(f"Added Act node: {node_id}")
        
        return node_id
    
    def add_section_node(
        self,
        act_name: str,
        section: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a Section node to the graph.
        
        Args:
            act_name: Name of the parent act
            section: Section number/identifier
            metadata: Optional metadata dict
            
        Returns:
            Node ID
        """
        node_id = self.make_section_id(act_name, section)
        
        if node_id in self._seen_sections:
            return node_id
        
        self._seen_sections.add(node_id)
        
        attrs = {
            "node_type": NodeType.SECTION.value,
            "act": act_name,
            "section": section,
            "normalized_act": self.normalize_act_name(act_name),
            "normalized_section": self.normalize_section(section),
        }
        if metadata:
            attrs.update(metadata)
        
        self.graph.add_node(node_id, **attrs)
        logger.debug(f"Added Section node: {node_id}")
        
        return node_id
    
    def add_case_node(
        self,
        case_id: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a Case node to the graph.
        
        Args:
            case_id: Case identifier (doc_id or citation)
            metadata: Optional metadata dict
            
        Returns:
            Node ID
        """
        node_id = self.make_case_id(case_id)
        
        if node_id in self._seen_cases:
            return node_id
        
        self._seen_cases.add(node_id)
        
        attrs = {
            "node_type": NodeType.CASE.value,
            "case_id": case_id,
        }
        if metadata:
            attrs.update(metadata)
        
        self.graph.add_node(node_id, **attrs)
        logger.debug(f"Added Case node: {node_id}")
        
        return node_id
    
    # ─────────────────────────────────────────────
    # EDGE CREATION
    # ─────────────────────────────────────────────
    
    def add_has_section_edge(self, act_name: str, section: str) -> Tuple[str, str]:
        """
        Add HAS_SECTION edge: Act --> Section
        
        Args:
            act_name: Name of the act
            section: Section number
            
        Returns:
            Tuple of (act_node_id, section_node_id)
        """
        act_id = self.add_act_node(act_name)
        section_id = self.add_section_node(act_name, section)
        
        if not self.graph.has_edge(act_id, section_id):
            self.graph.add_edge(
                act_id,
                section_id,
                edge_type=EdgeType.HAS_SECTION.value
            )
            logger.debug(f"Added edge: {act_id} --HAS_SECTION--> {section_id}")
        
        return act_id, section_id
    
    def add_mentions_section_edge(
        self,
        case_id: str,
        act_name: str,
        section: str
    ) -> Tuple[str, str]:
        """
        Add MENTIONS_SECTION edge: Case --> Section
        
        Args:
            case_id: Case identifier
            act_name: Name of the act
            section: Section number
            
        Returns:
            Tuple of (case_node_id, section_node_id)
        """
        case_node_id = self.add_case_node(case_id)
        section_id = self.add_section_node(act_name, section)
        
        # Ensure Act-->Section edge exists
        self.add_has_section_edge(act_name, section)
        
        if not self.graph.has_edge(case_node_id, section_id):
            self.graph.add_edge(
                case_node_id,
                section_id,
                edge_type=EdgeType.MENTIONS_SECTION.value
            )
            logger.debug(f"Added edge: {case_node_id} --MENTIONS_SECTION--> {section_id}")
        
        return case_node_id, section_id
    
    def add_belongs_to_act_edge(self, case_id: str, act_name: str) -> Tuple[str, str]:
        """
        Add BELONGS_TO_ACT edge: Case --> Act
        
        Args:
            case_id: Case identifier
            act_name: Name of the act
            
        Returns:
            Tuple of (case_node_id, act_node_id)
        """
        case_node_id = self.add_case_node(case_id)
        act_id = self.add_act_node(act_name)
        
        if not self.graph.has_edge(case_node_id, act_id):
            self.graph.add_edge(
                case_node_id,
                act_id,
                edge_type=EdgeType.BELONGS_TO_ACT.value
            )
            logger.debug(f"Added edge: {case_node_id} --BELONGS_TO_ACT--> {act_id}")
        
        return case_node_id, act_id
    
    # ─────────────────────────────────────────────
    # GRAPH BUILDING
    # ─────────────────────────────────────────────
    
    def build_from_documents(self) -> int:
        """
        Build graph from canonical documents.
        
        Returns:
            Number of documents processed
        """
        if not self.documents_dir.exists():
            logger.warning(f"Documents directory not found: {self.documents_dir}")
            return 0
        
        doc_count = 0
        
        for doc_path in self.documents_dir.glob("*.json"):
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                
                self._process_document(doc)
                doc_count += 1
                
            except Exception as e:
                logger.error(f"Error processing document {doc_path}: {e}")
        
        logger.info(f"Processed {doc_count} documents")
        return doc_count
    
    def build_from_chunks(self) -> int:
        """
        Build graph from chunk metadata.
        
        Returns:
            Number of chunks processed
        """
        if not self.chunks_dir.exists():
            logger.warning(f"Chunks directory not found: {self.chunks_dir}")
            return 0
        
        chunk_count = 0
        
        # First try to load from index
        index_path = self.chunks_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                
                chunks = index_data.get("chunks", {})
                for chunk_id, chunk_meta in chunks.items():
                    self._process_chunk_metadata(chunk_id, chunk_meta)
                    chunk_count += 1
                
                logger.info(f"Processed {chunk_count} chunks from index")
                return chunk_count
                
            except Exception as e:
                logger.warning(f"Could not load index, falling back to individual files: {e}")
        
        # Fallback: read individual chunk files
        for chunk_path in self.chunks_dir.glob("*.json"):
            if chunk_path.name == "index.json":
                continue
            
            try:
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    chunk = json.load(f)
                
                self._process_chunk_metadata(chunk.get("chunk_id", chunk_path.stem), chunk)
                chunk_count += 1
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_path}: {e}")
        
        logger.info(f"Processed {chunk_count} chunks")
        return chunk_count
    
    def _process_document(self, doc: Dict) -> None:
        """Process a single document and add to graph."""
        doc_id = doc.get("doc_id", "")
        doc_type = doc.get("doc_type", "")
        act = doc.get("act")
        court = doc.get("court")
        year = doc.get("year")
        citation = doc.get("citation")
        
        if doc_type == "bare_act" and act:
            # Add Act node
            self.add_act_node(act, metadata={
                "year": year,
                "source": "document",
            })
        
        elif doc_type == "case_law":
            # Add Case node
            case_identifier = citation or doc_id
            self.add_case_node(case_identifier, metadata={
                "doc_id": doc_id,
                "court": court,
                "year": year,
                "citation": citation,
                "source": "document",
            })
            
            # If case mentions an act, add edge
            if act:
                self.add_belongs_to_act_edge(case_identifier, act)
    
    def _process_chunk_metadata(self, chunk_id: str, chunk: Dict) -> None:
        """Process chunk metadata and add to graph."""
        doc_id = chunk.get("doc_id", "")
        doc_type = chunk.get("doc_type", "")
        act = chunk.get("act")
        section = chunk.get("section")
        court = chunk.get("court")
        year = chunk.get("year")
        citation = chunk.get("citation")
        
        if doc_type == "bare_act":
            # Add Act and Section nodes
            if act:
                self.add_act_node(act, metadata={"year": year})
                
                if section:
                    self.add_has_section_edge(act, section)
        
        elif doc_type == "case_law":
            # Add Case node
            case_identifier = citation or doc_id
            self.add_case_node(case_identifier, metadata={
                "doc_id": doc_id,
                "court": court,
                "year": year,
                "citation": citation,
                "source": "chunk",
            })
            
            # Add edges
            if act:
                self.add_belongs_to_act_edge(case_identifier, act)
                
                if section:
                    self.add_mentions_section_edge(case_identifier, act, section)
    
    def build(self) -> GraphStats:
        """
        Build the complete legal graph.
        
        Reads from both documents and chunks directories.
        
        Returns:
            GraphStats with build statistics
        """
        import time
        start_time = time.time()
        
        logger.info("Starting legal graph build...")
        
        # Build from documents first
        doc_count = self.build_from_documents()
        
        # Then from chunks (may add more detail)
        chunk_count = self.build_from_chunks()
        
        # Calculate statistics
        build_time = time.time() - start_time
        self._stats = self._calculate_stats()
        self._stats.build_time_seconds = round(build_time, 2)
        self._stats.build_timestamp = datetime.utcnow().isoformat()
        
        logger.info(f"Graph build complete in {build_time:.2f}s")
        logger.info(f"  Documents processed: {doc_count}")
        logger.info(f"  Chunks processed: {chunk_count}")
        logger.info(f"  Total nodes: {self._stats.total_nodes}")
        logger.info(f"  Total edges: {self._stats.total_edges}")
        
        return self._stats
    
    def _calculate_stats(self) -> GraphStats:
        """Calculate graph statistics."""
        stats = GraphStats()
        
        stats.total_nodes = self.graph.number_of_nodes()
        stats.total_edges = self.graph.number_of_edges()
        
        # Count nodes by type
        nodes_by_type = {t.value: 0 for t in NodeType}
        acts = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            node_type = attrs.get("node_type", "unknown")
            nodes_by_type[node_type] = nodes_by_type.get(node_type, 0) + 1
            
            if node_type == NodeType.ACT.value:
                acts.append(attrs.get("name", node_id))
        
        stats.nodes_by_type = nodes_by_type
        stats.acts = sorted(acts)
        
        # Count edges by type
        edges_by_type = {t.value: 0 for t in EdgeType}
        
        for u, v, attrs in self.graph.edges(data=True):
            edge_type = attrs.get("edge_type", "unknown")
            edges_by_type[edge_type] = edges_by_type.get(edge_type, 0) + 1
        
        stats.edges_by_type = edges_by_type
        
        return stats
    
    # ─────────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────────
    
    def validate(self) -> ValidationResult:
        """
        Validate the graph for integrity.
        
        Checks:
        - No orphan Section nodes (must have parent Act)
        - No Case without Act or Section
        - Graph is connected where expected
        
        Returns:
            ValidationResult
        """
        result = ValidationResult()
        
        # Check for orphan sections (no incoming HAS_SECTION edge)
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") == NodeType.SECTION.value:
                # Check if any Act points to this section
                has_parent = False
                for pred in self.graph.predecessors(node_id):
                    pred_attrs = self.graph.nodes[pred]
                    if pred_attrs.get("node_type") == NodeType.ACT.value:
                        edge_data = self.graph.get_edge_data(pred, node_id)
                        if edge_data and edge_data.get("edge_type") == EdgeType.HAS_SECTION.value:
                            has_parent = True
                            break
                
                if not has_parent:
                    result.orphan_sections.append(node_id)
                    result.warnings.append(f"Orphan section: {node_id}")
        
        # Check for cases without connections
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") == NodeType.CASE.value:
                has_act = False
                has_section = False
                
                for succ in self.graph.successors(node_id):
                    succ_attrs = self.graph.nodes[succ]
                    if succ_attrs.get("node_type") == NodeType.ACT.value:
                        has_act = True
                    elif succ_attrs.get("node_type") == NodeType.SECTION.value:
                        has_section = True
                
                if not has_act:
                    result.cases_without_act.append(node_id)
                    result.warnings.append(f"Case without act: {node_id}")
                
                if not has_section:
                    result.cases_without_section.append(node_id)
                    # This is less severe, just a warning
        
        # Set validity
        if result.orphan_sections:
            result.is_valid = False
            result.errors.append(f"Found {len(result.orphan_sections)} orphan sections")
        
        self._stats.orphan_sections = len(result.orphan_sections)
        self._stats.orphan_cases = len(result.cases_without_act)
        
        return result
    
    # ─────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────
    
    def save(self, version: str = "v1") -> Tuple[Path, Path]:
        """
        Save the graph to disk.
        
        Args:
            version: Version string for filename
            
        Returns:
            Tuple of (pickle_path, json_path)
        """
        pickle_path = self.output_dir / f"legal_graph_{version}.pkl"
        json_path = self.output_dir / f"legal_graph_{version}.json"
        
        # Save as pickle (full graph)
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.graph, f)
        logger.info(f"Saved graph to {pickle_path}")
        
        # Save as JSON (for inspection)
        graph_data = {
            "metadata": {
                "version": version,
                "created_at": datetime.utcnow().isoformat(),
                "stats": self._stats.to_dict(),
            },
            "nodes": [
                {"id": n, **attrs}
                for n, attrs in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **attrs}
                for u, v, attrs in self.graph.edges(data=True)
            ],
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved graph JSON to {json_path}")
        
        return pickle_path, json_path
    
    @classmethod
    def load(cls, pickle_path: str) -> 'LegalGraphBuilder':
        """
        Load a graph from pickle file.
        
        Args:
            pickle_path: Path to pickle file
            
        Returns:
            LegalGraphBuilder with loaded graph
        """
        builder = cls()
        
        with open(pickle_path, 'rb') as f:
            builder.graph = pickle.load(f)
        
        # Rebuild seen sets
        for node_id, attrs in builder.graph.nodes(data=True):
            node_type = attrs.get("node_type")
            if node_type == NodeType.ACT.value:
                builder._seen_acts.add(node_id)
            elif node_type == NodeType.SECTION.value:
                builder._seen_sections.add(node_id)
            elif node_type == NodeType.CASE.value:
                builder._seen_cases.add(node_id)
        
        builder._stats = builder._calculate_stats()
        
        logger.info(f"Loaded graph from {pickle_path}")
        return builder
    
    # ─────────────────────────────────────────────
    # QUERY METHODS (for Phase 2)
    # ─────────────────────────────────────────────
    
    def get_sections_for_act(self, act_name: str) -> List[str]:
        """Get all sections for an act."""
        act_id = self.make_act_id(act_name)
        
        if act_id not in self.graph:
            return []
        
        sections = []
        for succ in self.graph.successors(act_id):
            attrs = self.graph.nodes[succ]
            if attrs.get("node_type") == NodeType.SECTION.value:
                sections.append(attrs.get("section", succ))
        
        return sorted(sections)
    
    def get_cases_for_section(self, act_name: str, section: str) -> List[str]:
        """Get all cases mentioning a section."""
        section_id = self.make_section_id(act_name, section)
        
        if section_id not in self.graph:
            return []
        
        cases = []
        for pred in self.graph.predecessors(section_id):
            attrs = self.graph.nodes[pred]
            if attrs.get("node_type") == NodeType.CASE.value:
                cases.append(attrs.get("case_id", pred))
        
        return cases
    
    def get_stats(self) -> GraphStats:
        """Get current graph statistics."""
        return self._stats
    
    def print_summary(self) -> None:
        """Print a summary of the graph."""
        stats = self._stats
        
        print("\n" + "=" * 60)
        print("LEGAL GRAPH SUMMARY")
        print("=" * 60)
        print(f"\nTotal Nodes: {stats.total_nodes}")
        print(f"Total Edges: {stats.total_edges}")
        
        print("\nNodes by Type:")
        for node_type, count in stats.nodes_by_type.items():
            print(f"  - {node_type}: {count}")
        
        print("\nEdges by Type:")
        for edge_type, count in stats.edges_by_type.items():
            print(f"  - {edge_type}: {count}")
        
        if stats.acts:
            print(f"\nActs ({len(stats.acts)}):")
            for act in stats.acts[:10]:  # Show first 10
                print(f"  - {act}")
            if len(stats.acts) > 10:
                print(f"  ... and {len(stats.acts) - 10} more")
        
        print(f"\nBuild Time: {stats.build_time_seconds}s")
        print(f"Build Timestamp: {stats.build_timestamp}")
        print("=" * 60 + "\n")
