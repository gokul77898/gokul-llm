"""
Precedent Labeler - Phase 5B

Labels precedents with legal roles based on relation chains.

PURELY STRUCTURED LABELING.
NO NATURAL LANGUAGE GENERATION.
NO LLM USAGE.
NO GRAPH ACCESS.

Uses ONLY:
- precedents from Phase 5A
- relation_chain for labeling
"""

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Dict, Any

from .precedent_extractor import PrecedentExplanation

logger = logging.getLogger(__name__)


@dataclass
class LabeledPrecedent:
    """
    Precedent with legal role label.
    
    NO natural language - only structured data.
    """
    cited_semantic_id: str
    node_id: str
    authority_level: str
    legal_role: Literal[
        "primary_statute",
        "judicial_interpretation",
        "applied_precedent",
        "supporting_precedent",
        "overruled_precedent"
    ]
    relation_chain: List[str] = field(default_factory=list)
    graph_path: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cited_semantic_id": self.cited_semantic_id,
            "node_id": self.node_id,
            "authority_level": self.authority_level,
            "legal_role": self.legal_role,
            "relation_chain": self.relation_chain,
            "graph_path": self.graph_path,
        }


class PrecedentLabeler:
    """
    Labels precedents with legal roles.
    
    NO graph access.
    NO LLM calls.
    NO natural language generation.
    
    Uses ONLY relation_chain from PrecedentExplanation.
    """
    
    # Legal role priority (higher = more important)
    ROLE_PRIORITY = {
        "primary_statute": 5,
        "judicial_interpretation": 4,
        "applied_precedent": 3,
        "supporting_precedent": 2,
        "overruled_precedent": 1,
    }
    
    # ─────────────────────────────────────────────
    # LABELING RULES
    # ─────────────────────────────────────────────
    
    def _classify_legal_role(
        self,
        precedent: PrecedentExplanation
    ) -> Literal[
        "primary_statute",
        "judicial_interpretation",
        "applied_precedent",
        "supporting_precedent",
        "overruled_precedent"
    ]:
        """
        Classify legal role based on STRICT ORDER.
        
        Rules (in order):
        1. ACT or SECTION → primary_statute
        2. relation_chain contains OVERRULED → overruled_precedent
        3. relation_chain contains INTERPRETS_SECTION → judicial_interpretation
        4. relation_chain contains APPLIES_SECTION → applied_precedent
        5. CASE default → supporting_precedent
        
        Args:
            precedent: PrecedentExplanation
            
        Returns:
            Legal role
        """
        # Rule 1: ACT or SECTION → primary_statute
        if precedent.node_type in ["ACT", "SECTION"]:
            return "primary_statute"
        
        # Rule 2: relation_chain contains OVERRULED → overruled_precedent
        if any("OVERRULE" in rel.upper() for rel in precedent.relation_chain):
            return "overruled_precedent"
        
        # Rule 3: relation_chain contains INTERPRETS_SECTION → judicial_interpretation
        if any("INTERPRETS_SECTION" in rel for rel in precedent.relation_chain):
            return "judicial_interpretation"
        
        # Rule 4: relation_chain contains APPLIES_SECTION → applied_precedent
        if any("APPLIES_SECTION" in rel for rel in precedent.relation_chain):
            return "applied_precedent"
        
        # Rule 5: CASE default → supporting_precedent
        return "supporting_precedent"
    
    # ─────────────────────────────────────────────
    # MAIN LABELING METHOD
    # ─────────────────────────────────────────────
    
    def label(
        self,
        precedents: List[PrecedentExplanation]
    ) -> List[LabeledPrecedent]:
        """
        Label precedents with legal roles.
        
        NO graph access.
        NO LLM calls.
        
        Args:
            precedents: List of PrecedentExplanation from Phase 5A
            
        Returns:
            List of LabeledPrecedent, sorted by role priority
        """
        logger.info(f"Labeling {len(precedents)} precedents")
        
        # Handle empty input
        if not precedents:
            logger.info("No precedents to label, returning empty list")
            return []
        
        labeled = []
        
        # Label each precedent
        for precedent in precedents:
            # Classify legal role
            legal_role = self._classify_legal_role(precedent)
            
            # Create labeled precedent
            labeled_precedent = LabeledPrecedent(
                cited_semantic_id=precedent.cited_semantic_id,
                node_id=precedent.node_id,
                authority_level=precedent.authority_level,
                legal_role=legal_role,
                relation_chain=precedent.relation_chain,
                graph_path=precedent.graph_path,
            )
            
            labeled.append(labeled_precedent)
        
        # Sort by role priority
        labeled.sort(
            key=lambda x: self.ROLE_PRIORITY.get(x.legal_role, 0),
            reverse=True
        )
        
        logger.info(f"Labeled {len(labeled)} precedents")
        
        return labeled
    
    # ─────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────
    
    def get_labeling_stats(
        self,
        labeled: List[LabeledPrecedent]
    ) -> Dict[str, Any]:
        """Get statistics about labeled precedents."""
        if not labeled:
            return {
                "total": 0,
                "by_role": {},
                "by_authority": {},
            }
        
        by_role = {}
        by_authority = {}
        
        for precedent in labeled:
            # Count by role
            by_role[precedent.legal_role] = by_role.get(precedent.legal_role, 0) + 1
            
            # Count by authority
            by_authority[precedent.authority_level] = by_authority.get(precedent.authority_level, 0) + 1
        
        return {
            "total": len(labeled),
            "by_role": by_role,
            "by_authority": by_authority,
        }
    
    def print_labeled_precedents(
        self,
        labeled: List[LabeledPrecedent]
    ) -> None:
        """Print labeled precedents to console."""
        print("\n" + "═" * 60)
        print("LABELED PRECEDENTS")
        print("═" * 60)
        
        if not labeled:
            print("\n  No labeled precedents")
            return
        
        stats = self.get_labeling_stats(labeled)
        
        print(f"\n📊 Statistics:")
        print(f"   Total: {stats['total']}")
        print(f"   By Role: {stats['by_role']}")
        print(f"   By Authority: {stats['by_authority']}")
        
        print(f"\n🏷️  Labeled Precedents:\n")
        
        for i, precedent in enumerate(labeled, 1):
            print(f"  {i}. {precedent.node_id}")
            print(f"     Semantic ID: {precedent.cited_semantic_id}")
            print(f"     Legal Role: {precedent.legal_role}")
            print(f"     Authority: {precedent.authority_level}")
            
            if precedent.relation_chain:
                print(f"     Relations: {' → '.join(precedent.relation_chain)}")
            
            print()
