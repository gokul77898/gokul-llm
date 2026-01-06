"""
Explanation Assembler - Phase 5C

Assembles court-style legal explanations from structured precedents.

FIXED TEMPLATES ONLY.
NO LLM USAGE.
NO GRAPH ACCESS.
NO INFERENCE BEYOND LABELS.

Uses ONLY:
- grounded_result for answer
- labeled_precedents for structure
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

from .precedent_labeler import LabeledPrecedent
from ..generation.graph_grounded_generator import GroundedAnswerResult

logger = logging.getLogger(__name__)


@dataclass
class LegalExplanation:
    """
    Assembled legal explanation.
    
    Court-style structure with verified precedents only.
    """
    answer: str
    statutory_basis: List[str] = field(default_factory=list)
    judicial_interpretations: List[str] = field(default_factory=list)
    applied_precedents: List[str] = field(default_factory=list)
    supporting_precedents: List[str] = field(default_factory=list)
    excluded_precedents: List[str] = field(default_factory=list)
    explanation_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "statutory_basis": self.statutory_basis,
            "judicial_interpretations": self.judicial_interpretations,
            "applied_precedents": self.applied_precedents,
            "supporting_precedents": self.supporting_precedents,
            "excluded_precedents": self.excluded_precedents,
            "explanation_text": self.explanation_text,
        }


class ExplanationAssembler:
    """
    Assembles court-style legal explanations.
    
    NO graph access.
    NO LLM calls.
    NO inference beyond labels.
    
    Uses ONLY labeled_precedents structure.
    """
    
    # Fixed templates for each section
    TEMPLATES = {
        "statutory_basis": "The statutory basis for this answer is derived from {items}.",
        "judicial_interpretations": "The judicial interpretation of these provisions is established in {items}.",
        "applied_precedents": "These provisions have been applied in {items}.",
        "supporting_precedents": "Additional supporting precedents include {items}.",
        "excluded_precedents": "The following precedents were excluded from consideration: {items}.",
    }
    
    # ─────────────────────────────────────────────
    # GROUPING BY LEGAL ROLE
    # ─────────────────────────────────────────────
    
    def _group_by_role(
        self,
        labeled_precedents: List[LabeledPrecedent]
    ) -> Dict[str, List[str]]:
        """
        Group precedents by legal role.
        
        Args:
            labeled_precedents: List of labeled precedents
            
        Returns:
            Dict mapping role to list of node_ids
        """
        groups = {
            "statutory_basis": [],
            "judicial_interpretations": [],
            "applied_precedents": [],
            "supporting_precedents": [],
            "excluded_precedents": [],
        }
        
        for precedent in labeled_precedents:
            node_id = precedent.node_id
            
            if precedent.legal_role == "primary_statute":
                groups["statutory_basis"].append(node_id)
            elif precedent.legal_role == "judicial_interpretation":
                groups["judicial_interpretations"].append(node_id)
            elif precedent.legal_role == "applied_precedent":
                groups["applied_precedents"].append(node_id)
            elif precedent.legal_role == "supporting_precedent":
                groups["supporting_precedents"].append(node_id)
            elif precedent.legal_role == "overruled_precedent":
                groups["excluded_precedents"].append(node_id)
        
        return groups
    
    # ─────────────────────────────────────────────
    # TEXT GENERATION (FIXED TEMPLATES)
    # ─────────────────────────────────────────────
    
    def _format_node_list(self, node_ids: List[str]) -> str:
        """
        Format list of node IDs for display.
        
        Args:
            node_ids: List of node IDs
            
        Returns:
            Formatted string
        """
        if not node_ids:
            return ""
        
        if len(node_ids) == 1:
            return node_ids[0]
        elif len(node_ids) == 2:
            return f"{node_ids[0]} and {node_ids[1]}"
        else:
            return ", ".join(node_ids[:-1]) + f", and {node_ids[-1]}"
    
    def _generate_section_text(
        self,
        section_name: str,
        node_ids: List[str]
    ) -> str:
        """
        Generate text for a section using fixed template.
        
        Args:
            section_name: Name of section
            node_ids: List of node IDs
            
        Returns:
            Formatted section text
        """
        if not node_ids:
            return ""
        
        template = self.TEMPLATES.get(section_name, "")
        if not template:
            return ""
        
        formatted_items = self._format_node_list(node_ids)
        return template.format(items=formatted_items)
    
    def _generate_explanation_text(
        self,
        groups: Dict[str, List[str]]
    ) -> str:
        """
        Generate full explanation text from groups.
        
        Uses fixed templates only.
        Omits empty sections.
        
        Args:
            groups: Dict mapping role to node_ids
            
        Returns:
            Full explanation text
        """
        sections = []
        
        # Process sections in priority order
        section_order = [
            "statutory_basis",
            "judicial_interpretations",
            "applied_precedents",
            "supporting_precedents",
            "excluded_precedents",
        ]
        
        for section_name in section_order:
            node_ids = groups.get(section_name, [])
            if node_ids:  # Only include non-empty sections
                section_text = self._generate_section_text(section_name, node_ids)
                if section_text:
                    sections.append(section_text)
        
        # Join sections with newlines
        return "\n\n".join(sections)
    
    # ─────────────────────────────────────────────
    # MAIN ASSEMBLY METHOD
    # ─────────────────────────────────────────────
    
    def assemble(
        self,
        grounded_result: GroundedAnswerResult,
        labeled_precedents: List[LabeledPrecedent]
    ) -> LegalExplanation:
        """
        Assemble legal explanation from grounded result and labeled precedents.
        
        NO graph access.
        NO LLM calls.
        NO inference beyond labels.
        
        Args:
            grounded_result: Result from graph-grounded generation
            labeled_precedents: List of labeled precedents from Phase 5B
            
        Returns:
            Assembled legal explanation
        """
        logger.info(f"Assembling explanation from {len(labeled_precedents)} precedents")
        
        # Group precedents by role
        groups = self._group_by_role(labeled_precedents)
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(groups)
        
        # Create explanation
        explanation = LegalExplanation(
            answer=grounded_result.answer,
            statutory_basis=groups["statutory_basis"],
            judicial_interpretations=groups["judicial_interpretations"],
            applied_precedents=groups["applied_precedents"],
            supporting_precedents=groups["supporting_precedents"],
            excluded_precedents=groups["excluded_precedents"],
            explanation_text=explanation_text,
        )
        
        logger.info(f"Assembled explanation with {len(explanation_text)} chars")
        
        return explanation
    
    # ─────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────
    
    def get_explanation_stats(
        self,
        explanation: LegalExplanation
    ) -> Dict[str, Any]:
        """Get statistics about explanation."""
        return {
            "answer_length": len(explanation.answer),
            "statutory_basis_count": len(explanation.statutory_basis),
            "judicial_interpretations_count": len(explanation.judicial_interpretations),
            "applied_precedents_count": len(explanation.applied_precedents),
            "supporting_precedents_count": len(explanation.supporting_precedents),
            "excluded_precedents_count": len(explanation.excluded_precedents),
            "explanation_text_length": len(explanation.explanation_text),
        }
    
    def print_explanation(
        self,
        explanation: LegalExplanation
    ) -> None:
        """Print explanation to console."""
        print("\n" + "═" * 60)
        print("LEGAL EXPLANATION")
        print("═" * 60)
        
        print(f"\n📝 Answer:")
        print(f"   {explanation.answer[:200]}...")
        
        stats = self.get_explanation_stats(explanation)
        
        print(f"\n📊 Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        if explanation.statutory_basis:
            print(f"\n⚖️  Statutory Basis ({len(explanation.statutory_basis)}):")
            for node_id in explanation.statutory_basis:
                print(f"   • {node_id}")
        
        if explanation.judicial_interpretations:
            print(f"\n👨‍⚖️  Judicial Interpretations ({len(explanation.judicial_interpretations)}):")
            for node_id in explanation.judicial_interpretations:
                print(f"   • {node_id}")
        
        if explanation.applied_precedents:
            print(f"\n📚 Applied Precedents ({len(explanation.applied_precedents)}):")
            for node_id in explanation.applied_precedents:
                print(f"   • {node_id}")
        
        if explanation.supporting_precedents:
            print(f"\n🔗 Supporting Precedents ({len(explanation.supporting_precedents)}):")
            for node_id in explanation.supporting_precedents:
                print(f"   • {node_id}")
        
        if explanation.excluded_precedents:
            print(f"\n❌ Excluded Precedents ({len(explanation.excluded_precedents)}):")
            for node_id in explanation.excluded_precedents:
                print(f"   • {node_id}")
        
        if explanation.explanation_text:
            print(f"\n📄 Explanation Text:")
            print(f"\n{explanation.explanation_text}")
        
        print("\n" + "═" * 60)
