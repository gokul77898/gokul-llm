"""
Tests for Explanation Assembler - Phase 5C

Tests correct grouping by role, section ordering,
deterministic text generation, and constraint compliance.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.explanation.explanation_assembler import (
    ExplanationAssembler,
    LegalExplanation,
)
from src.explanation.precedent_labeler import LabeledPrecedent
from src.generation.graph_grounded_generator import GroundedAnswerResult


class TestExplanationAssemblerSetup(unittest.TestCase):
    """Test assembler initialization."""
    
    def test_init(self):
        """Test initialization."""
        assembler = ExplanationAssembler()
        self.assertIsNotNone(assembler)


class TestEmptyInput(unittest.TestCase):
    """Test empty input handling."""
    
    def setUp(self):
        """Set up assembler."""
        self.assembler = ExplanationAssembler()
    
    def test_empty_precedents(self):
        """Test that empty precedents list is handled safely."""
        grounded_result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=[],
            graph_paths_used=[],
        )
        
        explanation = self.assembler.assemble(grounded_result, [])
        
        # Should have answer but empty sections
        self.assertEqual(explanation.answer, "Test answer")
        self.assertEqual(len(explanation.statutory_basis), 0)
        self.assertEqual(len(explanation.judicial_interpretations), 0)
        self.assertEqual(len(explanation.applied_precedents), 0)
        self.assertEqual(len(explanation.supporting_precedents), 0)
        self.assertEqual(len(explanation.excluded_precedents), 0)
        self.assertEqual(explanation.explanation_text, "")


class TestGroupingByRole(unittest.TestCase):
    """Test correct grouping by legal role."""
    
    def setUp(self):
        """Set up assembler."""
        self.assembler = ExplanationAssembler()
    
    def test_primary_statute_grouped_correctly(self):
        """Test that primary_statute is grouped into statutory_basis."""
        precedents = [
            LabeledPrecedent(
                cited_semantic_id="sem1",
                node_id="SECTION::IPC::420",
                authority_level="statute",
                legal_role="primary_statute",
            ),
        ]
        
        grounded_result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=["sem1"],
            graph_paths_used=[],
        )
        
        explanation = self.assembler.assemble(grounded_result, precedents)
        
        self.assertEqual(len(explanation.statutory_basis), 1)
        self.assertIn("SECTION::IPC::420", explanation.statutory_basis)
    
    def test_judicial_interpretation_grouped_correctly(self):
        """Test that judicial_interpretation is grouped correctly."""
        precedents = [
            LabeledPrecedent(
                cited_semantic_id="sem1",
                node_id="CASE::CASE1",
                authority_level="supreme_court",
                legal_role="judicial_interpretation",
            ),
        ]
        
        grounded_result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=["sem1"],
            graph_paths_used=[],
        )
        
        explanation = self.assembler.assemble(grounded_result, precedents)
        
        self.assertEqual(len(explanation.judicial_interpretations), 1)
        self.assertIn("CASE::CASE1", explanation.judicial_interpretations)
    
    def test_applied_precedent_grouped_correctly(self):
        """Test that applied_precedent is grouped correctly."""
        precedents = [
            LabeledPrecedent(
                cited_semantic_id="sem1",
                node_id="CASE::CASE1",
                authority_level="high_court",
                legal_role="applied_precedent",
            ),
        ]
        
        grounded_result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=["sem1"],
            graph_paths_used=[],
        )
        
        explanation = self.assembler.assemble(grounded_result, precedents)
        
        self.assertEqual(len(explanation.applied_precedents), 1)
        self.assertIn("CASE::CASE1", explanation.applied_precedents)
    
    def test_supporting_precedent_grouped_correctly(self):
        """Test that supporting_precedent is grouped correctly."""
        precedents = [
            LabeledPrecedent(
                cited_semantic_id="sem1",
                node_id="CASE::CASE1",
                authority_level="high_court",
                legal_role="supporting_precedent",
            ),
        ]
        
        grounded_result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=["sem1"],
            graph_paths_used=[],
        )
        
        explanation = self.assembler.assemble(grounded_result, precedents)
        
        self.assertEqual(len(explanation.supporting_precedents), 1)
        self.assertIn("CASE::CASE1", explanation.supporting_precedents)
    
    def test_overruled_precedent_grouped_correctly(self):
        """Test that overruled_precedent is grouped into excluded_precedents."""
        precedents = [
            LabeledPrecedent(
                cited_semantic_id="sem1",
                node_id="CASE::CASE1",
                authority_level="high_court",
                legal_role="overruled_precedent",
            ),
        ]
        
        grounded_result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=["sem1"],
            graph_paths_used=[],
        )
        
        explanation = self.assembler.assemble(grounded_result, precedents)
        
        self.assertEqual(len(explanation.excluded_precedents), 1)
        self.assertIn("CASE::CASE1", explanation.excluded_precedents)


class TestMultipleGrouping(unittest.TestCase):
    """Test grouping with multiple precedents."""
    
    def setUp(self):
        """Set up assembler."""
        self.assembler = ExplanationAssembler()
    
    def test_multiple_precedents_grouped_correctly(self):
        """Test that multiple precedents are grouped into correct sections."""
        precedents = [
            LabeledPrecedent(
                cited_semantic_id="sem1",
                node_id="SECTION::IPC::420",
                authority_level="statute",
                legal_role="primary_statute",
            ),
            LabeledPrecedent(
                cited_semantic_id="sem2",
                node_id="CASE::CASE1",
                authority_level="supreme_court",
                legal_role="judicial_interpretation",
            ),
            LabeledPrecedent(
                cited_semantic_id="sem3",
                node_id="CASE::CASE2",
                authority_level="high_court",
                legal_role="applied_precedent",
            ),
        ]
        
        grounded_result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=["sem1", "sem2", "sem3"],
            graph_paths_used=[],
        )
        
        explanation = self.assembler.assemble(grounded_result, precedents)
        
        self.assertEqual(len(explanation.statutory_basis), 1)
        self.assertEqual(len(explanation.judicial_interpretations), 1)
        self.assertEqual(len(explanation.applied_precedents), 1)


class TestSectionOrdering(unittest.TestCase):
    """Test that sections are ordered correctly in explanation text."""
    
    def setUp(self):
        """Set up assembler."""
        self.assembler = ExplanationAssembler()
    
    def test_section_order_in_text(self):
        """Test that sections appear in correct order in explanation text."""
        precedents = [
            LabeledPrecedent(
                cited_semantic_id="sem1",
                node_id="SECTION::IPC::420",
                authority_level="statute",
                legal_role="primary_statute",
            ),
            LabeledPrecedent(
                cited_semantic_id="sem2",
                node_id="CASE::CASE1",
                authority_level="supreme_court",
                legal_role="judicial_interpretation",
            ),
        ]
        
        grounded_result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=["sem1", "sem2"],
            graph_paths_used=[],
        )
        
        explanation = self.assembler.assemble(grounded_result, precedents)
        
        # Check that statutory_basis appears before judicial_interpretations
        text = explanation.explanation_text
        statutory_pos = text.find("statutory basis")
        judicial_pos = text.find("judicial interpretation")
        
        self.assertGreater(statutory_pos, -1)
        self.assertGreater(judicial_pos, -1)
        self.assertLess(statutory_pos, judicial_pos)


class TestEmptySectionsOmitted(unittest.TestCase):
    """Test that empty sections are omitted from explanation text."""
    
    def setUp(self):
        """Set up assembler."""
        self.assembler = ExplanationAssembler()
    
    def test_empty_sections_not_in_text(self):
        """Test that empty sections are not included in explanation text."""
        precedents = [
            LabeledPrecedent(
                cited_semantic_id="sem1",
                node_id="SECTION::IPC::420",
                authority_level="statute",
                legal_role="primary_statute",
            ),
        ]
        
        grounded_result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=["sem1"],
            graph_paths_used=[],
        )
        
        explanation = self.assembler.assemble(grounded_result, precedents)
        
        # Should have statutory_basis but not other sections
        self.assertIn("statutory basis", explanation.explanation_text.lower())
        self.assertNotIn("judicial interpretation", explanation.explanation_text.lower())
        self.assertNotIn("applied", explanation.explanation_text.lower())


class TestDeterministicText(unittest.TestCase):
    """Test that explanation text is deterministic."""
    
    def setUp(self):
        """Set up assembler."""
        self.assembler = ExplanationAssembler()
    
    def test_deterministic_text_generation(self):
        """Test that same input produces same text."""
        precedents = [
            LabeledPrecedent(
                cited_semantic_id="sem1",
                node_id="SECTION::IPC::420",
                authority_level="statute",
                legal_role="primary_statute",
            ),
            LabeledPrecedent(
                cited_semantic_id="sem2",
                node_id="CASE::CASE1",
                authority_level="supreme_court",
                legal_role="judicial_interpretation",
            ),
        ]
        
        grounded_result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=["sem1", "sem2"],
            graph_paths_used=[],
        )
        
        explanation1 = self.assembler.assemble(grounded_result, precedents)
        explanation2 = self.assembler.assemble(grounded_result, precedents)
        
        # Should produce identical text
        self.assertEqual(explanation1.explanation_text, explanation2.explanation_text)


class TestNoHallucinatedReferences(unittest.TestCase):
    """Test that no hallucinated references appear."""
    
    def setUp(self):
        """Set up assembler."""
        self.assembler = ExplanationAssembler()
    
    def test_only_provided_nodes_in_text(self):
        """Test that only provided node_ids appear in text."""
        precedents = [
            LabeledPrecedent(
                cited_semantic_id="sem1",
                node_id="SECTION::IPC::420",
                authority_level="statute",
                legal_role="primary_statute",
            ),
        ]
        
        grounded_result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=["sem1"],
            graph_paths_used=[],
        )
        
        explanation = self.assembler.assemble(grounded_result, precedents)
        
        # Should contain the provided node_id
        self.assertIn("SECTION::IPC::420", explanation.explanation_text)
        
        # Should not contain any other node_ids
        self.assertNotIn("SECTION::IPC::421", explanation.explanation_text)
        self.assertNotIn("CASE::FAKE", explanation.explanation_text)


class TestNodeListFormatting(unittest.TestCase):
    """Test node list formatting."""
    
    def setUp(self):
        """Set up assembler."""
        self.assembler = ExplanationAssembler()
    
    def test_single_node_formatting(self):
        """Test formatting of single node."""
        result = self.assembler._format_node_list(["NODE1"])
        self.assertEqual(result, "NODE1")
    
    def test_two_nodes_formatting(self):
        """Test formatting of two nodes."""
        result = self.assembler._format_node_list(["NODE1", "NODE2"])
        self.assertEqual(result, "NODE1 and NODE2")
    
    def test_multiple_nodes_formatting(self):
        """Test formatting of multiple nodes."""
        result = self.assembler._format_node_list(["NODE1", "NODE2", "NODE3"])
        self.assertEqual(result, "NODE1, NODE2, and NODE3")


class TestLegalExplanationDataclass(unittest.TestCase):
    """Test LegalExplanation dataclass."""
    
    def test_to_dict(self):
        """Test serialization to dict."""
        explanation = LegalExplanation(
            answer="Test answer",
            statutory_basis=["SECTION::IPC::420"],
            judicial_interpretations=["CASE::CASE1"],
            applied_precedents=[],
            supporting_precedents=[],
            excluded_precedents=[],
            explanation_text="Test text",
        )
        
        result = explanation.to_dict()
        
        # Check all fields present
        self.assertIn("answer", result)
        self.assertIn("statutory_basis", result)
        self.assertIn("judicial_interpretations", result)
        self.assertIn("applied_precedents", result)
        self.assertIn("supporting_precedents", result)
        self.assertIn("excluded_precedents", result)
        self.assertIn("explanation_text", result)
        
        # Check values
        self.assertEqual(result["answer"], "Test answer")
        self.assertEqual(result["statutory_basis"], ["SECTION::IPC::420"])
        self.assertEqual(result["judicial_interpretations"], ["CASE::CASE1"])


class TestUtilityMethods(unittest.TestCase):
    """Test utility methods."""
    
    def setUp(self):
        """Set up assembler."""
        self.assembler = ExplanationAssembler()
    
    def test_get_explanation_stats(self):
        """Test stats calculation."""
        explanation = LegalExplanation(
            answer="Test answer",
            statutory_basis=["SECTION::IPC::420"],
            judicial_interpretations=["CASE::CASE1"],
            applied_precedents=[],
            supporting_precedents=[],
            excluded_precedents=[],
            explanation_text="Test text",
        )
        
        stats = self.assembler.get_explanation_stats(explanation)
        
        self.assertEqual(stats["statutory_basis_count"], 1)
        self.assertEqual(stats["judicial_interpretations_count"], 1)
        self.assertEqual(stats["applied_precedents_count"], 0)
        self.assertGreater(stats["answer_length"], 0)
        self.assertGreater(stats["explanation_text_length"], 0)


class TestNoGraphAccess(unittest.TestCase):
    """Test that assembler does not access graph."""
    
    def setUp(self):
        """Set up assembler."""
        self.assembler = ExplanationAssembler()
    
    def test_assembly_without_graph(self):
        """Test that assembly works without any graph access."""
        # Assembler should not have any graph reference
        self.assertFalse(hasattr(self.assembler, 'graph'))
        self.assertFalse(hasattr(self.assembler, 'traverser'))
        
        # Should still be able to assemble
        precedents = [
            LabeledPrecedent(
                cited_semantic_id="sem1",
                node_id="SECTION::IPC::420",
                authority_level="statute",
                legal_role="primary_statute",
            ),
        ]
        
        grounded_result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=["sem1"],
            graph_paths_used=[],
        )
        
        explanation = self.assembler.assemble(grounded_result, precedents)
        
        self.assertIsNotNone(explanation)
        self.assertEqual(len(explanation.statutory_basis), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
