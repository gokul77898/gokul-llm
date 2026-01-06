"""
Tests for Precedent Labeler - Phase 5B

Tests deterministic labeling, correct role assignment,
correct ordering, and constraint compliance.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.explanation.precedent_labeler import (
    PrecedentLabeler,
    LabeledPrecedent,
)
from src.explanation.precedent_extractor import PrecedentExplanation


class TestPrecedentLabelerSetup(unittest.TestCase):
    """Test labeler initialization."""
    
    def test_init(self):
        """Test initialization."""
        labeler = PrecedentLabeler()
        self.assertIsNotNone(labeler)


class TestEmptyInput(unittest.TestCase):
    """Test empty input handling."""
    
    def setUp(self):
        """Set up labeler."""
        self.labeler = PrecedentLabeler()
    
    def test_empty_list(self):
        """Test that empty list returns empty list."""
        result = self.labeler.label([])
        self.assertEqual(len(result), 0)


class TestPrimaryStatuteLabeling(unittest.TestCase):
    """Test primary_statute labeling."""
    
    def setUp(self):
        """Set up labeler."""
        self.labeler = PrecedentLabeler()
    
    def test_act_labeled_as_primary_statute(self):
        """Test that ACT nodes are labeled as primary_statute."""
        precedent = PrecedentExplanation(
            cited_semantic_id="sem1",
            node_id="ACT::IPC",
            node_type="ACT",
            authority_level="statute",
            relation_chain=[],
            graph_path=["ACT::IPC"],
        )
        
        result = self.labeler.label([precedent])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].legal_role, "primary_statute")
    
    def test_section_labeled_as_primary_statute(self):
        """Test that SECTION nodes are labeled as primary_statute."""
        precedent = PrecedentExplanation(
            cited_semantic_id="sem1",
            node_id="SECTION::IPC::420",
            node_type="SECTION",
            authority_level="statute",
            relation_chain=[],
            graph_path=["SECTION::IPC::420"],
        )
        
        result = self.labeler.label([precedent])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].legal_role, "primary_statute")


class TestOverruledPrecedentLabeling(unittest.TestCase):
    """Test overruled_precedent labeling."""
    
    def setUp(self):
        """Set up labeler."""
        self.labeler = PrecedentLabeler()
    
    def test_overruled_case_labeled(self):
        """Test that cases with OVERRULED relation are labeled as overruled_precedent."""
        precedent = PrecedentExplanation(
            cited_semantic_id="sem1",
            node_id="CASE::CASE1",
            node_type="CASE",
            authority_level="high_court",
            relation_chain=["OVERRULES_CASE"],
            graph_path=["CASE::CASE2", "CASE::CASE1"],
        )
        
        result = self.labeler.label([precedent])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].legal_role, "overruled_precedent")


class TestJudicialInterpretationLabeling(unittest.TestCase):
    """Test judicial_interpretation labeling."""
    
    def setUp(self):
        """Set up labeler."""
        self.labeler = PrecedentLabeler()
    
    def test_interprets_section_labeled(self):
        """Test that cases with INTERPRETS_SECTION are labeled as judicial_interpretation."""
        precedent = PrecedentExplanation(
            cited_semantic_id="sem1",
            node_id="CASE::CASE1",
            node_type="CASE",
            authority_level="supreme_court",
            relation_chain=["INTERPRETS_SECTION"],
            graph_path=["CASE::CASE1", "SECTION::IPC::420"],
        )
        
        result = self.labeler.label([precedent])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].legal_role, "judicial_interpretation")


class TestAppliedPrecedentLabeling(unittest.TestCase):
    """Test applied_precedent labeling."""
    
    def setUp(self):
        """Set up labeler."""
        self.labeler = PrecedentLabeler()
    
    def test_applies_section_labeled(self):
        """Test that cases with APPLIES_SECTION are labeled as applied_precedent."""
        precedent = PrecedentExplanation(
            cited_semantic_id="sem1",
            node_id="CASE::CASE1",
            node_type="CASE",
            authority_level="high_court",
            relation_chain=["APPLIES_SECTION"],
            graph_path=["CASE::CASE1", "SECTION::IPC::420"],
        )
        
        result = self.labeler.label([precedent])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].legal_role, "applied_precedent")


class TestSupportingPrecedentLabeling(unittest.TestCase):
    """Test supporting_precedent labeling."""
    
    def setUp(self):
        """Set up labeler."""
        self.labeler = PrecedentLabeler()
    
    def test_case_default_labeled(self):
        """Test that CASE nodes without specific relations are labeled as supporting_precedent."""
        precedent = PrecedentExplanation(
            cited_semantic_id="sem1",
            node_id="CASE::CASE1",
            node_type="CASE",
            authority_level="high_court",
            relation_chain=["CITES_CASE"],  # Not a special relation
            graph_path=["CASE::CASE1"],
        )
        
        result = self.labeler.label([precedent])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].legal_role, "supporting_precedent")


class TestLabelingRulePriority(unittest.TestCase):
    """Test that labeling rules are applied in strict order."""
    
    def setUp(self):
        """Set up labeler."""
        self.labeler = PrecedentLabeler()
    
    def test_statute_takes_priority_over_relations(self):
        """Test that ACT/SECTION always get primary_statute regardless of relations."""
        # Even if a section has relations, it should still be primary_statute
        precedent = PrecedentExplanation(
            cited_semantic_id="sem1",
            node_id="SECTION::IPC::420",
            node_type="SECTION",
            authority_level="statute",
            relation_chain=["APPLIES_SECTION", "INTERPRETS_SECTION"],
            graph_path=["SECTION::IPC::420"],
        )
        
        result = self.labeler.label([precedent])
        
        self.assertEqual(result[0].legal_role, "primary_statute")
    
    def test_overruled_takes_priority_over_other_relations(self):
        """Test that OVERRULED takes priority over INTERPRETS/APPLIES."""
        precedent = PrecedentExplanation(
            cited_semantic_id="sem1",
            node_id="CASE::CASE1",
            node_type="CASE",
            authority_level="high_court",
            relation_chain=["INTERPRETS_SECTION", "OVERRULES_CASE"],
            graph_path=["CASE::CASE1"],
        )
        
        result = self.labeler.label([precedent])
        
        self.assertEqual(result[0].legal_role, "overruled_precedent")
    
    def test_interprets_takes_priority_over_applies(self):
        """Test that INTERPRETS_SECTION takes priority over APPLIES_SECTION."""
        precedent = PrecedentExplanation(
            cited_semantic_id="sem1",
            node_id="CASE::CASE1",
            node_type="CASE",
            authority_level="supreme_court",
            relation_chain=["APPLIES_SECTION", "INTERPRETS_SECTION"],
            graph_path=["CASE::CASE1"],
        )
        
        result = self.labeler.label([precedent])
        
        self.assertEqual(result[0].legal_role, "judicial_interpretation")


class TestRolePrioritySorting(unittest.TestCase):
    """Test sorting by role priority."""
    
    def setUp(self):
        """Set up labeler."""
        self.labeler = PrecedentLabeler()
    
    def test_sorted_by_role_priority(self):
        """Test that results are sorted by role priority."""
        precedents = [
            PrecedentExplanation(
                cited_semantic_id="sem1",
                node_id="CASE::CASE1",
                node_type="CASE",
                authority_level="high_court",
                relation_chain=["CITES_CASE"],
                graph_path=["CASE::CASE1"],
            ),
            PrecedentExplanation(
                cited_semantic_id="sem2",
                node_id="SECTION::IPC::420",
                node_type="SECTION",
                authority_level="statute",
                relation_chain=[],
                graph_path=["SECTION::IPC::420"],
            ),
            PrecedentExplanation(
                cited_semantic_id="sem3",
                node_id="CASE::CASE2",
                node_type="CASE",
                authority_level="supreme_court",
                relation_chain=["INTERPRETS_SECTION"],
                graph_path=["CASE::CASE2"],
            ),
        ]
        
        result = self.labeler.label(precedents)
        
        # Should be sorted: primary_statute, judicial_interpretation, supporting_precedent
        self.assertEqual(result[0].legal_role, "primary_statute")
        self.assertEqual(result[1].legal_role, "judicial_interpretation")
        self.assertEqual(result[2].legal_role, "supporting_precedent")


class TestDeterministicLabeling(unittest.TestCase):
    """Test that labeling is deterministic."""
    
    def setUp(self):
        """Set up labeler."""
        self.labeler = PrecedentLabeler()
    
    def test_deterministic_output(self):
        """Test that same input produces same output."""
        precedents = [
            PrecedentExplanation(
                cited_semantic_id="sem1",
                node_id="SECTION::IPC::420",
                node_type="SECTION",
                authority_level="statute",
                relation_chain=[],
                graph_path=["SECTION::IPC::420"],
            ),
            PrecedentExplanation(
                cited_semantic_id="sem2",
                node_id="CASE::CASE1",
                node_type="CASE",
                authority_level="high_court",
                relation_chain=["APPLIES_SECTION"],
                graph_path=["CASE::CASE1"],
            ),
        ]
        
        result1 = self.labeler.label(precedents)
        result2 = self.labeler.label(precedents)
        
        # Should produce same number of results
        self.assertEqual(len(result1), len(result2))
        
        # Should have same roles in same order
        roles1 = [p.legal_role for p in result1]
        roles2 = [p.legal_role for p in result2]
        self.assertEqual(roles1, roles2)
        
        # Should have same node_ids in same order
        node_ids1 = [p.node_id for p in result1]
        node_ids2 = [p.node_id for p in result2]
        self.assertEqual(node_ids1, node_ids2)


class TestLabeledPrecedentDataclass(unittest.TestCase):
    """Test LabeledPrecedent dataclass."""
    
    def test_to_dict(self):
        """Test serialization to dict."""
        labeled = LabeledPrecedent(
            cited_semantic_id="sem1",
            node_id="SECTION::IPC::420",
            authority_level="statute",
            legal_role="primary_statute",
            relation_chain=[],
            graph_path=["SECTION::IPC::420"],
        )
        
        result = labeled.to_dict()
        
        # Check all fields present
        self.assertIn("cited_semantic_id", result)
        self.assertIn("node_id", result)
        self.assertIn("authority_level", result)
        self.assertIn("legal_role", result)
        self.assertIn("relation_chain", result)
        self.assertIn("graph_path", result)
        
        # Check values
        self.assertEqual(result["cited_semantic_id"], "sem1")
        self.assertEqual(result["node_id"], "SECTION::IPC::420")
        self.assertEqual(result["authority_level"], "statute")
        self.assertEqual(result["legal_role"], "primary_statute")


class TestUtilityMethods(unittest.TestCase):
    """Test utility methods."""
    
    def setUp(self):
        """Set up labeler."""
        self.labeler = PrecedentLabeler()
    
    def test_get_labeling_stats_empty(self):
        """Test stats for empty list."""
        stats = self.labeler.get_labeling_stats([])
        
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["by_role"], {})
        self.assertEqual(stats["by_authority"], {})
    
    def test_get_labeling_stats(self):
        """Test stats calculation."""
        labeled = [
            LabeledPrecedent(
                cited_semantic_id="sem1",
                node_id="SECTION::IPC::420",
                authority_level="statute",
                legal_role="primary_statute",
            ),
            LabeledPrecedent(
                cited_semantic_id="sem2",
                node_id="CASE::CASE1",
                authority_level="high_court",
                legal_role="applied_precedent",
            ),
        ]
        
        stats = self.labeler.get_labeling_stats(labeled)
        
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["by_role"]["primary_statute"], 1)
        self.assertEqual(stats["by_role"]["applied_precedent"], 1)
        self.assertEqual(stats["by_authority"]["statute"], 1)
        self.assertEqual(stats["by_authority"]["high_court"], 1)


class TestNoGraphAccess(unittest.TestCase):
    """Test that labeler does not access graph."""
    
    def setUp(self):
        """Set up labeler."""
        self.labeler = PrecedentLabeler()
    
    def test_labeling_without_graph(self):
        """Test that labeling works without any graph access."""
        # Labeler should not have any graph reference
        self.assertFalse(hasattr(self.labeler, 'graph'))
        self.assertFalse(hasattr(self.labeler, 'traverser'))
        
        # Should still be able to label
        precedent = PrecedentExplanation(
            cited_semantic_id="sem1",
            node_id="SECTION::IPC::420",
            node_type="SECTION",
            authority_level="statute",
            relation_chain=[],
            graph_path=["SECTION::IPC::420"],
        )
        
        result = self.labeler.label([precedent])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].legal_role, "primary_statute")


if __name__ == "__main__":
    unittest.main(verbosity=2)
