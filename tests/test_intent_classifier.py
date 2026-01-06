"""
Tests for Legal Intent Classifier - Phase 7

Tests correct intent classification, refusal triggers,
deterministic behavior, and constraint compliance.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.intent.legal_intent_classifier import (
    LegalIntentClassifier,
    IntentResult,
    IntentClass,
    IntentRefusalReason,
)


class TestIntentClassifierSetup(unittest.TestCase):
    """Test classifier initialization."""
    
    def test_init(self):
        """Test initialization."""
        classifier = LegalIntentClassifier()
        self.assertIsNotNone(classifier)


class TestFactualIntent(unittest.TestCase):
    """Test factual intent classification."""
    
    def setUp(self):
        """Set up classifier."""
        self.classifier = LegalIntentClassifier()
    
    def test_section_query_factual(self):
        """Test that section queries are classified as factual or definitional (both allowed)."""
        result = self.classifier.classify("What is Section 420 IPC?")
        
        # Can be either definitional or factual, both are allowed
        self.assertIn(result.intent, [IntentClass.FACTUAL, IntentClass.DEFINITIONAL])
        self.assertTrue(result.allowed)
        self.assertIsNone(result.refusal_reason)
    
    def test_act_query_factual(self):
        """Test that act queries are classified as factual."""
        result = self.classifier.classify("Tell me about the Indian Penal Code")
        
        self.assertEqual(result.intent, IntentClass.FACTUAL)
        self.assertTrue(result.allowed)


class TestDefinitionalIntent(unittest.TestCase):
    """Test definitional intent classification."""
    
    def setUp(self):
        """Set up classifier."""
        self.classifier = LegalIntentClassifier()
    
    def test_what_is_definitional(self):
        """Test that 'what is' queries are definitional."""
        result = self.classifier.classify("What is cheating?")
        
        self.assertEqual(result.intent, IntentClass.DEFINITIONAL)
        self.assertTrue(result.allowed)
        self.assertIsNone(result.refusal_reason)
    
    def test_define_definitional(self):
        """Test that 'define' queries are definitional."""
        result = self.classifier.classify("Define theft")
        
        self.assertEqual(result.intent, IntentClass.DEFINITIONAL)
        self.assertTrue(result.allowed)
    
    def test_explain_definitional(self):
        """Test that 'explain' queries are definitional."""
        result = self.classifier.classify("Explain the concept of mens rea")
        
        self.assertEqual(result.intent, IntentClass.DEFINITIONAL)
        self.assertTrue(result.allowed)


class TestProceduralIntent(unittest.TestCase):
    """Test procedural intent classification."""
    
    def setUp(self):
        """Set up classifier."""
        self.classifier = LegalIntentClassifier()
    
    def test_how_to_file_procedural(self):
        """Test that 'how to file' queries are procedural."""
        result = self.classifier.classify("How to file a complaint?")
        
        self.assertEqual(result.intent, IntentClass.PROCEDURAL)
        self.assertTrue(result.allowed)
        self.assertIsNone(result.refusal_reason)
    
    def test_procedure_for_procedural(self):
        """Test that 'procedure for' queries are procedural."""
        result = self.classifier.classify("What is the procedure for bail?")
        
        self.assertEqual(result.intent, IntentClass.PROCEDURAL)
        self.assertTrue(result.allowed)


class TestAdvisoryIntent(unittest.TestCase):
    """Test advisory intent classification and blocking."""
    
    def setUp(self):
        """Set up classifier."""
        self.classifier = LegalIntentClassifier()
    
    def test_should_i_advisory(self):
        """Test that 'should I' queries are blocked as advisory."""
        result = self.classifier.classify("Should I plead guilty?")
        
        self.assertEqual(result.intent, IntentClass.ADVISORY)
        self.assertFalse(result.allowed)
        self.assertEqual(result.refusal_reason, IntentRefusalReason.LEGAL_ADVICE_NOT_PROVIDED)
    
    def test_advise_me_advisory(self):
        """Test that 'advise me' queries are blocked."""
        result = self.classifier.classify("Please advise me on my case")
        
        self.assertEqual(result.intent, IntentClass.ADVISORY)
        self.assertFalse(result.allowed)
    
    def test_my_case_advisory(self):
        """Test that 'my case' queries are blocked."""
        result = self.classifier.classify("What should I do in my case?")
        
        self.assertEqual(result.intent, IntentClass.ADVISORY)
        self.assertFalse(result.allowed)


class TestStrategicIntent(unittest.TestCase):
    """Test strategic intent classification and blocking."""
    
    def setUp(self):
        """Set up classifier."""
        self.classifier = LegalIntentClassifier()
    
    def test_how_to_avoid_strategic(self):
        """Test that 'how to avoid' queries are blocked as strategic."""
        result = self.classifier.classify("How to avoid conviction?")
        
        self.assertEqual(result.intent, IntentClass.STRATEGIC)
        self.assertFalse(result.allowed)
        self.assertEqual(result.refusal_reason, IntentRefusalReason.STRATEGIC_ASSISTANCE_BLOCKED)
    
    def test_beat_the_case_strategic(self):
        """Test that 'beat the case' queries are blocked."""
        result = self.classifier.classify("How can I beat the case?")
        
        self.assertEqual(result.intent, IntentClass.STRATEGIC)
        self.assertFalse(result.allowed)
    
    def test_loophole_strategic(self):
        """Test that 'loophole' queries are blocked."""
        result = self.classifier.classify("Are there any loopholes in this law?")
        
        self.assertEqual(result.intent, IntentClass.STRATEGIC)
        self.assertFalse(result.allowed)


class TestSpeculativeIntent(unittest.TestCase):
    """Test speculative intent classification and blocking."""
    
    def setUp(self):
        """Set up classifier."""
        self.classifier = LegalIntentClassifier()
    
    def test_will_i_speculative(self):
        """Test that 'will I' queries are blocked as speculative."""
        result = self.classifier.classify("Will I win this case?")
        
        self.assertEqual(result.intent, IntentClass.SPECULATIVE)
        self.assertFalse(result.allowed)
        self.assertEqual(result.refusal_reason, IntentRefusalReason.SPECULATIVE_QUERY_BLOCKED)
    
    def test_what_are_my_chances_speculative(self):
        """Test that 'what are my chances' queries are blocked."""
        result = self.classifier.classify("What are my chances of winning?")
        
        self.assertEqual(result.intent, IntentClass.SPECULATIVE)
        self.assertFalse(result.allowed)
    
    def test_predict_speculative(self):
        """Test that 'predict' queries are blocked."""
        result = self.classifier.classify("Can you predict the outcome?")
        
        self.assertEqual(result.intent, IntentClass.SPECULATIVE)
        self.assertFalse(result.allowed)


class TestIntentPriority(unittest.TestCase):
    """Test that intent classification follows strict priority order."""
    
    def setUp(self):
        """Set up classifier."""
        self.classifier = LegalIntentClassifier()
    
    def test_advisory_takes_priority_over_definitional(self):
        """Test that advisory intent takes priority over definitional."""
        # Query has both "what is" (definitional) and "should I" (advisory)
        result = self.classifier.classify("What is the law and should I plead guilty?")
        
        # Should be classified as advisory (higher priority)
        self.assertEqual(result.intent, IntentClass.ADVISORY)
        self.assertFalse(result.allowed)
    
    def test_strategic_takes_priority_over_procedural(self):
        """Test that strategic intent takes priority over procedural."""
        # Query has both "how to" (procedural) and "avoid" (strategic)
        result = self.classifier.classify("How to avoid getting caught?")
        
        # Should be classified as strategic (higher priority)
        self.assertEqual(result.intent, IntentClass.STRATEGIC)
        self.assertFalse(result.allowed)


class TestDeterministicBehavior(unittest.TestCase):
    """Test that classification is deterministic."""
    
    def setUp(self):
        """Set up classifier."""
        self.classifier = LegalIntentClassifier()
    
    def test_deterministic_classification(self):
        """Test that same query produces same classification."""
        query = "What is Section 420 IPC?"
        
        result1 = self.classifier.classify(query)
        result2 = self.classifier.classify(query)
        
        # Should have same intent
        self.assertEqual(result1.intent, result2.intent)
        self.assertEqual(result1.allowed, result2.allowed)
        self.assertEqual(result1.refusal_reason, result2.refusal_reason)
    
    def test_deterministic_patterns(self):
        """Test that matched patterns are consistent."""
        query = "Should I plead guilty?"
        
        result1 = self.classifier.classify(query)
        result2 = self.classifier.classify(query)
        
        # Should have same matched patterns
        self.assertEqual(result1.matched_patterns, result2.matched_patterns)


class TestRefusalExplanations(unittest.TestCase):
    """Test that refusal explanations are provided."""
    
    def setUp(self):
        """Set up classifier."""
        self.classifier = LegalIntentClassifier()
    
    def test_advisory_has_explanation(self):
        """Test that advisory refusal has explanation."""
        result = self.classifier.classify("Should I plead guilty?")
        
        self.assertIsNotNone(result.explanation)
        self.assertGreater(len(result.explanation), 0)
        self.assertIn("legal advice", result.explanation.lower())
    
    def test_strategic_has_explanation(self):
        """Test that strategic refusal has explanation."""
        result = self.classifier.classify("How to avoid conviction?")
        
        self.assertIsNotNone(result.explanation)
        self.assertGreater(len(result.explanation), 0)
        self.assertIn("strategic", result.explanation.lower())
    
    def test_speculative_has_explanation(self):
        """Test that speculative refusal has explanation."""
        result = self.classifier.classify("Will I win?")
        
        self.assertIsNotNone(result.explanation)
        self.assertGreater(len(result.explanation), 0)
        self.assertIn("predict", result.explanation.lower())


class TestIntentResultDataclass(unittest.TestCase):
    """Test IntentResult dataclass."""
    
    def test_to_dict(self):
        """Test serialization to dict."""
        result = IntentResult(
            query="Test query",
            intent=IntentClass.FACTUAL,
            allowed=True,
            refusal_reason=None,
            matched_patterns=["pattern1"],
            explanation="Test explanation",
        )
        
        result_dict = result.to_dict()
        
        # Check all fields present
        self.assertIn("query", result_dict)
        self.assertIn("intent", result_dict)
        self.assertIn("allowed", result_dict)
        self.assertIn("refusal_reason", result_dict)
        self.assertIn("matched_patterns", result_dict)
        self.assertIn("explanation", result_dict)


class TestNoMLNoLLM(unittest.TestCase):
    """Test that classifier doesn't use ML or LLM."""
    
    def test_classifier_has_no_ml_components(self):
        """Test that classifier has no ML components."""
        classifier = LegalIntentClassifier()
        
        # Should not have ML/LLM attributes
        self.assertFalse(hasattr(classifier, 'model'))
        self.assertFalse(hasattr(classifier, 'llm'))
        self.assertFalse(hasattr(classifier, 'embeddings'))
        self.assertFalse(hasattr(classifier, 'vectorizer'))
        self.assertFalse(hasattr(classifier, 'encoder'))


class TestCaseSensitivity(unittest.TestCase):
    """Test case insensitivity."""
    
    def setUp(self):
        """Set up classifier."""
        self.classifier = LegalIntentClassifier()
    
    def test_case_insensitive_classification(self):
        """Test that classification is case insensitive."""
        result1 = self.classifier.classify("SHOULD I PLEAD GUILTY?")
        result2 = self.classifier.classify("should i plead guilty?")
        result3 = self.classifier.classify("Should I Plead Guilty?")
        
        # All should be classified as advisory
        self.assertEqual(result1.intent, IntentClass.ADVISORY)
        self.assertEqual(result2.intent, IntentClass.ADVISORY)
        self.assertEqual(result3.intent, IntentClass.ADVISORY)


class TestUtilityMethods(unittest.TestCase):
    """Test utility methods."""
    
    def setUp(self):
        """Set up classifier."""
        self.classifier = LegalIntentClassifier()
    
    def test_get_classifier_stats(self):
        """Test stats retrieval."""
        stats = self.classifier.get_classifier_stats()
        
        self.assertIn("intent_classes", stats)
        self.assertIn("refusal_reasons", stats)
        self.assertIn("pattern_counts", stats)
        
        # Check intent classes
        self.assertEqual(len(stats["intent_classes"]), 6)
        
        # Check refusal reasons
        self.assertEqual(len(stats["refusal_reasons"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
