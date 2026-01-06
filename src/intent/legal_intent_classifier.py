"""
Legal Intent Classifier - Phase 7

Rule-based classification of legal query intent.

RULE-BASED ONLY.
NO ML.
NO LLM.
NO EMBEDDINGS.
DETERMINISTIC.
EXPLAINABLE.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class IntentClass(Enum):
    """
    Legal query intent classes.
    """
    FACTUAL = "factual"  # "What is Section 420?"
    DEFINITIONAL = "definitional"  # "Define cheating"
    PROCEDURAL = "procedural"  # "How to file a case?"
    ADVISORY = "advisory"  # "Should I plead guilty?"
    STRATEGIC = "strategic"  # "How to avoid conviction?"
    SPECULATIVE = "speculative"  # "Will I win this case?"


class IntentRefusalReason(Enum):
    """
    Reasons for refusing queries based on intent.
    """
    LEGAL_ADVICE_NOT_PROVIDED = "legal_advice_not_provided"
    STRATEGIC_ASSISTANCE_BLOCKED = "strategic_assistance_blocked"
    SPECULATIVE_QUERY_BLOCKED = "speculative_query_blocked"


@dataclass
class IntentResult:
    """
    Result of intent classification.
    """
    query: str
    intent: IntentClass
    allowed: bool
    refusal_reason: Optional[IntentRefusalReason] = None
    matched_patterns: List[str] = None
    explanation: str = ""
    
    def __post_init__(self):
        if self.matched_patterns is None:
            self.matched_patterns = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent.value,
            "allowed": self.allowed,
            "refusal_reason": self.refusal_reason.value if self.refusal_reason else None,
            "matched_patterns": self.matched_patterns,
            "explanation": self.explanation,
        }


class LegalIntentClassifier:
    """
    Rule-based legal query intent classifier.
    
    NO ML.
    NO LLM.
    NO EMBEDDINGS.
    
    Uses keyword patterns only.
    """
    
    # ─────────────────────────────────────────────
    # INTENT PATTERNS (STRICT ORDER)
    # ─────────────────────────────────────────────
    
    # Advisory patterns (BLOCKED)
    ADVISORY_PATTERNS = [
        r'\bshould i\b',
        r'\bshould we\b',
        r'\bwhat should\b',
        r'\badvise me\b',
        r'\badvise us\b',
        r'\brecommend\b',
        r'\bsuggestion\b',
        r'\bmy case\b',
        r'\bour case\b',
        r'\bplead guilty\b',
        r'\bplead not guilty\b',
        r'\bsettle\b',
        r'\bgo to trial\b',
    ]
    
    # Strategic patterns (BLOCKED)
    STRATEGIC_PATTERNS = [
        r'\bhow to avoid\b',
        r'\bhow to evade\b',
        r'\bhow to escape\b',
        r'\bhow to get away\b',
        r'\bbeat the charge\b',
        r'\bbeat the case\b',
        r'\bwin the case\b',
        r'\bdefeat the prosecution\b',
        r'\bloop\s*holes?\b',
        r'\bexploit\b',
        r'\btrick\b',
        r'\bmanipulate\b',
    ]
    
    # Speculative patterns (BLOCKED)
    SPECULATIVE_PATTERNS = [
        r'\bwill i\b',
        r'\bwill we\b',
        r'\bwill the\b',
        r'\bwhat will happen\b',
        r'\bwhat are my chances\b',
        r'\bwhat are the chances\b',
        r'\bprobability\b',
        r'\blikely to\b',
        r'\bpredict\b',
        r'\bforecast\b',
    ]
    
    # Procedural patterns (ALLOWED)
    PROCEDURAL_PATTERNS = [
        r'\bhow to file\b',
        r'\bhow to register\b',
        r'\bhow to submit\b',
        r'\bprocedure for\b',
        r'\bprocess for\b',
        r'\bsteps to\b',
        r'\brequirements for\b',
        r'\bdocuments needed\b',
        r'\btime limit\b',
        r'\bdeadline\b',
        r'\bjurisdiction\b',
    ]
    
    # Definitional patterns (ALLOWED)
    DEFINITIONAL_PATTERNS = [
        r'\bdefine\b',
        r'\bdefinition of\b',
        r'\bwhat is\b',
        r'\bwhat are\b',
        r'\bwhat does\b',
        r'\bmeaning of\b',
        r'\bexplain\b',
        r'\bdescribe\b',
    ]
    
    # Factual patterns (ALLOWED) - default if no other match
    FACTUAL_PATTERNS = [
        r'\bsection\s+\d+\b',
        r'\barticle\s+\d+\b',
        r'\bact\b',
        r'\bcode\b',
        r'\bstatute\b',
        r'\bprovision\b',
        r'\bclause\b',
        r'\bcase\s+law\b',
        r'\bprecedent\b',
        r'\bjudgment\b',
    ]
    
    # ─────────────────────────────────────────────
    # CLASSIFICATION LOGIC
    # ─────────────────────────────────────────────
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for pattern matching.
        
        Args:
            query: Raw query
            
        Returns:
            Normalized query (lowercase, trimmed)
        """
        return query.lower().strip()
    
    def _match_patterns(
        self,
        query: str,
        patterns: List[str]
    ) -> List[str]:
        """
        Match query against patterns.
        
        Args:
            query: Normalized query
            patterns: List of regex patterns
            
        Returns:
            List of matched patterns
        """
        matched = []
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                matched.append(pattern)
        return matched
    
    def _classify_intent(
        self,
        query: str
    ) -> tuple[IntentClass, List[str]]:
        """
        Classify query intent using pattern matching.
        
        STRICT ORDER:
        1. Advisory (BLOCKED)
        2. Strategic (BLOCKED)
        3. Speculative (BLOCKED)
        4. Procedural (ALLOWED)
        5. Definitional (ALLOWED)
        6. Factual (ALLOWED - default)
        
        Args:
            query: Normalized query
            
        Returns:
            Tuple of (intent, matched_patterns)
        """
        # Check advisory (BLOCKED)
        matched = self._match_patterns(query, self.ADVISORY_PATTERNS)
        if matched:
            return IntentClass.ADVISORY, matched
        
        # Check strategic (BLOCKED)
        matched = self._match_patterns(query, self.STRATEGIC_PATTERNS)
        if matched:
            return IntentClass.STRATEGIC, matched
        
        # Check speculative (BLOCKED)
        matched = self._match_patterns(query, self.SPECULATIVE_PATTERNS)
        if matched:
            return IntentClass.SPECULATIVE, matched
        
        # Check procedural (ALLOWED)
        matched = self._match_patterns(query, self.PROCEDURAL_PATTERNS)
        if matched:
            return IntentClass.PROCEDURAL, matched
        
        # Check definitional (ALLOWED)
        matched = self._match_patterns(query, self.DEFINITIONAL_PATTERNS)
        if matched:
            return IntentClass.DEFINITIONAL, matched
        
        # Check factual (ALLOWED)
        matched = self._match_patterns(query, self.FACTUAL_PATTERNS)
        if matched:
            return IntentClass.FACTUAL, matched
        
        # Default to factual if no patterns match
        return IntentClass.FACTUAL, []
    
    def _determine_refusal(
        self,
        intent: IntentClass
    ) -> Optional[IntentRefusalReason]:
        """
        Determine if intent should be refused.
        
        Args:
            intent: Classified intent
            
        Returns:
            Refusal reason or None if allowed
        """
        if intent == IntentClass.ADVISORY:
            return IntentRefusalReason.LEGAL_ADVICE_NOT_PROVIDED
        elif intent == IntentClass.STRATEGIC:
            return IntentRefusalReason.STRATEGIC_ASSISTANCE_BLOCKED
        elif intent == IntentClass.SPECULATIVE:
            return IntentRefusalReason.SPECULATIVE_QUERY_BLOCKED
        else:
            return None
    
    def _generate_explanation(
        self,
        intent: IntentClass,
        allowed: bool,
        refusal_reason: Optional[IntentRefusalReason]
    ) -> str:
        """
        Generate explanation for classification.
        
        Args:
            intent: Classified intent
            allowed: Whether query is allowed
            refusal_reason: Refusal reason if blocked
            
        Returns:
            Explanation string
        """
        if allowed:
            return f"Query classified as {intent.value} and is allowed to proceed."
        else:
            reason_map = {
                IntentRefusalReason.LEGAL_ADVICE_NOT_PROVIDED: (
                    "This system provides legal information only, not legal advice. "
                    "Please consult a qualified attorney for advice on your specific situation."
                ),
                IntentRefusalReason.STRATEGIC_ASSISTANCE_BLOCKED: (
                    "This system cannot provide strategic legal assistance or help "
                    "circumvent legal processes. Please consult a qualified attorney."
                ),
                IntentRefusalReason.SPECULATIVE_QUERY_BLOCKED: (
                    "This system cannot predict case outcomes or provide speculative "
                    "analysis. Please consult a qualified attorney for case assessment."
                ),
            }
            return reason_map.get(
                refusal_reason,
                "Query blocked due to safety constraints."
            )
    
    # ─────────────────────────────────────────────
    # MAIN CLASSIFICATION METHOD
    # ─────────────────────────────────────────────
    
    def classify(self, query: str) -> IntentResult:
        """
        Classify legal query intent.
        
        RULE-BASED ONLY.
        NO ML.
        NO LLM.
        NO EMBEDDINGS.
        
        Args:
            query: Legal query
            
        Returns:
            IntentResult with classification and refusal info
        """
        logger.info(f"Classifying query intent: {query[:50]}...")
        
        # Normalize query
        normalized_query = self._normalize_query(query)
        
        # Classify intent
        intent, matched_patterns = self._classify_intent(normalized_query)
        
        # Determine if allowed
        refusal_reason = self._determine_refusal(intent)
        allowed = refusal_reason is None
        
        # Generate explanation
        explanation = self._generate_explanation(intent, allowed, refusal_reason)
        
        result = IntentResult(
            query=query,
            intent=intent,
            allowed=allowed,
            refusal_reason=refusal_reason,
            matched_patterns=matched_patterns,
            explanation=explanation,
        )
        
        logger.info(
            f"Intent: {intent.value}, Allowed: {allowed}, "
            f"Patterns: {len(matched_patterns)}"
        )
        
        return result
    
    # ─────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────
    
    def get_classifier_stats(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        return {
            "intent_classes": [intent.value for intent in IntentClass],
            "refusal_reasons": [reason.value for reason in IntentRefusalReason],
            "pattern_counts": {
                "advisory": len(self.ADVISORY_PATTERNS),
                "strategic": len(self.STRATEGIC_PATTERNS),
                "speculative": len(self.SPECULATIVE_PATTERNS),
                "procedural": len(self.PROCEDURAL_PATTERNS),
                "definitional": len(self.DEFINITIONAL_PATTERNS),
                "factual": len(self.FACTUAL_PATTERNS),
            },
        }
    
    def print_classification(self, result: IntentResult) -> None:
        """Print classification result."""
        print("\n" + "═" * 60)
        print("INTENT CLASSIFICATION")
        print("═" * 60)
        
        print(f"\n📝 Query: {result.query}")
        print(f"🎯 Intent: {result.intent.value}")
        print(f"✅ Allowed: {'Yes' if result.allowed else 'No'}")
        
        if result.refusal_reason:
            print(f"❌ Refusal: {result.refusal_reason.value}")
        
        if result.matched_patterns:
            print(f"\n📋 Matched Patterns ({len(result.matched_patterns)}):")
            for pattern in result.matched_patterns[:3]:
                print(f"   • {pattern}")
            if len(result.matched_patterns) > 3:
                print(f"   ... and {len(result.matched_patterns) - 3} more")
        
        print(f"\n💬 Explanation:")
        print(f"   {result.explanation}")
        
        print("\n" + "═" * 60)
