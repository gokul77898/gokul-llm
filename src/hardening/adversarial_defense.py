"""
Adversarial Defense - Phase 8B

Detect and block adversarial query patterns.

RULE-BASED ONLY.
NO ML/LLM.
FAIL FAST.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class AdversarialPattern(Enum):
    """Types of adversarial patterns."""
    PROMPT_INJECTION = "prompt_injection"
    ROLE_PLAY = "role_play"
    INSTRUCTION_OVERRIDE = "instruction_override"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"


@dataclass
class AdversarialDetectionResult:
    """
    Result of adversarial pattern detection.
    """
    query: str
    is_adversarial: bool
    detected_patterns: List[AdversarialPattern]
    matched_rules: List[str]
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query[:100] + "..." if len(self.query) > 100 else self.query,
            "is_adversarial": self.is_adversarial,
            "detected_patterns": [p.value for p in self.detected_patterns],
            "matched_rules": self.matched_rules,
            "explanation": self.explanation,
        }


class AdversarialDefense:
    """
    Detect adversarial query patterns.
    
    RULE-BASED ONLY.
    NO ML/LLM.
    """
    
    # ─────────────────────────────────────────────
    # ADVERSARIAL PATTERNS
    # ─────────────────────────────────────────────
    
    # Prompt injection patterns
    PROMPT_INJECTION_PATTERNS = [
        r'\bignore\s+(previous|all|above|prior)\s+(instructions?|prompts?|rules?)\b',
        r'\bforget\s+(previous|all|above|prior)\s+(instructions?|prompts?|rules?)\b',
        r'\bdisregard\s+(previous|all|above|prior)\s+(instructions?|prompts?|rules?)\b',
        r'\boverride\s+(previous|all|above|prior)\s+(instructions?|prompts?|rules?)\b',
        r'\bnew\s+(instructions?|prompts?|rules?)\b',
        r'\bstart\s+over\b',
        r'\breset\s+(instructions?|prompts?|context)\b',
        r'\bignore\s+everything\b',
        r'\bforget\s+everything\b',
    ]
    
    # Role-play patterns
    ROLE_PLAY_PATTERNS = [
        r'\bpretend\s+(you\s+are|to\s+be)\b',
        r'\bact\s+as\s+(if|a|an)\b',
        r'\bimagine\s+you\s+are\b',
        r'\byou\s+are\s+now\s+(a|an)\b',
        r'\bplay\s+the\s+role\s+of\b',
        r'\broleplaying\b',
        r'\brole\s*play\b',
        r'\bsimulate\s+(being|a|an)\b',
        r'\bpretend\s+that\b',
    ]
    
    # Instruction override patterns
    INSTRUCTION_OVERRIDE_PATTERNS = [
        r'\byou\s+must\s+(now|always)\b',
        r'\byou\s+will\s+(now|always)\b',
        r'\byou\s+should\s+(now|always)\b',
        r'\bfrom\s+now\s+on\b',
        r'\bstarting\s+now\b',
        r'\bchange\s+your\s+(behavior|rules|instructions)\b',
        r'\bmodify\s+your\s+(behavior|rules|instructions)\b',
        r'\bupdate\s+your\s+(behavior|rules|instructions)\b',
        r'\bnew\s+rule\b',
        r'\bnew\s+instruction\b',
    ]
    
    # System prompt leak patterns
    SYSTEM_PROMPT_LEAK_PATTERNS = [
        r'\bshow\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?)\b',
        r'\bwhat\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions?)\b',
        r'\brepeat\s+(your|the)\s+(system\s+)?(prompt|instructions?)\b',
        r'\bprint\s+(your|the)\s+(system\s+)?(prompt|instructions?)\b',
        r'\bdisplay\s+(your|the)\s+(system\s+)?(prompt|instructions?)\b',
        r'\breveal\s+(your|the)\s+(system\s+)?(prompt|instructions?)\b',
        r'\bexpose\s+(your|the)\s+(system\s+)?(prompt|instructions?)\b',
        r'\bshare\s+(your|the)\s+(system\s+)?(prompt|instructions?)\b',
    ]
    
    def __init__(self):
        """Initialize adversarial defense."""
        logger.info("Adversarial defense initialized")
    
    # ─────────────────────────────────────────────
    # DETECTION LOGIC
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
    
    def _detect_prompt_injection(self, query: str) -> List[str]:
        """Detect prompt injection patterns."""
        return self._match_patterns(query, self.PROMPT_INJECTION_PATTERNS)
    
    def _detect_role_play(self, query: str) -> List[str]:
        """Detect role-play patterns."""
        return self._match_patterns(query, self.ROLE_PLAY_PATTERNS)
    
    def _detect_instruction_override(self, query: str) -> List[str]:
        """Detect instruction override patterns."""
        return self._match_patterns(query, self.INSTRUCTION_OVERRIDE_PATTERNS)
    
    def _detect_system_prompt_leak(self, query: str) -> List[str]:
        """Detect system prompt leak patterns."""
        return self._match_patterns(query, self.SYSTEM_PROMPT_LEAK_PATTERNS)
    
    def detect(self, query: str) -> AdversarialDetectionResult:
        """
        Detect adversarial patterns in query.
        
        RULE-BASED ONLY.
        NO ML/LLM.
        
        Args:
            query: Query string
            
        Returns:
            AdversarialDetectionResult
        """
        logger.info(f"Detecting adversarial patterns: {query[:50]}...")
        
        # Normalize query
        normalized_query = self._normalize_query(query)
        
        detected_patterns = []
        matched_rules = []
        
        # Check prompt injection
        prompt_injection_matches = self._detect_prompt_injection(normalized_query)
        if prompt_injection_matches:
            detected_patterns.append(AdversarialPattern.PROMPT_INJECTION)
            matched_rules.extend(prompt_injection_matches)
        
        # Check role-play
        role_play_matches = self._detect_role_play(normalized_query)
        if role_play_matches:
            detected_patterns.append(AdversarialPattern.ROLE_PLAY)
            matched_rules.extend(role_play_matches)
        
        # Check instruction override
        instruction_override_matches = self._detect_instruction_override(normalized_query)
        if instruction_override_matches:
            detected_patterns.append(AdversarialPattern.INSTRUCTION_OVERRIDE)
            matched_rules.extend(instruction_override_matches)
        
        # Check system prompt leak
        system_prompt_leak_matches = self._detect_system_prompt_leak(normalized_query)
        if system_prompt_leak_matches:
            detected_patterns.append(AdversarialPattern.SYSTEM_PROMPT_LEAK)
            matched_rules.extend(system_prompt_leak_matches)
        
        is_adversarial = len(detected_patterns) > 0
        
        # Generate explanation
        if is_adversarial:
            pattern_names = ", ".join([p.value for p in detected_patterns])
            explanation = (
                f"Query blocked due to detected adversarial patterns: {pattern_names}. "
                f"This system only responds to legitimate legal information queries."
            )
        else:
            explanation = "No adversarial patterns detected."
        
        result = AdversarialDetectionResult(
            query=query,
            is_adversarial=is_adversarial,
            detected_patterns=detected_patterns,
            matched_rules=matched_rules,
            explanation=explanation,
        )
        
        logger.info(
            f"Adversarial detection: {is_adversarial}, "
            f"Patterns: {len(detected_patterns)}, "
            f"Rules: {len(matched_rules)}"
        )
        
        return result
    
    # ─────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────
    
    def get_defense_stats(self) -> Dict[str, Any]:
        """Get defense statistics."""
        return {
            "pattern_types": [p.value for p in AdversarialPattern],
            "pattern_counts": {
                "prompt_injection": len(self.PROMPT_INJECTION_PATTERNS),
                "role_play": len(self.ROLE_PLAY_PATTERNS),
                "instruction_override": len(self.INSTRUCTION_OVERRIDE_PATTERNS),
                "system_prompt_leak": len(self.SYSTEM_PROMPT_LEAK_PATTERNS),
            },
        }
