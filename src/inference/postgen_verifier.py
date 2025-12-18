"""
Post-Generation Verifier

Phase R6b: Post-Generation Verification

Verifies decoder output against evidence AFTER generation
and overrides any unsafe output with a refusal.

This is the FINAL safety net.

VERIFICATION CHECKS (in order):
1. Citation Presence - output must contain valid citations
2. Section Consistency - sections must exist in evidence
3. Act/Statute Consistency - acts must exist in evidence
4. Court Consistency - courts must exist in evidence
5. Citation Integrity - citations must be sequential and valid

NO ML used. Deterministic verification only.
Model output is NEVER trusted blindly.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from pathlib import Path


class VerificationStatus(str, Enum):
    """Verification result status."""
    PASS = "pass"
    REFUSE = "refuse"


@dataclass
class VerificationResult:
    """
    Result of post-generation verification.
    
    Attributes:
        status: pass or refuse
        refusal_reason: Machine-readable reason if refused
        refusal_message: User-safe message if refused
        extracted_sections: Sections found in output
        extracted_acts: Acts found in output
        extracted_courts: Courts found in output
        extracted_citations: Citation markers found in output
    """
    status: VerificationStatus
    refusal_reason: Optional[str] = None
    refusal_message: Optional[str] = None
    extracted_sections: List[str] = field(default_factory=list)
    extracted_acts: List[str] = field(default_factory=list)
    extracted_courts: List[str] = field(default_factory=list)
    extracted_citations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "refusal_reason": self.refusal_reason,
            "refusal_message": self.refusal_message,
            "extracted_sections": self.extracted_sections,
            "extracted_acts": self.extracted_acts,
            "extracted_courts": self.extracted_courts,
            "extracted_citations": self.extracted_citations,
        }


# User-safe refusal message
DEFAULT_REFUSAL_MESSAGE = "The system cannot provide an answer due to insufficient or invalid legal grounding."


class PostGenerationVerifier:
    """
    Verifies decoder output against evidence.
    
    Runs AFTER decoder generation.
    Any violation → server-side refusal.
    Decoder is never re-prompted.
    """
    
    # Known acts/statutes
    KNOWN_ACTS = {
        "ipc": ["ipc", "indian penal code", "penal code"],
        "crpc": ["crpc", "cr.p.c", "criminal procedure code", "code of criminal procedure"],
        "cpc": ["cpc", "c.p.c", "civil procedure code", "code of civil procedure"],
        "iea": ["iea", "indian evidence act", "evidence act"],
        "it_act": ["it act", "information technology act"],
        "constitution": ["constitution", "article"],
        "contract_act": ["contract act", "indian contract act"],
        "companies_act": ["companies act"],
        "gst": ["gst", "goods and services tax"],
        "income_tax": ["income tax act", "it act 1961"],
    }
    
    # Known courts
    KNOWN_COURTS = {
        "supreme_court": ["supreme court", "sc", "hon'ble supreme court", "apex court"],
        "high_court": ["high court", "hc", "hon'ble high court"],
        "district_court": ["district court", "sessions court", "district judge"],
        "magistrate": ["magistrate", "jmfc", "cjm", "acjm"],
        "tribunal": ["tribunal", "nclt", "nclat", "itat", "sat"],
    }
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize verifier.
        
        Args:
            log_dir: Directory for verification logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "rag_postgen.jsonl"
    
    def verify(
        self,
        output_text: str,
        evidence_chunks: List[Dict[str, Any]],
        evidence_context: str = "",
    ) -> VerificationResult:
        """
        Verify decoder output against evidence.
        
        Args:
            output_text: Decoder output text
            evidence_chunks: List of evidence chunk dictionaries
                Expected keys: chunk_id, section, act, court
            evidence_context: Full evidence context string
            
        Returns:
            VerificationResult with pass/refuse status
        """
        # Extract information from output
        extracted_citations = self._extract_citations(output_text)
        extracted_sections = self._extract_sections(output_text)
        extracted_acts = self._extract_acts(output_text)
        extracted_courts = self._extract_courts(output_text)
        
        # Build evidence sets
        evidence_sections = self._get_evidence_sections(evidence_chunks, evidence_context)
        evidence_acts = self._get_evidence_acts(evidence_chunks, evidence_context)
        evidence_courts = self._get_evidence_courts(evidence_chunks, evidence_context)
        max_citation = len(evidence_chunks)
        
        # ─────────────────────────────────────────────
        # CHECK 1: Citation Presence
        # ─────────────────────────────────────────────
        if not extracted_citations:
            return VerificationResult(
                status=VerificationStatus.REFUSE,
                refusal_reason="no_citation_in_output",
                refusal_message=DEFAULT_REFUSAL_MESSAGE,
                extracted_sections=extracted_sections,
                extracted_acts=extracted_acts,
                extracted_courts=extracted_courts,
                extracted_citations=extracted_citations,
            )
        
        # ─────────────────────────────────────────────
        # CHECK 2: Section Consistency
        # ─────────────────────────────────────────────
        for section in extracted_sections:
            if section not in evidence_sections:
                return VerificationResult(
                    status=VerificationStatus.REFUSE,
                    refusal_reason=f"hallucinated_section:{section}",
                    refusal_message=DEFAULT_REFUSAL_MESSAGE,
                    extracted_sections=extracted_sections,
                    extracted_acts=extracted_acts,
                    extracted_courts=extracted_courts,
                    extracted_citations=extracted_citations,
                )
        
        # ─────────────────────────────────────────────
        # CHECK 3: Act/Statute Consistency
        # ─────────────────────────────────────────────
        for act in extracted_acts:
            if act not in evidence_acts:
                return VerificationResult(
                    status=VerificationStatus.REFUSE,
                    refusal_reason=f"hallucinated_act:{act}",
                    refusal_message=DEFAULT_REFUSAL_MESSAGE,
                    extracted_sections=extracted_sections,
                    extracted_acts=extracted_acts,
                    extracted_courts=extracted_courts,
                    extracted_citations=extracted_citations,
                )
        
        # ─────────────────────────────────────────────
        # CHECK 4: Court Consistency
        # ─────────────────────────────────────────────
        for court in extracted_courts:
            if court not in evidence_courts:
                return VerificationResult(
                    status=VerificationStatus.REFUSE,
                    refusal_reason=f"hallucinated_court:{court}",
                    refusal_message=DEFAULT_REFUSAL_MESSAGE,
                    extracted_sections=extracted_sections,
                    extracted_acts=extracted_acts,
                    extracted_courts=extracted_courts,
                    extracted_citations=extracted_citations,
                )
        
        # ─────────────────────────────────────────────
        # CHECK 5: Citation Integrity
        # ─────────────────────────────────────────────
        for cit in extracted_citations:
            try:
                cit_num = int(cit)
                if cit_num < 1 or cit_num > max_citation:
                    return VerificationResult(
                        status=VerificationStatus.REFUSE,
                        refusal_reason=f"invalid_citation:[{cit}]",
                        refusal_message=DEFAULT_REFUSAL_MESSAGE,
                        extracted_sections=extracted_sections,
                        extracted_acts=extracted_acts,
                        extracted_courts=extracted_courts,
                        extracted_citations=extracted_citations,
                    )
            except ValueError:
                # Non-numeric citation like [ABC]
                return VerificationResult(
                    status=VerificationStatus.REFUSE,
                    refusal_reason=f"invalid_citation:[{cit}]",
                    refusal_message=DEFAULT_REFUSAL_MESSAGE,
                    extracted_sections=extracted_sections,
                    extracted_acts=extracted_acts,
                    extracted_courts=extracted_courts,
                    extracted_citations=extracted_citations,
                )
        
        # ─────────────────────────────────────────────
        # ALL CHECKS PASSED
        # ─────────────────────────────────────────────
        return VerificationResult(
            status=VerificationStatus.PASS,
            extracted_sections=extracted_sections,
            extracted_acts=extracted_acts,
            extracted_courts=extracted_courts,
            extracted_citations=extracted_citations,
        )
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citation markers [1], [2], etc."""
        return re.findall(r'\[(\d+)\]', text)
    
    def _extract_sections(self, text: str) -> List[str]:
        """Extract section numbers from text."""
        # Match patterns like "Section 420", "section 302A", "S. 420"
        patterns = [
            r'section\s+(\d+[a-z]?)',
            r's\.\s*(\d+[a-z]?)',
            r'sec\.\s*(\d+[a-z]?)',
        ]
        
        sections = set()
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            sections.update(matches)
        
        return list(sections)
    
    def _extract_acts(self, text: str) -> List[str]:
        """Extract act/statute names from text."""
        text_lower = text.lower()
        found_acts = set()
        
        for act_key, variants in self.KNOWN_ACTS.items():
            for variant in variants:
                if variant in text_lower:
                    found_acts.add(act_key)
                    break
        
        return list(found_acts)
    
    def _extract_courts(self, text: str) -> List[str]:
        """Extract court names from text."""
        text_lower = text.lower()
        found_courts = set()
        
        for court_key, variants in self.KNOWN_COURTS.items():
            for variant in variants:
                if variant in text_lower:
                    found_courts.add(court_key)
                    break
        
        return list(found_courts)
    
    def _get_evidence_sections(
        self,
        chunks: List[Dict[str, Any]],
        context: str,
    ) -> Set[str]:
        """Get all sections from evidence."""
        sections = set()
        
        # From chunk metadata
        for chunk in chunks:
            section = chunk.get("section")
            if section:
                # Normalize section number
                section_str = str(section).lower()
                # Extract just the number
                match = re.search(r'(\d+[a-z]?)', section_str)
                if match:
                    sections.add(match.group(1))
        
        # From context text
        context_sections = self._extract_sections(context)
        sections.update(context_sections)
        
        return sections
    
    def _get_evidence_acts(
        self,
        chunks: List[Dict[str, Any]],
        context: str,
    ) -> Set[str]:
        """Get all acts from evidence."""
        acts = set()
        
        # From chunk metadata
        for chunk in chunks:
            act = chunk.get("act")
            if act:
                act_lower = str(act).lower()
                for act_key, variants in self.KNOWN_ACTS.items():
                    for variant in variants:
                        if variant in act_lower:
                            acts.add(act_key)
                            break
        
        # From context text
        context_acts = self._extract_acts(context)
        acts.update(context_acts)
        
        return acts
    
    def _get_evidence_courts(
        self,
        chunks: List[Dict[str, Any]],
        context: str,
    ) -> Set[str]:
        """Get all courts from evidence."""
        courts = set()
        
        # From chunk metadata
        for chunk in chunks:
            court = chunk.get("court")
            if court:
                court_lower = str(court).lower()
                for court_key, variants in self.KNOWN_COURTS.items():
                    for variant in variants:
                        if variant in court_lower:
                            courts.add(court_key)
                            break
        
        # From context text
        context_courts = self._extract_courts(context)
        courts.update(context_courts)
        
        return courts
    
    def log_verification(
        self,
        query: str,
        decoder: str,
        result: VerificationResult,
    ) -> None:
        """
        Log verification run.
        
        Args:
            query: User query
            decoder: Decoder model ID
            result: Verification result
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query[:500],
            "decoder": decoder,
            "verification_status": result.status.value,
            "refusal_reason": result.refusal_reason,
            "extracted_sections": result.extracted_sections,
            "extracted_acts": result.extracted_acts,
            "extracted_courts": result.extracted_courts,
        }
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"Warning: Failed to log verification: {e}")
