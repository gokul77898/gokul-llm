"""
Legal Reasoning API - Phase 6

User-facing API for legal reasoning queries.

STRICT ORCHESTRATION.
NO NEW REASONING.
NO RETRIES.
NO HIDING REFUSALS.
"""

import logging
from datetime import datetime
from typing import Optional

from .schemas import LegalAnswerResponse, RefusalReason
from ..generation.graph_grounded_generator import (
    GraphGroundedGenerator,
    GroundedAnswerResult,
)
from ..explanation.precedent_extractor import PrecedentPathExtractor
from ..explanation.precedent_labeler import PrecedentLabeler
from ..explanation.explanation_assembler import ExplanationAssembler
from ..graph.graph_rag_filter import GraphRAGFilter
from ..graph.legal_graph_traverser import LegalGraphTraverser
from ..intent.legal_intent_classifier import LegalIntentClassifier
from ..hardening.resource_limits import ResourceLimits, ResourceLimitViolation
from ..hardening.adversarial_defense import AdversarialDefense
from ..hardening.observability import ObservabilityLogger, PhaseType

logger = logging.getLogger(__name__)


class LegalReasoningAPI:
    """
    User-facing API for legal reasoning.
    
    Orchestrates the complete pipeline:
    - Phase 4: Graph-grounded generation
    - Phase 5A: Precedent extraction
    - Phase 5B: Precedent labeling
    - Phase 5C: Explanation assembly
    
    NO new reasoning.
    NO retries.
    NO hiding refusals.
    """
    
    def __init__(
        self,
        traverser: LegalGraphTraverser,
        retriever,
        generator=None,
        enable_hardening: bool = True,
    ):
        """
        Initialize API.
        
        Args:
            traverser: LegalGraphTraverser for graph operations
            retriever: REAL retriever for RAG (NO MOCKS ALLOWED)
            generator: Optional LLM generator (None for mock)
            enable_hardening: Enable Phase 8 hardening (default: True)
        """
        # ENFORCE REAL RETRIEVER CONTRACT
        if retriever is None:
            raise RuntimeError(
                "REAL RETRIEVER REQUIRED. "
                "Mock retrieval is forbidden in production."
            )
        
        # Verify retriever has required interface
        if not hasattr(retriever, 'retrieve'):
            raise RuntimeError(
                "Retriever must have 'retrieve' method."
            )
        
        # Test retriever interface
        try:
            test_result = retriever.retrieve("test", 1)
            if not isinstance(test_result, list):
                raise RuntimeError(
                    "Retriever must return list from retrieve() method."
                )
        except Exception as e:
            raise RuntimeError(
                f"Retriever interface test failed: {e}"
            )
        # Phase 8: Hardening components
        self.enable_hardening = enable_hardening
        if enable_hardening:
            self.resource_limits = ResourceLimits()
            self.adversarial_defense = AdversarialDefense()
        
        # Phase 7: Intent classifier
        self.intent_classifier = LegalIntentClassifier()
        
        # Phase 3B: Graph filter
        self.graph_filter = GraphRAGFilter(traverser)
        
        # Phase 4: Graph-grounded generator
        self.grounded_generator = GraphGroundedGenerator(
            graph_filter=self.graph_filter,
            retriever=retriever,
            generator=generator,
        )
        
        # Phase 5A: Precedent extractor
        self.precedent_extractor = PrecedentPathExtractor(traverser)
        
        # Phase 5B: Precedent labeler
        self.precedent_labeler = PrecedentLabeler()
        
        # Phase 5C: Explanation assembler
        self.explanation_assembler = ExplanationAssembler()
        
        logger.info(f"Legal Reasoning API initialized (hardening={'enabled' if enable_hardening else 'disabled'})")
    
    # ─────────────────────────────────────────────
    # MAIN API METHOD
    # ─────────────────────────────────────────────
    
    def answer_query(
        self,
        query: str,
        top_k: int = 10
    ) -> LegalAnswerResponse:
        """
        Answer a legal query with full reasoning.
        
        Pipeline:
        1. Call GraphGroundedGenerator
        2. If refused → return refusal response
        3. If answered → extract precedents
        4. Label precedents
        5. Assemble explanation
        6. Return complete response
        
        Args:
            query: Legal query
            top_k: Number of chunks to retrieve
            
        Returns:
            LegalAnswerResponse with answer or refusal
        """
        logger.info(f"Processing query: {query}")
        timestamp = datetime.utcnow().isoformat()
        
        # Initialize observability
        obs_logger = ObservabilityLogger() if self.enable_hardening else None
        
        # ─────────────────────────────────────────────
        # STEP -2: Resource Limits (Phase 8A)
        # ─────────────────────────────────────────────
        
        if self.enable_hardening:
            if obs_logger:
                obs_logger.log_phase_start(PhaseType.RESOURCE_VALIDATION)
            
            try:
                self.resource_limits.validate_all(query, top_k)
                if obs_logger:
                    obs_logger.log_phase_end(PhaseType.RESOURCE_VALIDATION, success=True)
            except ResourceLimitViolation as e:
                if obs_logger:
                    obs_logger.log_refusal(PhaseType.RESOURCE_VALIDATION, str(e))
                logger.warning(f"Resource limit violation: {e}")
                return self._create_hardening_refusal_response(
                    query=query,
                    reason="resource_limit_exceeded",
                    message=str(e),
                    timestamp=timestamp
                )
            except Exception as e:
                if obs_logger:
                    obs_logger.log_error(PhaseType.RESOURCE_VALIDATION, str(e))
                logger.error(f"Error in resource validation: {e}")
                return self._create_error_response(query, timestamp, str(e))
        
        # ─────────────────────────────────────────────
        # STEP -1: Adversarial Defense (Phase 8B)
        # ─────────────────────────────────────────────
        
        if self.enable_hardening:
            if obs_logger:
                obs_logger.log_phase_start(PhaseType.ADVERSARIAL_DETECTION)
            
            try:
                adversarial_result = self.adversarial_defense.detect(query)
                if obs_logger:
                    obs_logger.log_phase_end(
                        PhaseType.ADVERSARIAL_DETECTION,
                        success=True,
                        metadata={"is_adversarial": adversarial_result.is_adversarial}
                    )
                
                if adversarial_result.is_adversarial:
                    if obs_logger:
                        obs_logger.log_refusal(
                            PhaseType.ADVERSARIAL_DETECTION,
                            "adversarial_pattern_detected"
                        )
                    logger.warning(f"Adversarial query detected: {adversarial_result.detected_patterns}")
                    return self._create_hardening_refusal_response(
                        query=query,
                        reason="adversarial_query_blocked",
                        message=adversarial_result.explanation,
                        timestamp=timestamp
                    )
            except Exception as e:
                if obs_logger:
                    obs_logger.log_error(PhaseType.ADVERSARIAL_DETECTION, str(e))
                logger.error(f"Error in adversarial detection: {e}")
                return self._create_error_response(query, timestamp, str(e))
        
        # ─────────────────────────────────────────────
        # STEP 0: Intent Classification (Phase 7)
        # ─────────────────────────────────────────────
        
        if obs_logger:
            obs_logger.log_phase_start(PhaseType.INTENT_CLASSIFICATION)
        
        try:
            intent_result = self.intent_classifier.classify(query)
            if obs_logger:
                obs_logger.log_phase_end(
                    PhaseType.INTENT_CLASSIFICATION,
                    success=True,
                    metadata={"intent": intent_result.intent.value, "allowed": intent_result.allowed}
                )
            logger.info(f"Intent: {intent_result.intent.value}, Allowed: {intent_result.allowed}")
        except Exception as e:
            if obs_logger:
                obs_logger.log_error(PhaseType.INTENT_CLASSIFICATION, str(e))
            logger.error(f"Error in intent classification: {e}")
            return self._create_error_response(query, timestamp, str(e))
        
        # Check if intent is blocked
        if not intent_result.allowed:
            if obs_logger:
                obs_logger.log_refusal(
                    PhaseType.INTENT_CLASSIFICATION,
                    intent_result.refusal_reason.value
                )
            logger.info(f"Query blocked by intent classifier: {intent_result.refusal_reason.value}")
            return self._create_intent_refusal_response(
                query=query,
                intent_result=intent_result,
                timestamp=timestamp
            )
        
        # ─────────────────────────────────────────────
        # STEP 1: Graph-Grounded Generation (Phase 4)
        # ─────────────────────────────────────────────
        
        try:
            grounded_result = self.grounded_generator.generate_answer(
                query=query,
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"Error in grounded generation: {e}")
            return self._create_error_response(query, timestamp, str(e))
        
        # ─────────────────────────────────────────────
        # STEP 2: Check for Refusal
        # ─────────────────────────────────────────────
        
        if grounded_result.refusal_reason:
            logger.info(f"Query refused: {grounded_result.refusal_reason}")
            return self._create_refusal_response(
                query=query,
                grounded_result=grounded_result,
                timestamp=timestamp
            )
        
        # ─────────────────────────────────────────────
        # STEP 3: Precedent Extraction (Phase 5A)
        # ─────────────────────────────────────────────
        
        try:
            precedents = self.precedent_extractor.extract(grounded_result)
            logger.info(f"Extracted {len(precedents)} precedents")
        except Exception as e:
            logger.error(f"Error in precedent extraction: {e}")
            return self._create_error_response(query, timestamp, str(e))
        
        # ─────────────────────────────────────────────
        # STEP 4: Precedent Labeling (Phase 5B)
        # ─────────────────────────────────────────────
        
        try:
            labeled_precedents = self.precedent_labeler.label(precedents)
            logger.info(f"Labeled {len(labeled_precedents)} precedents")
        except Exception as e:
            logger.error(f"Error in precedent labeling: {e}")
            return self._create_error_response(query, timestamp, str(e))
        
        # ─────────────────────────────────────────────
        # STEP 5: Explanation Assembly (Phase 5C)
        # ─────────────────────────────────────────────
        
        try:
            explanation = self.explanation_assembler.assemble(
                grounded_result=grounded_result,
                labeled_precedents=labeled_precedents
            )
            logger.info("Assembled explanation")
        except Exception as e:
            logger.error(f"Error in explanation assembly: {e}")
            return self._create_error_response(query, timestamp, str(e))
        
        # ─────────────────────────────────────────────
        # STEP 6: Create Response
        # ─────────────────────────────────────────────
        
        response = LegalAnswerResponse(
            query=query,
            answered=True,
            answer=explanation.answer,
            statutory_basis=explanation.statutory_basis,
            judicial_interpretations=explanation.judicial_interpretations,
            applied_precedents=explanation.applied_precedents,
            supporting_precedents=explanation.supporting_precedents,
            excluded_precedents=explanation.excluded_precedents,
            explanation_text=explanation.explanation_text,
            refusal_reason=None,
            retrieved_count=grounded_result.retrieved_count,
            allowed_chunks_count=grounded_result.allowed_chunks_count,
            excluded_chunks_count=grounded_result.excluded_chunks_count,
            cited_count=len(grounded_result.cited_semantic_ids),
            grounded=grounded_result.grounded,
            timestamp=timestamp,
        )
        
        logger.info("Query answered successfully")
        return response
    
    # ─────────────────────────────────────────────
    # HELPER METHODS
    # ─────────────────────────────────────────────
    
    def _create_refusal_response(
        self,
        query: str,
        grounded_result: GroundedAnswerResult,
        timestamp: str
    ) -> LegalAnswerResponse:
        """
        Create refusal response.
        
        Args:
            query: Original query
            grounded_result: Result from grounded generation
            timestamp: ISO timestamp
            
        Returns:
            LegalAnswerResponse with refusal
        """
        # Map refusal reason to enum
        refusal_reason = grounded_result.refusal_reason or "unknown"
        
        if "excluded" in refusal_reason.lower():
            reason = RefusalReason.ALL_CHUNKS_EXCLUDED.value
        elif "retrieval" in refusal_reason.lower():
            reason = RefusalReason.NO_RETRIEVAL.value
        elif "citation" in refusal_reason.lower():
            reason = RefusalReason.CITATION_VALIDATION_FAILED.value
        elif "graph" in refusal_reason.lower() or "filter" in refusal_reason.lower():
            reason = RefusalReason.GRAPH_FILTER_BLOCKED.value
        else:
            reason = RefusalReason.UNKNOWN.value
        
        return LegalAnswerResponse(
            query=query,
            answered=False,
            answer=grounded_result.answer,  # May contain refusal message
            refusal_reason=reason,
            retrieved_count=grounded_result.retrieved_count,
            allowed_chunks_count=grounded_result.allowed_chunks_count,
            excluded_chunks_count=grounded_result.excluded_chunks_count,
            cited_count=len(grounded_result.cited_semantic_ids),
            grounded=False,
            timestamp=timestamp,
        )
    
    def _create_hardening_refusal_response(
        self,
        query: str,
        reason: str,
        message: str,
        timestamp: str
    ) -> LegalAnswerResponse:
        """
        Create refusal response for hardening violations.
        
        Args:
            query: Original query
            reason: Refusal reason
            message: Refusal message
            timestamp: ISO timestamp
            
        Returns:
            LegalAnswerResponse with hardening refusal
        """
        return LegalAnswerResponse(
            query=query,
            answered=False,
            answer=message,
            refusal_reason=reason,
            timestamp=timestamp,
        )
    
    def _create_intent_refusal_response(
        self,
        query: str,
        intent_result,
        timestamp: str
    ) -> LegalAnswerResponse:
        """
        Create refusal response for blocked intent.
        
        Args:
            query: Original query
            intent_result: IntentResult from classifier
            timestamp: ISO timestamp
            
        Returns:
            LegalAnswerResponse with intent refusal
        """
        return LegalAnswerResponse(
            query=query,
            answered=False,
            answer=intent_result.explanation,
            refusal_reason=intent_result.refusal_reason.value,
            timestamp=timestamp,
        )
    
    def _create_error_response(
        self,
        query: str,
        timestamp: str,
        error: str
    ) -> LegalAnswerResponse:
        """
        Create error response.
        
        Args:
            query: Original query
            timestamp: ISO timestamp
            error: Error message
            
        Returns:
            LegalAnswerResponse with error
        """
        return LegalAnswerResponse(
            query=query,
            answered=False,
            answer=f"Error processing query: {error}",
            refusal_reason=RefusalReason.UNKNOWN.value,
            timestamp=timestamp,
        )
    
    # ─────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────
    
    def get_api_stats(self) -> dict:
        """Get API statistics."""
        return {
            "components": {
                "graph_filter": "GraphRAGFilter",
                "grounded_generator": "GraphGroundedGenerator",
                "precedent_extractor": "PrecedentPathExtractor",
                "precedent_labeler": "PrecedentLabeler",
                "explanation_assembler": "ExplanationAssembler",
            },
            "pipeline": [
                "Phase 4: Graph-Grounded Generation",
                "Phase 5A: Precedent Extraction",
                "Phase 5B: Precedent Labeling",
                "Phase 5C: Explanation Assembly",
            ],
        }
