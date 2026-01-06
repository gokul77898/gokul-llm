"""
Observability - Phase 8D

Structured logging for system observability.

NO PROMPT LOGGING.
NO EVIDENCE LOGGING.
NO MODEL INTERNALS EXPOSED.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class PhaseType(Enum):
    """Pipeline phase types."""
    INTENT_CLASSIFICATION = "intent_classification"
    RESOURCE_VALIDATION = "resource_validation"
    ADVERSARIAL_DETECTION = "adversarial_detection"
    RETRIEVAL = "retrieval"
    GRAPH_FILTERING = "graph_filtering"
    GENERATION = "generation"
    PRECEDENT_EXTRACTION = "precedent_extraction"
    PRECEDENT_LABELING = "precedent_labeling"
    EXPLANATION_ASSEMBLY = "explanation_assembly"


class ObservabilityLogger:
    """
    Structured logging for system observability.
    
    NO PROMPT LOGGING.
    NO EVIDENCE LOGGING.
    NO MODEL INTERNALS.
    """
    
    def __init__(self, request_id: Optional[str] = None):
        """
        Initialize observability logger.
        
        Args:
            request_id: Optional request ID for tracking
        """
        self.request_id = request_id or self._generate_request_id()
        self.phase_logs = []
        self.start_time = datetime.utcnow()
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return f"req_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # ─────────────────────────────────────────────
    # PHASE LOGGING
    # ─────────────────────────────────────────────
    
    def log_phase_start(
        self,
        phase: PhaseType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log phase start.
        
        Args:
            phase: Phase type
            metadata: Optional metadata (NO PROMPTS/EVIDENCE)
        """
        log_entry = {
            "request_id": self.request_id,
            "phase": phase.value,
            "event": "start",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        
        self.phase_logs.append(log_entry)
        logger.info(f"[{self.request_id}] Phase {phase.value} started")
    
    def log_phase_end(
        self,
        phase: PhaseType,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log phase end.
        
        Args:
            phase: Phase type
            success: Whether phase succeeded
            metadata: Optional metadata (NO PROMPTS/EVIDENCE)
        """
        log_entry = {
            "request_id": self.request_id,
            "phase": phase.value,
            "event": "end",
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        
        self.phase_logs.append(log_entry)
        status = "succeeded" if success else "failed"
        logger.info(f"[{self.request_id}] Phase {phase.value} {status}")
    
    def log_refusal(
        self,
        phase: PhaseType,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log refusal.
        
        Args:
            phase: Phase where refusal occurred
            reason: Refusal reason
            metadata: Optional metadata (NO PROMPTS/EVIDENCE)
        """
        log_entry = {
            "request_id": self.request_id,
            "phase": phase.value,
            "event": "refusal",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        
        self.phase_logs.append(log_entry)
        logger.warning(f"[{self.request_id}] Refusal in {phase.value}: {reason}")
    
    def log_error(
        self,
        phase: PhaseType,
        error: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log error.
        
        Args:
            phase: Phase where error occurred
            error: Error message
            metadata: Optional metadata (NO PROMPTS/EVIDENCE)
        """
        log_entry = {
            "request_id": self.request_id,
            "phase": phase.value,
            "event": "error",
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        
        self.phase_logs.append(log_entry)
        logger.error(f"[{self.request_id}] Error in {phase.value}: {error}")
    
    # ─────────────────────────────────────────────
    # METRICS LOGGING
    # ─────────────────────────────────────────────
    
    def log_metrics(
        self,
        phase: PhaseType,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Log phase metrics.
        
        Args:
            phase: Phase type
            metrics: Metrics dictionary (NO PROMPTS/EVIDENCE)
        """
        # Filter out sensitive data
        safe_metrics = self._sanitize_metrics(metrics)
        
        log_entry = {
            "request_id": self.request_id,
            "phase": phase.value,
            "event": "metrics",
            "metrics": safe_metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        self.phase_logs.append(log_entry)
        logger.debug(f"[{self.request_id}] Metrics for {phase.value}: {safe_metrics}")
    
    def _sanitize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metrics to remove sensitive data.
        
        NO PROMPTS.
        NO EVIDENCE.
        NO MODEL INTERNALS.
        
        Args:
            metrics: Raw metrics
            
        Returns:
            Sanitized metrics
        """
        safe_metrics = {}
        
        # Allowed metric keys
        allowed_keys = {
            "count", "duration", "length", "size",
            "retrieved_count", "allowed_count", "excluded_count",
            "cited_count", "precedent_count", "labeled_count",
            "success", "grounded", "intent", "allowed",
        }
        
        for key, value in metrics.items():
            # Only include allowed keys
            if any(allowed in key.lower() for allowed in allowed_keys):
                # Ensure value is not sensitive
                if isinstance(value, (int, float, bool, str)):
                    if isinstance(value, str) and len(value) < 100:
                        safe_metrics[key] = value
                    elif not isinstance(value, str):
                        safe_metrics[key] = value
        
        return safe_metrics
    
    # ─────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get request summary.
        
        Returns:
            Summary dictionary
        """
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()
        
        # Count phases
        phase_counts = {}
        for log_entry in self.phase_logs:
            phase = log_entry.get("phase")
            if phase:
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        # Count events
        refusals = sum(1 for log in self.phase_logs if log.get("event") == "refusal")
        errors = sum(1 for log in self.phase_logs if log.get("event") == "error")
        
        return {
            "request_id": self.request_id,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_phases": len(phase_counts),
            "phase_counts": phase_counts,
            "refusals": refusals,
            "errors": errors,
            "total_logs": len(self.phase_logs),
        }
    
    def print_summary(self) -> None:
        """Print request summary."""
        summary = self.get_summary()
        
        print("\n" + "═" * 60)
        print("REQUEST SUMMARY")
        print("═" * 60)
        
        print(f"\n📋 Request ID: {summary['request_id']}")
        print(f"⏱️  Duration: {summary['duration_seconds']:.2f}s")
        print(f"📊 Total Phases: {summary['total_phases']}")
        
        if summary['refusals'] > 0:
            print(f"❌ Refusals: {summary['refusals']}")
        
        if summary['errors'] > 0:
            print(f"⚠️  Errors: {summary['errors']}")
        
        print("\n" + "═" * 60)
    
    def export_logs(self) -> str:
        """
        Export logs as JSON.
        
        Returns:
            JSON string of logs
        """
        return json.dumps({
            "summary": self.get_summary(),
            "logs": self.phase_logs,
        }, indent=2)
