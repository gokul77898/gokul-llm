"""
Resource Limits - Phase 8A

Enforce resource limits to prevent abuse and ensure system stability.

FAIL FAST.
EXPLICIT REFUSALS.
NO SILENT FAILURES.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any

logger = logging.getLogger(__name__)


class LimitType(Enum):
    """Types of resource limits."""
    QUERY_LENGTH = "query_length"
    TOP_K = "top_k"
    GRAPH_DEPTH = "graph_depth"
    EXECUTION_TIMEOUT = "execution_timeout"


@dataclass
class ResourceLimitViolation(Exception):
    """
    Exception raised when a resource limit is violated.
    """
    limit_type: LimitType
    actual_value: Any
    max_value: Any
    message: str
    
    def __str__(self):
        return self.message


class ResourceLimits:
    """
    Enforce resource limits for system hardening.
    
    NO SILENT FAILURES.
    FAIL FAST WITH EXPLICIT REFUSAL.
    """
    
    # ─────────────────────────────────────────────
    # DEFAULT LIMITS
    # ─────────────────────────────────────────────
    
    DEFAULT_MAX_QUERY_LENGTH = 1000  # characters
    DEFAULT_MAX_TOP_K = 50  # chunks
    DEFAULT_MAX_GRAPH_DEPTH = 10  # hops
    DEFAULT_EXECUTION_TIMEOUT = 300  # seconds (5 minutes)
    
    def __init__(
        self,
        max_query_length: int = DEFAULT_MAX_QUERY_LENGTH,
        max_top_k: int = DEFAULT_MAX_TOP_K,
        max_graph_depth: int = DEFAULT_MAX_GRAPH_DEPTH,
        execution_timeout: int = DEFAULT_EXECUTION_TIMEOUT,
    ):
        """
        Initialize resource limits.
        
        Args:
            max_query_length: Maximum query length in characters
            max_top_k: Maximum number of chunks to retrieve
            max_graph_depth: Maximum graph traversal depth
            execution_timeout: Maximum execution time in seconds
        """
        self.max_query_length = max_query_length
        self.max_top_k = max_top_k
        self.max_graph_depth = max_graph_depth
        self.execution_timeout = execution_timeout
        
        logger.info(
            f"Resource limits initialized: "
            f"query_length={max_query_length}, "
            f"top_k={max_top_k}, "
            f"graph_depth={max_graph_depth}, "
            f"timeout={execution_timeout}s"
        )
    
    # ─────────────────────────────────────────────
    # LIMIT ENFORCEMENT
    # ─────────────────────────────────────────────
    
    def validate_query_length(self, query: str) -> None:
        """
        Validate query length.
        
        Args:
            query: Query string
            
        Raises:
            ResourceLimitViolation: If query exceeds max length
        """
        query_length = len(query)
        
        if query_length > self.max_query_length:
            logger.warning(
                f"Query length violation: {query_length} > {self.max_query_length}"
            )
            raise ResourceLimitViolation(
                limit_type=LimitType.QUERY_LENGTH,
                actual_value=query_length,
                max_value=self.max_query_length,
                message=(
                    f"Query length ({query_length} characters) exceeds maximum "
                    f"allowed length ({self.max_query_length} characters). "
                    f"Please shorten your query."
                ),
            )
        
        logger.debug(f"Query length validated: {query_length}/{self.max_query_length}")
    
    def validate_top_k(self, top_k: int) -> None:
        """
        Validate top_k parameter.
        
        Args:
            top_k: Number of chunks to retrieve
            
        Raises:
            ResourceLimitViolation: If top_k exceeds max
        """
        if top_k > self.max_top_k:
            logger.warning(f"Top-K violation: {top_k} > {self.max_top_k}")
            raise ResourceLimitViolation(
                limit_type=LimitType.TOP_K,
                actual_value=top_k,
                max_value=self.max_top_k,
                message=(
                    f"Requested top_k ({top_k}) exceeds maximum "
                    f"allowed value ({self.max_top_k}). "
                    f"Please reduce the number of chunks."
                ),
            )
        
        if top_k <= 0:
            logger.warning(f"Invalid top_k: {top_k}")
            raise ResourceLimitViolation(
                limit_type=LimitType.TOP_K,
                actual_value=top_k,
                max_value=self.max_top_k,
                message=f"top_k must be positive, got {top_k}",
            )
        
        logger.debug(f"Top-K validated: {top_k}/{self.max_top_k}")
    
    def validate_graph_depth(self, depth: int) -> None:
        """
        Validate graph traversal depth.
        
        Args:
            depth: Graph traversal depth
            
        Raises:
            ResourceLimitViolation: If depth exceeds max
        """
        if depth > self.max_graph_depth:
            logger.warning(f"Graph depth violation: {depth} > {self.max_graph_depth}")
            raise ResourceLimitViolation(
                limit_type=LimitType.GRAPH_DEPTH,
                actual_value=depth,
                max_value=self.max_graph_depth,
                message=(
                    f"Graph traversal depth ({depth}) exceeds maximum "
                    f"allowed depth ({self.max_graph_depth})."
                ),
            )
        
        logger.debug(f"Graph depth validated: {depth}/{self.max_graph_depth}")
    
    def validate_all(self, query: str, top_k: int) -> None:
        """
        Validate all resource limits.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            
        Raises:
            ResourceLimitViolation: If any limit is violated
        """
        self.validate_query_length(query)
        self.validate_top_k(top_k)
    
    # ─────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────
    
    def get_limits(self) -> Dict[str, int]:
        """Get current limits."""
        return {
            "max_query_length": self.max_query_length,
            "max_top_k": self.max_top_k,
            "max_graph_depth": self.max_graph_depth,
            "execution_timeout": self.execution_timeout,
        }
    
    def check_query_within_limits(self, query: str) -> bool:
        """Check if query is within limits without raising exception."""
        return len(query) <= self.max_query_length
    
    def check_top_k_within_limits(self, top_k: int) -> bool:
        """Check if top_k is within limits without raising exception."""
        return 0 < top_k <= self.max_top_k
