"""
Phase-7: Observability & Contracts

Structured logging and audit records for RAG operations.
"""

from .logging import (
    StructuredLogger,
    log_retrieval,
    log_synthesis,
    log_grounding_failure,
    log_inference_guard_trigger
)
from .audit import AuditRecord, create_audit_record, log_audit_record, print_audit_record
from .exit_codes import (
    EXIT_SUCCESS,
    EXIT_GROUNDED_REFUSAL,
    EXIT_CONTRACT_VIOLATION,
    get_exit_code,
    get_exit_code_description
)

__all__ = [
    'StructuredLogger',
    'log_retrieval',
    'log_synthesis',
    'log_grounding_failure',
    'log_inference_guard_trigger',
    'AuditRecord',
    'create_audit_record',
    'log_audit_record',
    'print_audit_record',
    'EXIT_SUCCESS',
    'EXIT_GROUNDED_REFUSAL',
    'EXIT_CONTRACT_VIOLATION',
    'get_exit_code',
    'get_exit_code_description',
]
