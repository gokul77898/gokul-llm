"""
API Module

Contains:
- main.py: FastAPI production server (legacy)
- legal_reasoning_api.py: Phase 6 Legal Reasoning API
- schemas.py: API response schemas
"""

# Phase 6 exports
from .legal_reasoning_api import LegalReasoningAPI
from .schemas import LegalAnswerResponse, RefusalReason

__all__ = [
    'LegalReasoningAPI',
    'LegalAnswerResponse',
    'RefusalReason',
]
