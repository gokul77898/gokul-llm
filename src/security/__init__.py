"""
System Identity and Security

Identity protection and security overrides for the legal AI system.
"""

from .identity import IdentityProtection, check_identity_query, get_identity_response
from .prompt_builder import SystemPromptBuilder, build_system_prompt

__all__ = [
    'IdentityProtection',
    'check_identity_query',
    'get_identity_response',
    'SystemPromptBuilder',
    'build_system_prompt',
]
