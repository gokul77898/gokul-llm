"""
Phase-7: Observability & Contracts - Exit Codes

Standard exit codes for RAG operations.
"""

# Exit codes
EXIT_SUCCESS = 0           # Success - answer provided and grounded
EXIT_GROUNDED_REFUSAL = 2  # Grounded refusal - evidence insufficient but valid
EXIT_CONTRACT_VIOLATION = 3  # Contract violation - grounding failed, invalid citations, etc.

# Exit code descriptions
EXIT_CODE_DESCRIPTIONS = {
    EXIT_SUCCESS: "Success - answer provided and grounded",
    EXIT_GROUNDED_REFUSAL: "Grounded refusal - evidence insufficient but valid",
    EXIT_CONTRACT_VIOLATION: "Contract violation - grounding failed or invalid citations",
}


def get_exit_code(
    is_grounded: bool,
    refusal_reason: str = None,
    invalid_citations: list = None,
    uncovered_claims: list = None
) -> int:
    """Determine appropriate exit code based on operation result.
    
    Args:
        is_grounded: Whether answer is grounded
        refusal_reason: Refusal reason (if any)
        invalid_citations: List of invalid citations
        uncovered_claims: List of uncovered claims
    
    Returns:
        Exit code (0, 2, or 3)
    """
    # Success: grounded answer with no violations
    if is_grounded and not refusal_reason:
        return EXIT_SUCCESS
    
    # Grounded refusal: valid refusal due to insufficient evidence
    if refusal_reason and is_grounded:
        return EXIT_GROUNDED_REFUSAL
    
    # Contract violation: invalid citations or uncovered claims
    if invalid_citations or uncovered_claims or not is_grounded:
        return EXIT_CONTRACT_VIOLATION
    
    # Default to contract violation for safety
    return EXIT_CONTRACT_VIOLATION


def get_exit_code_description(exit_code: int) -> str:
    """Get description for exit code.
    
    Args:
        exit_code: Exit code
    
    Returns:
        Description string
    """
    return EXIT_CODE_DESCRIPTIONS.get(exit_code, "Unknown exit code")
