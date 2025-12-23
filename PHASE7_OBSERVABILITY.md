# Phase-7: Observability & Contracts

## Overview

Phase-7 implements **observability and contracts** with structured logging, audit records, and proper exit codes for all RAG operations. All refusals produce audit output for compliance and debugging.

**Key Principle:** Complete observability of all operations with structured logging and audit trails.

## Architecture

### Observability Flow

```
Operation Start
    ↓
Log Retrieval (structured)
    ↓
Log Synthesis (structured)
    ↓
Log Grounding Failure (if applicable)
    ↓
Create Audit Record
    ↓
Log Audit Record
    ↓
Exit with Proper Code (0, 2, or 3)
```

## Implementation

### 1. Structured Logging (`src/observability/logging.py`)

Structured JSON logging for all RAG operations.

#### Log Types

**Retrieval Log:**
```python
log_retrieval(
    query="What is Section 420 IPC?",
    retrieved_count=5,
    semantic_ids=["IPC_420_0", "IPC_420_1", ...],
    top_k=5,
    collection_name="legal_chunks",
    duration_ms=150.5  # Optional
)
```

**Output:**
```json
{
  "timestamp": "2025-12-21T13:19:00.000Z",
  "event_type": "retrieval",
  "query": "What is Section 420 IPC?",
  "retrieved_count": 5,
  "semantic_ids": ["IPC_420_0", "IPC_420_1"],
  "top_k": 5,
  "collection_name": "legal_chunks",
  "duration_ms": 150.5
}
```

**Synthesis Log:**
```python
log_synthesis(
    query="What is Section 420 IPC?",
    phase="C3+",
    is_sufficient=True,
    is_grounded=True,
    cited_sources=["IPC_420_0"],
    retrieved_semantic_ids=["IPC_420_0", "IPC_420_1"],
    answer_length=250,
    duration_ms=500.0  # Optional
)
```

**Output:**
```json
{
  "timestamp": "2025-12-21T13:19:01.000Z",
  "event_type": "synthesis",
  "query": "What is Section 420 IPC?",
  "phase": "C3+",
  "is_sufficient": true,
  "is_grounded": true,
  "cited_sources": ["IPC_420_0"],
  "retrieved_semantic_ids": ["IPC_420_0", "IPC_420_1"],
  "answer_length": 250,
  "citation_count": 1,
  "duration_ms": 500.0
}
```

**Grounding Failure Log:**
```python
log_grounding_failure(
    query="What is Section 420 IPC?",
    phase="C3+",
    failure_type="insufficient_evidence",
    reason="No chunks contain definitional language",
    retrieved_semantic_ids=["IPC_420_0"],
    cited_sources=[],
    invalid_citations=None,
    uncovered_claims=None
)
```

**Output:**
```json
{
  "timestamp": "2025-12-21T13:19:01.000Z",
  "event_type": "grounding_failure",
  "query": "What is Section 420 IPC?",
  "phase": "C3+",
  "failure_type": "insufficient_evidence",
  "reason": "No chunks contain definitional language",
  "retrieved_semantic_ids": ["IPC_420_0"]
}
```

**Inference Guard Trigger Log:**
```python
log_inference_guard_trigger(
    guard_type="env_var_missing",
    reason="ALLOW_LLM_INFERENCE environment variable not set to 1",
    model_name="Qwen/Qwen2.5-32B-Instruct",
    device="cuda",
    enable_inference=True,
    env_var_set=False
)
```

**Output:**
```json
{
  "timestamp": "2025-12-21T13:19:01.000Z",
  "event_type": "inference_guard_trigger",
  "guard_type": "env_var_missing",
  "reason": "ALLOW_LLM_INFERENCE environment variable not set to 1",
  "model_name": "Qwen/Qwen2.5-32B-Instruct",
  "device": "cuda",
  "enable_inference": true,
  "env_var_set": false
}
```

### 2. Audit Records (`src/observability/audit.py`)

Comprehensive audit records for all operations including refusals.

#### AuditRecord Dataclass

```python
@dataclass
class AuditRecord:
    """Audit record for RAG operation."""
    
    query: str                              # User query
    retrieved_semantic_ids: List[str]       # Retrieved chunks
    cited_ids: List[str]                    # Cited chunks
    refusal_reason: Optional[str]           # Refusal reason (None if success)
    phase: str                              # Phase (C2, C3, C3+)
    timestamp: str                          # UTC timestamp
    is_grounded: bool                       # Grounding status
    is_sufficient: Optional[bool]           # Evidence sufficiency (C3+ only)
    invalid_citations: Optional[List[str]]  # Invalid citations
    uncovered_claims: Optional[List[str]]   # Uncovered claims
```

#### Audit Record Creation

```python
from src.observability import create_audit_record, log_audit_record

# Create audit record
audit_record = create_audit_record(
    query="What is Section 420 IPC?",
    retrieved_semantic_ids=["IPC_420_0", "IPC_420_1"],
    cited_ids=["IPC_420_0"],
    refusal_reason=None,  # None = success
    phase="C3+",
    is_grounded=True,
    is_sufficient=True,
    invalid_citations=None,
    uncovered_claims=None
)

# Log audit record
log_audit_record(audit_record)
```

#### Audit Record Output

**Success:**
```json
{
  "event_type": "audit_record",
  "query": "What is Section 420 IPC?",
  "retrieved_semantic_ids": ["IPC_420_0", "IPC_420_1"],
  "cited_ids": ["IPC_420_0"],
  "refusal_reason": null,
  "phase": "C3+",
  "timestamp": "2025-12-21T13:19:01.000Z",
  "is_grounded": true,
  "is_sufficient": true,
  "invalid_citations": null,
  "uncovered_claims": null
}
```

**Refusal:**
```json
{
  "event_type": "audit_record",
  "query": "What is the definition of employer?",
  "retrieved_semantic_ids": ["MinimumWagesAct_1_0"],
  "cited_ids": [],
  "refusal_reason": "No chunks contain definitional language",
  "phase": "C3+",
  "timestamp": "2025-12-21T13:19:02.000Z",
  "is_grounded": true,
  "is_sufficient": false,
  "invalid_citations": null,
  "uncovered_claims": null
}
```

### 3. Exit Codes (`src/observability/exit_codes.py`)

Standard exit codes for all operations.

#### Exit Code Definitions

| Code | Name | Description |
|------|------|-------------|
| **0** | `EXIT_SUCCESS` | Success - answer provided and grounded |
| **2** | `EXIT_GROUNDED_REFUSAL` | Grounded refusal - evidence insufficient but valid |
| **3** | `EXIT_CONTRACT_VIOLATION` | Contract violation - grounding failed or invalid citations |

#### Exit Code Logic

```python
from src.observability import get_exit_code

exit_code = get_exit_code(
    is_grounded=True,
    refusal_reason="Evidence insufficient",
    invalid_citations=None,
    uncovered_claims=None
)
# Returns: 2 (EXIT_GROUNDED_REFUSAL)

exit_code = get_exit_code(
    is_grounded=False,
    refusal_reason=None,
    invalid_citations=["INVALID_ID"],
    uncovered_claims=None
)
# Returns: 3 (EXIT_CONTRACT_VIOLATION)

exit_code = get_exit_code(
    is_grounded=True,
    refusal_reason=None,
    invalid_citations=None,
    uncovered_claims=None
)
# Returns: 0 (EXIT_SUCCESS)
```

## Usage

### Structured Logging

```python
from src.observability import (
    log_retrieval,
    log_synthesis,
    log_grounding_failure,
    log_inference_guard_trigger
)

# Log retrieval
log_retrieval(
    query=query,
    retrieved_count=len(chunks),
    semantic_ids=[c.semantic_id for c in chunks],
    top_k=5,
    collection_name="legal_chunks"
)

# Log synthesis
log_synthesis(
    query=query,
    phase="C3+",
    is_sufficient=True,
    is_grounded=True,
    cited_sources=["IPC_420_0"],
    retrieved_semantic_ids=["IPC_420_0", "IPC_420_1"],
    answer_length=250
)

# Log grounding failure
log_grounding_failure(
    query=query,
    phase="C3+",
    failure_type="insufficient_evidence",
    reason="No definitional language found",
    retrieved_semantic_ids=["IPC_420_0"]
)
```

### Audit Records

```python
from src.observability import (
    create_audit_record,
    log_audit_record,
    print_audit_record
)

# Create audit record
audit_record = create_audit_record(
    query=query,
    retrieved_semantic_ids=retrieved_ids,
    cited_ids=cited_ids,
    refusal_reason=refusal_reason,
    phase="C3+",
    is_grounded=is_grounded,
    is_sufficient=is_sufficient
)

# Log to structured log
log_audit_record(audit_record)

# Print human-readable
print_audit_record(audit_record, verbose=True)
```

### Exit Codes

```python
from src.observability import get_exit_code
import sys

# Determine exit code
exit_code = get_exit_code(
    is_grounded=result.is_grounded,
    refusal_reason=result.refusal_reason,
    invalid_citations=result.invalid_citations,
    uncovered_claims=result.uncovered_claims
)

# Exit with proper code
sys.exit(exit_code)
```

## Integration

### c3_generate.py (Phase C2)

```python
# Phase-7: Structured logging and audit
from src.observability import (
    log_retrieval,
    log_synthesis,
    log_grounding_failure,
    create_audit_record,
    log_audit_record,
    get_exit_code
)

# Log retrieval
log_retrieval(query, len(chunks), semantic_ids, top_k, collection_name)

# Generate answer
result = generate_grounded_answer(query, chunks)

# Log synthesis
log_synthesis(query, "C2", True, result.is_grounded, ...)

# Log failure if applicable
if not result.is_grounded:
    log_grounding_failure(query, "C2", "contract_violation", ...)

# Create audit record
audit_record = create_audit_record(query, retrieved_ids, cited_ids, ...)
log_audit_record(audit_record)

# Exit with proper code
exit_code = get_exit_code(result.is_grounded, result.refusal_reason, ...)
sys.exit(exit_code)
```

### c3_synthesize.py (Phase C3+)

```python
# Phase-7: Structured logging and audit
from src.observability import (
    log_retrieval,
    log_synthesis,
    log_grounding_failure,
    create_audit_record,
    log_audit_record,
    get_exit_code
)

# Log retrieval
log_retrieval(query, len(chunks), semantic_ids, top_k, collection_name)

# Synthesize answer
result = synthesize_answer(query, chunks)

# Log synthesis
log_synthesis(query, "C3+", result.is_sufficient, result.is_grounded, ...)

# Log failure if applicable
if not result.is_sufficient or not result.is_grounded:
    log_grounding_failure(query, "C3+", failure_type, ...)

# Create audit record
audit_record = create_audit_record(
    query, retrieved_ids, cited_ids, refusal_reason,
    "C3+", result.is_grounded, result.is_sufficient,
    result.invalid_citations, result.uncovered_claims
)
log_audit_record(audit_record)

# Exit with proper code
exit_code = get_exit_code(
    result.is_grounded, result.refusal_reason,
    result.invalid_citations, result.uncovered_claims
)
sys.exit(exit_code)
```

## Refusal Tracking

### All Refusals Produce Audit Output

**Refusal Scenarios:**

1. **Insufficient Evidence (C3+)**
   - Audit record with `refusal_reason`
   - Exit code: 2 (GROUNDED_REFUSAL)
   - Grounding failure log

2. **Invalid Citations**
   - Audit record with `invalid_citations`
   - Exit code: 3 (CONTRACT_VIOLATION)
   - Grounding failure log

3. **Uncovered Claims (C3+)**
   - Audit record with `uncovered_claims`
   - Exit code: 3 (CONTRACT_VIOLATION)
   - Grounding failure log

4. **Contract Violation**
   - Audit record with `is_grounded=False`
   - Exit code: 3 (CONTRACT_VIOLATION)
   - Grounding failure log

### Example Refusal Audit

```json
{
  "event_type": "audit_record",
  "query": "What is the definition of employer?",
  "retrieved_semantic_ids": ["MinimumWagesAct_1_0", "MinimumWagesAct_1_1"],
  "cited_ids": [],
  "refusal_reason": "No chunks contain definitional language",
  "phase": "C3+",
  "timestamp": "2025-12-21T13:19:02.000Z",
  "is_grounded": true,
  "is_sufficient": false,
  "invalid_citations": null,
  "uncovered_claims": null
}
```

## Exit Code Scenarios

### Scenario 1: Success

**Operation:**
- Query answered
- Evidence sufficient
- Answer grounded
- All citations valid

**Exit Code:** 0 (EXIT_SUCCESS)

**Audit:**
```json
{
  "refusal_reason": null,
  "is_grounded": true,
  "is_sufficient": true
}
```

### Scenario 2: Grounded Refusal

**Operation:**
- Evidence insufficient
- Valid refusal
- No contract violation

**Exit Code:** 2 (EXIT_GROUNDED_REFUSAL)

**Audit:**
```json
{
  "refusal_reason": "No chunks contain definitional language",
  "is_grounded": true,
  "is_sufficient": false
}
```

### Scenario 3: Contract Violation

**Operation:**
- Invalid citations
- OR uncovered claims
- OR grounding failed

**Exit Code:** 3 (EXIT_CONTRACT_VIOLATION)

**Audit:**
```json
{
  "refusal_reason": null,
  "is_grounded": false,
  "invalid_citations": ["INVALID_ID"]
}
```

## Files Created/Modified

### Created Files

1. **`src/observability/__init__.py`** - Package initialization
2. **`src/observability/logging.py`** (180 lines) - Structured logging
3. **`src/observability/audit.py`** (220 lines) - Audit records
4. **`src/observability/exit_codes.py`** (70 lines) - Exit code logic

### Modified Files

1. **`scripts/c3_generate.py`** - Added Phase-7 observability
2. **`scripts/c3_synthesize.py`** - Added Phase-7 observability

## Logging Configuration

### Python Logging Setup

```python
import logging
import json

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Raw JSON output
    handlers=[
        logging.FileHandler('rag_audit.log'),
        logging.StreamHandler()
    ]
)
```

### Log Parsing

```bash
# Parse structured logs
cat rag_audit.log | jq 'select(.event_type == "audit_record")'

# Count refusals
cat rag_audit.log | jq 'select(.event_type == "audit_record" and .refusal_reason != null)' | wc -l

# Find contract violations
cat rag_audit.log | jq 'select(.event_type == "grounding_failure" and .failure_type == "contract_violation")'
```

## Monitoring & Alerting

### Key Metrics

1. **Refusal Rate** - Percentage of queries resulting in refusal
2. **Contract Violation Rate** - Percentage of contract violations
3. **Average Citations** - Average number of citations per answer
4. **Retrieval Success** - Percentage of successful retrievals

### Example Queries

```bash
# Refusal rate
cat rag_audit.log | jq -s 'map(select(.event_type == "audit_record")) | {total: length, refusals: map(select(.refusal_reason != null)) | length}'

# Contract violations
cat rag_audit.log | jq -s 'map(select(.event_type == "grounding_failure" and .failure_type == "contract_violation")) | length'

# Average citations
cat rag_audit.log | jq -s 'map(select(.event_type == "synthesis")) | map(.citation_count) | add / length'
```

## Design Principles

### Complete Observability

- All operations logged
- All refusals tracked
- All failures recorded
- Structured JSON format

### Audit Trail

- Every query has audit record
- Refusals explicitly tracked
- Contract violations logged
- Timestamps for all events

### Proper Exit Codes

- 0 = Success
- 2 = Grounded refusal (valid)
- 3 = Contract violation (invalid)

### Structured Data

- JSON format for parsing
- Consistent schema
- Machine-readable
- Human-readable summaries

## Summary

### Delivered

✓ **Structured logging** - Retrieval, synthesis, failures, guards  
✓ **Audit records** - Complete tracking with AuditRecord dataclass  
✓ **Refusal tracking** - All refusals produce audit output  
✓ **Exit codes** - 0=success, 2=grounded refusal, 3=contract violation  
✓ **Integration** - Both c3_generate.py and c3_synthesize.py  

### Structured Logs

✓ **Retrieval** - Query, chunks, semantic IDs  
✓ **Synthesis** - Phase, grounding, citations  
✓ **Grounding failure** - Type, reason, details  
✓ **Inference guard** - Guard type, reason  

### Audit Records

✓ **Query tracking** - All queries recorded  
✓ **Refusal tracking** - All refusals logged  
✓ **Citation tracking** - Retrieved vs cited  
✓ **Phase tracking** - C2, C3, C3+  

Phase-7 Observability & Contracts is complete and production-ready with comprehensive structured logging, audit records, and proper exit codes for all operations.
