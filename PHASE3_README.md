# Phase-3: Evidence-Sufficient Synthesis

## Overview

Phase-3 extends the C3 grounded generation system to enable **multi-chunk answer synthesis** while preserving strict grounding guarantees from Phase-2.

**Key Additions:**
1. Evidence sufficiency checking before generation
2. Multi-chunk citation enforcement
3. Claim coverage validation
4. Deterministic synthesis (temperature=0)

## Core Principles

Phase-3 maintains **ALL Phase-2 constraints** and adds:

1. **Evidence Sufficiency** - Check if retrieved chunks can answer the query
2. **Multi-Chunk Synthesis** - Combine information from multiple sources
3. **Complete Citation** - Cite ALL sources used, not just primary source
4. **Claim Coverage** - Every sentence must be covered by cited chunks

## Architecture

```
Query → Retrieve Chunks → Check Sufficiency → Synthesize → Validate → Answer
                              ↓ Insufficient
                            Refuse
```

### Phase-3 Pipeline

```python
1. Retrieve chunks (Phase-1)
2. Check evidence sufficiency (NEW)
   - If insufficient → Force refusal
3. Build synthesis prompt (NEW)
4. Generate answer (deterministic, temp=0)
5. Validate multi-chunk citations (NEW)
6. Check claim coverage (NEW)
7. Return validated result
```

## Evidence Sufficiency Checker

### Function: `is_evidence_sufficient(query, retrieved_chunks)`

Checks if retrieved evidence can answer the query **before** generation.

### Query Type Classification

| Type | Patterns | Requirements |
|------|----------|--------------|
| **definition** | "What is...", "Define...", "What does X mean" | Explicit definitional language required |
| **punishment** | "What is the punishment", "penalty for" | Explicit penalty/punishment information required |
| **procedure** | "How to...", "What are the steps" | Procedural language required |
| **scope** | "extent", "applicability", "applies to" | Scope information required |
| **general** | Other queries | At least one relevant chunk required |

### Sufficiency Rules

**Definition Queries:**
```python
# SUFFICIENT: Has definitional language
"Employer means any person who employs workers."
"Section 2 defines employer as..."

# INSUFFICIENT: No definitional language
"The employer must pay minimum wages."
```

**Punishment Queries:**
```python
# SUFFICIENT: Has penalty information
"Shall be punished with imprisonment for seven years."
"Penalty: Fine up to Rs. 10,000"

# INSUFFICIENT: No penalty information
"Section 420 deals with cheating."
```

**General Queries:**
```python
# SUFFICIENT: Has relevant content
At least 1 chunk with key terms from query

# INSUFFICIENT: No relevant content
No chunks contain query terms
```

### Example Usage

```python
from scripts.c3_synthesize import is_evidence_sufficient

sufficiency = is_evidence_sufficient(query, retrieved_chunks)

if not sufficiency.is_sufficient:
    # Force refusal
    return REFUSAL_MESSAGE
else:
    # Proceed with synthesis
    answer = synthesize_answer(query, retrieved_chunks)
```

## Multi-Chunk Citation Enforcement

### Function: `validate_multi_chunk_citations(answer_text, retrieved_chunks)`

Ensures answer cites **ALL** chunks it uses information from.

### Rules

1. **No Primary Source Shortcuts** - Can't cite only one source when using multiple
2. **Complete Citation** - If info from N chunks, must cite all N semantic_ids
3. **Exact Matching** - Uses semantic_id exact matching (no fuzzy logic)

### Example

```python
# ✓ VALID: Cites all sources used
"According to [IPC_420_0], cheating is illegal. Additionally, 
[MinimumWagesAct_2_1] defines employer as any person who employs workers."

# ✗ INVALID: Uses info from both but only cites one
"According to [IPC_420_0], cheating is illegal. Employer means 
any person who employs workers."  # Missing [MinimumWagesAct_2_1]
```

## Claim Coverage Validation

### Function: `check_claim_coverage(sentence, retrieved_chunks)`

Validates that each sentence/claim is covered by cited chunks.

### Coverage Rules

1. **Cited Sentences** - Must have semantic overlap with cited chunks (≥30%)
2. **Meta-Statements** - Don't require evidence ("Based on the evidence...")
3. **No Citations** - Rejected (unless meta-statement)

### Semantic Overlap

```python
# Calculate overlap
sentence_words = set(sentence.lower().split()) - stop_words
chunk_words = set(chunk.text.lower().split()) - stop_words

overlap = len(sentence_words & chunk_words) / len(sentence_words)

# Coverage threshold
if overlap >= 0.3:  # 30% overlap
    covered = True
```

### Multi-Citation Leniency

For sentences citing multiple sources, threshold is reduced to 21% (0.3 × 0.7):

```python
# Single citation: 30% threshold
"According to [IPC_420_0], cheating is punishable."

# Multiple citations: 21% threshold (more lenient)
"Per [IPC_420_0] and [MinimumWagesAct_2_1], both are regulated."
```

## Deterministic Synthesis

### Temperature = 0

All synthesis uses `temperature=0` for deterministic generation:

```python
response = openai.ChatCompletion.create(
    model=model,
    messages=[...],
    temperature=0.0,  # Deterministic
    max_tokens=500
)
```

### Legal-Style Summarization

Prompt enforces:
- Precise, legal-style language
- No creative phrasing
- No rhetorical language
- Direct quotes or close paraphrases

## Usage

### Basic Synthesis

```bash
# Mock mode (testing)
python scripts/c3_synthesize.py "What is Section 420 IPC?"

# With real LLM
export OPENAI_API_KEY="your-key"
python scripts/c3_synthesize.py "What is Section 420 IPC?" --use-llm

# Verbose output (shows claim coverage details)
python scripts/c3_synthesize.py "What is cheating?" --verbose

# JSON output
python scripts/c3_synthesize.py "What is the minimum wage?" --json
```

### Example Output

```
======================================================================
PHASE-3 C3 EVIDENCE-SUFFICIENT SYNTHESIS
======================================================================

Query: What is Section 420 IPC?

Answer:
According to [IPC_420_0], Section 420 IPC deals with cheating and 
dishonestly inducing delivery of property. The punishment under 
[IPC_420_0] is imprisonment for a term which may extend to seven years 
and fine.

----------------------------------------------------------------------
VALIDATION
----------------------------------------------------------------------
Evidence Sufficient: ✓ YES
Grounded: ✓ YES
Citations: 1
Cited Sources: IPC_420_0
Uncovered Claims: 0
Retrieved Chunks: 5

Status: ✓ PASS
```

## Testing

### Run All Tests

```bash
# Evidence sufficiency tests (18 tests)
python tests/test_c3_sufficiency.py

# Multi-chunk synthesis tests (24 tests)
python tests/test_c3_multi_chunk.py
```

### Test Coverage

**Evidence Sufficiency (18 tests):**
- ✓ Query type classification (definition, punishment, procedure, scope)
- ✓ Sufficient evidence detection
- ✓ Insufficient evidence detection
- ✓ Type-specific requirements
- ✓ Edge cases

**Multi-Chunk Synthesis (24 tests):**
- ✓ Sentence splitting
- ✓ Claim coverage validation
- ✓ Multi-chunk citation enforcement
- ✓ Synthesis integration
- ✓ Evidence sufficiency integration
- ✓ Edge cases

### Test Results

```
======================================================================
PHASE-3 EVIDENCE SUFFICIENCY - TEST SUITE
======================================================================
✓ 18/18 tests PASSED

======================================================================
PHASE-3 MULTI-CHUNK SYNTHESIS - TEST SUITE
======================================================================
✓ 24/24 tests PASSED
```

## API Reference

### is_evidence_sufficient(query, retrieved_chunks)

Check if evidence is sufficient to answer query.

**Args:**
- `query` (str): User query
- `retrieved_chunks` (List[RetrievedChunk]): Retrieved evidence

**Returns:**
- `EvidenceSufficiency`: Sufficiency assessment with reason

**Example:**
```python
sufficiency = is_evidence_sufficient(
    "What is the definition of employer?",
    retrieved_chunks
)

if sufficiency.is_sufficient:
    print(f"Sufficient: {sufficiency.reason}")
else:
    print(f"Insufficient: {sufficiency.reason}")
```

### check_claim_coverage(sentence, retrieved_chunks, min_overlap=0.3)

Check if a sentence is covered by retrieved chunks.

**Args:**
- `sentence` (str): Sentence to check
- `retrieved_chunks` (List[RetrievedChunk]): Retrieved evidence
- `min_overlap` (float): Minimum word overlap ratio (default: 0.3)

**Returns:**
- `ClaimCoverage`: Coverage assessment with covering chunks

**Example:**
```python
coverage = check_claim_coverage(
    "According to [IPC_420_0], cheating is illegal.",
    retrieved_chunks
)

if coverage.is_covered:
    print(f"Covered by: {coverage.covering_chunks}")
else:
    print(f"Not covered: {coverage.reason}")
```

### validate_multi_chunk_citations(answer_text, retrieved_chunks)

Validate that answer cites all chunks it uses.

**Args:**
- `answer_text` (str): Generated answer
- `retrieved_chunks` (List[RetrievedChunk]): Retrieved evidence

**Returns:**
- `Tuple[bool, List[str], str]`: (is_valid, missing_citations, reason)

**Example:**
```python
is_valid, missing, reason = validate_multi_chunk_citations(
    answer_text,
    retrieved_chunks
)

if not is_valid:
    print(f"Missing citations: {missing}")
```

### synthesize_answer(query, retrieved_chunks, use_mock=True, model="gpt-3.5-turbo")

Synthesize answer from multiple chunks with full validation.

**Args:**
- `query` (str): User query
- `retrieved_chunks` (List[RetrievedChunk]): Retrieved evidence
- `use_mock` (bool): Use mock generator (for testing)
- `model` (str): LLM model name

**Returns:**
- `SynthesizedAnswer`: Validated synthesized answer

**Example:**
```python
result = synthesize_answer(
    "What is Section 420 IPC?",
    retrieved_chunks,
    use_mock=False,
    model="gpt-4"
)

if result.is_grounded and result.is_sufficient:
    print(result.answer)
else:
    print(f"Refusal: {result.refusal_reason}")
```

## Integration with Phase-2

Phase-3 extends Phase-2 without breaking changes:

```bash
# Phase-2: Single-chunk generation
python scripts/c3_generate.py "What is Section 420 IPC?"

# Phase-3: Multi-chunk synthesis
python scripts/c3_synthesize.py "What is Section 420 IPC?"
```

Both scripts:
- Use same retrieval system
- Enforce same C3 contract
- Return same output format
- Support same CLI options

## Comparison: Phase-2 vs Phase-3

| Feature | Phase-2 | Phase-3 |
|---------|---------|---------|
| **Evidence checking** | None | Pre-generation sufficiency check |
| **Multi-chunk synthesis** | No | Yes |
| **Citation enforcement** | Valid citations | ALL sources cited |
| **Claim coverage** | Basic | Sentence-level validation |
| **Query type handling** | Generic | Type-specific requirements |
| **Refusal logic** | Post-generation | Pre + post generation |

## Design Decisions

### ✓ What We Did

1. **Pre-generation sufficiency check** - Refuse early if evidence insufficient
2. **Type-specific requirements** - Different rules for definition vs punishment queries
3. **Multi-chunk citation enforcement** - Must cite ALL sources used
4. **Sentence-level coverage** - Validate each claim separately
5. **Deterministic synthesis** - Temperature=0 for reproducibility

### ✗ What We Avoided

1. **Loosening Phase-2 constraints** - All C3 rules still enforced
2. **Fuzzy citation matching** - Exact semantic_id matching only
3. **Partial answer generation** - Either fully grounded or refused
4. **Creative synthesis** - Legal-style only, no rhetorical language
5. **Retry mechanisms** - Single-pass generation

## Exit Codes

- `0`: Answer is grounded and sufficient
- `1`: Answer is not grounded or insufficient

## Future Enhancements

1. **Cross-chunk reasoning** - Support reasoning across multiple chunks
2. **Confidence scores** - Add confidence to sufficiency assessments
3. **Explanation generation** - Explain why evidence is sufficient/insufficient
4. **Interactive refinement** - Allow user to request more evidence
5. **Batch synthesis** - Synthesize answers for multiple queries

## References

- Phase-1 RAG: `PHASE1_README.md`
- Phase-2 C3: `C3_GROUNDED_GENERATION.md`
- Semantic IDs: `SEMANTIC_ID_IMPLEMENTATION.md`
- Evaluation: `PHASE2_EVALUATION_README.md`
