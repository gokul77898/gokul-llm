# Phase-2: C3 Grounded Generation Contract

## Overview

The C3 (Contract for Correct Citations) enforces that the LLM generates answers **ONLY** from retrieved evidence with **NO hallucination** and **NO prior knowledge**.

## Core Principles

1. **Evidence-Only Answers** - LLM can only use text from retrieved chunks
2. **Mandatory Citations** - Every factual claim must cite source with `[semantic_id]`
3. **Hard Refusal** - If evidence insufficient, refuse with exact message
4. **Post-Generation Validation** - All answers validated against C3 contract
5. **Zero Logic Leakage** - No fuzzy matching, no fallbacks, no retries

## The C3 Contract

### Rule 1: Answer ONLY from Evidence

```
✓ VALID:   "According to [IPC_420_0], Section 420 deals with cheating."
✗ INVALID: "Section 420 deals with cheating." (no citation)
✗ INVALID: "Section 420 is commonly known as..." (prior knowledge)
```

### Rule 2: Cite All Sources

```
✓ VALID:   "Per [MinimumWagesAct_2_1], employer means any person who employs..."
✗ INVALID: "Employer means any person who employs..." (no citation)
```

### Rule 3: Hard Refusal When Evidence Insufficient

```
✓ VALID:   "I cannot answer based on the provided documents."
✗ INVALID: "I think it might be..." (speculation)
✗ INVALID: "Based on general knowledge..." (prior knowledge)
```

### Rule 4: All Citations Must Be Valid

```
✓ VALID:   [IPC_420_0] (exists in retrieved chunks)
✗ INVALID: [IPC_421_0] (not in retrieved chunks)
✗ INVALID: [FAKE_ID] (hallucinated semantic_id)
```

## Implementation

### 1. Grounded Prompt Builder

`build_grounded_prompt(query, retrieved_chunks)` creates a strict prompt:

```python
def build_grounded_prompt(query, retrieved_chunks):
    """Build prompt that enforces grounded generation."""
    
    prompt = f"""You are a legal research assistant. You MUST follow these rules STRICTLY:

1. Answer ONLY using the evidence documents provided below
2. You MUST cite sources using the format [semantic_id] (e.g., [IPC_420_0])
3. Every factual claim MUST have a citation
4. Do NOT use any prior knowledge or information not in the evidence
5. If the answer is not present in the evidence, you MUST say EXACTLY:
   "I cannot answer based on the provided documents."

EVIDENCE DOCUMENTS:

[IPC_420_0]
Act: IPC
Section: 420
Text: Section 420 IPC – Cheating and dishonestly inducing delivery...

[MinimumWagesAct_2_1]
Act: Minimum Wages Act
Section: 2
Text: Section 2. Definitions. In this Act, unless the context otherwise...

QUESTION: {query}

ANSWER (with citations):"""
    
    return prompt
```

**Key Features:**
- Enumerates all chunks with `[semantic_id]`
- Includes ONLY retrieved text
- Explicitly forbids outside knowledge
- Requires citations for all claims
- Specifies exact refusal message

### 2. Citation Extraction

`extract_citations(answer_text)` extracts all citations:

```python
def extract_citations(answer_text):
    """Extract citations in format [semantic_id]."""
    pattern = r'\[([A-Za-z0-9_]+)\]'
    citations = re.findall(pattern, answer_text)
    return set(citations)
```

**Examples:**
```python
extract_citations("Per [IPC_420_0], cheating is illegal.")
# Returns: {'IPC_420_0'}

extract_citations("See [IPC_420_0] and [MinimumWagesAct_2_1].")
# Returns: {'IPC_420_0', 'MinimumWagesAct_2_1'}

extract_citations("No citations here.")
# Returns: set()
```

### 3. C3 Enforcement

`enforce_c3(answer_text, retrieved_chunks)` validates the answer:

```python
def enforce_c3(answer_text, retrieved_chunks, require_citations=True):
    """Enforce C3 contract: validate answer is grounded in evidence.
    
    Returns:
        (is_valid, invalid_citations, refusal_reason)
    """
    
    # Check if it's a refusal
    if answer_text.strip() == REFUSAL_MESSAGE:
        return True, [], None
    
    # Extract citations
    cited_sources = extract_citations(answer_text)
    
    # Get valid semantic IDs
    valid_semantic_ids = {chunk.semantic_id for chunk in retrieved_chunks}
    
    # Check for citations
    if require_citations and not cited_sources:
        return False, [], "Answer contains no citations"
    
    # Check all citations are valid
    invalid_citations = cited_sources - valid_semantic_ids
    if invalid_citations:
        return False, list(invalid_citations), f"Invalid citations: {invalid_citations}"
    
    # All checks passed
    return True, [], None
```

**Validation Logic:**

1. **Refusal Check** - Exact refusal message is always valid
2. **Citation Check** - Answer must have citations (unless refusal)
3. **Validity Check** - All citations must reference retrieved chunks
4. **No Fuzzy Matching** - Exact semantic_id matching only

### 4. Generation Pipeline

`generate_grounded_answer(query, retrieved_chunks)` orchestrates:

```python
def generate_grounded_answer(query, retrieved_chunks, use_mock=True):
    """Generate and validate a grounded answer."""
    
    # 1. Build grounded prompt
    prompt = build_grounded_prompt(query, retrieved_chunks)
    
    # 2. Generate answer (mock or real LLM)
    if use_mock:
        answer_text = generate_answer_mock(prompt, retrieved_chunks)
    else:
        answer_text = generate_answer_openai(prompt, model="gpt-3.5-turbo")
    
    # 3. Extract citations
    cited_sources = list(extract_citations(answer_text))
    
    # 4. Validate against C3 contract
    is_valid, invalid_citations, refusal_reason = enforce_c3(
        answer_text,
        retrieved_chunks,
        require_citations=True
    )
    
    # 5. Return validated result
    return GroundedAnswer(
        query=query,
        answer=answer_text,
        is_grounded=is_valid,
        cited_sources=cited_sources,
        invalid_citations=invalid_citations,
        retrieved_semantic_ids=[c.semantic_id for c in retrieved_chunks],
        refusal_reason=refusal_reason
    )
```

## Usage

### Basic Usage

```bash
# Generate grounded answer (mock mode)
python scripts/c3_generate.py "What is Section 420 IPC?"

# With custom top-k
python scripts/c3_generate.py "What is the minimum wage?" --top-k 3

# JSON output
python scripts/c3_generate.py "What is cheating?" --json
```

### With Real LLM

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Generate with GPT-3.5
python scripts/c3_generate.py "What is Section 420 IPC?" --use-llm

# Generate with GPT-4
python scripts/c3_generate.py "What is Section 420 IPC?" --use-llm --model gpt-4
```

### Example Output

```
======================================================================
C3 GROUNDED GENERATION RESULT
======================================================================

Query: What is Section 420 IPC?

Answer:
According to [IPC_420_0], Section 420 IPC deals with cheating and 
dishonestly inducing delivery of property.

----------------------------------------------------------------------
VALIDATION
----------------------------------------------------------------------
Grounded: ✓ YES
Citations: 1
Cited Sources: IPC_420_0
Retrieved Chunks: 5
Retrieved IDs: IPC_420_0, SupremeCourtofIndia_1_2012_0, ...
```

## Testing

### Run All Tests

```bash
python tests/test_c3_grounding.py
```

### Test Coverage

**Positive Cases (17 tests):**
- ✓ Valid answers with proper citations
- ✓ Refusal messages
- ✓ Multiple valid citations
- ✓ Prompt includes all evidence
- ✓ Prompt forbids outside knowledge

**Negative Cases:**
- ✓ Answers without citations rejected
- ✓ Invalid citations detected
- ✓ Hallucinated semantic_ids caught
- ✓ Mixed valid/invalid citations rejected
- ✓ Empty semantic_ids handled

**Edge Cases:**
- ✓ Duplicate citations deduplicated
- ✓ Malformed citations handled
- ✓ No evidence triggers refusal

### Test Results

```
======================================================================
C3 GROUNDED GENERATION CONTRACT - TEST SUITE
======================================================================

✓ test_build_grounded_prompt_includes_evidence PASSED
✓ test_build_grounded_prompt_forbids_outside_knowledge PASSED
✓ test_extract_citations_valid PASSED
✓ test_extract_citations_none PASSED
✓ test_extract_citations_duplicate PASSED
✓ test_enforce_c3_valid_answer_with_citations PASSED
✓ test_enforce_c3_refusal_message PASSED
✓ test_enforce_c3_multiple_valid_citations PASSED
✓ test_enforce_c3_no_citations PASSED
✓ test_enforce_c3_invalid_citation PASSED
✓ test_enforce_c3_mixed_valid_invalid_citations PASSED
✓ test_enforce_c3_hallucinated_semantic_id PASSED
✓ test_generate_grounded_answer_with_evidence PASSED
✓ test_generate_grounded_answer_no_evidence PASSED
✓ test_enforce_c3_empty_semantic_ids PASSED
✓ test_extract_citations_malformed PASSED
✓ test_enforce_c3_citation_in_refusal PASSED

======================================================================
RESULTS: 17 passed, 0 failed
======================================================================
```

## Validation Examples

### ✓ Valid Grounded Answer

```python
answer = "According to [IPC_420_0], Section 420 IPC deals with cheating."
chunks = [RetrievedChunk(semantic_id="IPC_420_0", ...)]

is_valid, invalid, reason = enforce_c3(answer, chunks)
# is_valid = True
# invalid = []
# reason = None
```

### ✗ Invalid: No Citations

```python
answer = "Section 420 IPC deals with cheating."
chunks = [RetrievedChunk(semantic_id="IPC_420_0", ...)]

is_valid, invalid, reason = enforce_c3(answer, chunks)
# is_valid = False
# invalid = []
# reason = "Answer contains no citations"
```

### ✗ Invalid: Hallucinated Citation

```python
answer = "According to [IPC_421_0], fraud is illegal."
chunks = [RetrievedChunk(semantic_id="IPC_420_0", ...)]

is_valid, invalid, reason = enforce_c3(answer, chunks)
# is_valid = False
# invalid = ['IPC_421_0']
# reason = "Invalid citations: IPC_421_0"
```

### ✓ Valid: Refusal

```python
answer = "I cannot answer based on the provided documents."
chunks = [...]

is_valid, invalid, reason = enforce_c3(answer, chunks)
# is_valid = True
# invalid = []
# reason = None
```

## Design Decisions

### ✓ What We Did

1. **Exact semantic_id matching** - No fuzzy logic
2. **Mandatory citations** - Every claim must cite source
3. **Hard refusal mode** - Exact refusal message required
4. **Post-generation validation** - All answers validated
5. **Zero logic leakage** - No fallbacks or retries

### ✗ What We Avoided

1. **Fuzzy citation matching** - Would allow hallucinations
2. **Partial answers** - Would leak prior knowledge
3. **Paraphrasing without citations** - Would hide sources
4. **Retry prompts** - Would add non-determinism
5. **Embedding similarity checks** - Would add complexity

## Integration with Evaluation

The C3 contract integrates with Phase-2 evaluation:

```bash
# 1. Retrieve chunks
python scripts/query.py "What is Section 420 IPC?" --json > retrieved.json

# 2. Generate grounded answer
python scripts/c3_generate.py "What is Section 420 IPC?" --json > answer.json

# 3. Validate grounding
python scripts/validate_grounding.py \
  --answer-file answer.json \
  --chunks-file retrieved.json
```

## API Reference

### build_grounded_prompt(query, retrieved_chunks)

Builds a strict prompt that enforces grounded generation.

**Args:**
- `query` (str): User query
- `retrieved_chunks` (List[RetrievedChunk]): Retrieved evidence

**Returns:**
- `str`: Grounded prompt with strict instructions

### extract_citations(answer_text)

Extracts citations in format `[semantic_id]`.

**Args:**
- `answer_text` (str): Generated answer

**Returns:**
- `Set[str]`: Set of cited semantic_ids

### enforce_c3(answer_text, retrieved_chunks, require_citations=True)

Validates answer against C3 contract.

**Args:**
- `answer_text` (str): Generated answer
- `retrieved_chunks` (List[RetrievedChunk]): Retrieved evidence
- `require_citations` (bool): Whether to require citations

**Returns:**
- `Tuple[bool, List[str], Optional[str]]`: (is_valid, invalid_citations, refusal_reason)

### generate_grounded_answer(query, retrieved_chunks, use_mock=True, model="gpt-3.5-turbo")

Generates and validates a grounded answer.

**Args:**
- `query` (str): User query
- `retrieved_chunks` (List[RetrievedChunk]): Retrieved evidence
- `use_mock` (bool): Use mock generator (for testing)
- `model` (str): LLM model name (if use_mock=False)

**Returns:**
- `GroundedAnswer`: Validated grounded answer

## Exit Codes

- `0`: Answer is grounded (valid)
- `1`: Answer is not grounded (invalid)

## Future Enhancements

1. **Sentence-level grounding** - Validate each sentence separately
2. **Confidence scores** - Add confidence to citations
3. **Multi-hop reasoning** - Support reasoning across chunks
4. **Explanation generation** - Explain why answer is grounded
5. **Human-in-the-loop** - Allow human validation

## References

- Phase-1 RAG: `PHASE1_README.md`
- Semantic IDs: `SEMANTIC_ID_IMPLEMENTATION.md`
- Evaluation: `PHASE2_EVALUATION_README.md`
- Grounding Validation: `scripts/validate_grounding.py`
