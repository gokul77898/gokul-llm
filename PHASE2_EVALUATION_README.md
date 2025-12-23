# Phase-2 RAG: Evaluation and Safety Checks

Deterministic evaluation and grounded generation validation for the Phase-1 RAG system.

## Overview

Phase-2 adds evaluation and safety checks **without modifying** Phase-1 ingestion, chunking, or indexing logic.

**Goals:**
1. Measure retrieval quality (Recall@k, Precision@k)
2. Validate grounded generation (no hallucinations)
3. Provide CI-ready pass/fail reports

## Directory Structure

```
eval/
├── queries.jsonl              # Test queries with expected evidence
├── expected_sources.jsonl     # Reference evidence chunks
└── README.md                  # Evaluation data format

scripts/
├── eval_rag.py               # Retrieval evaluation
├── validate_grounding.py     # Grounded generation validator
└── c3_generate.py            # C3 grounded answer generation
```

## Quick Start

### 1. Run Retrieval Evaluation

```bash
# Basic evaluation
python scripts/eval_rag.py

# With verbose output
python scripts/eval_rag.py --verbose

# Save report to file
python scripts/eval_rag.py --output eval/results.json

# Adjust top-k
python scripts/eval_rag.py --top-k 10

# Set pass threshold
python scripts/eval_rag.py --pass-threshold 0.8
```

**Output:**
- Mean Recall@k, Precision@k, F1 score
- Query-level results
- Missing/incorrect evidence
- Pass/fail status (exit code 0 = pass, 1 = fail)

### 2. Validate Grounded Generation

```bash
# Validate an answer against retrieved chunks
python scripts/validate_grounding.py \
  --answer "The Minimum Wages Act extends to the whole of India." \
  --chunks chunk_id_1,chunk_id_2,chunk_id_3

# From files
python scripts/validate_grounding.py \
  --answer-file answer.txt \
  --chunks-file chunks.json

# With verbose output
python scripts/validate_grounding.py \
  --answer "..." \
  --chunks "..." \
  --verbose

# Allow partial grounding (80% threshold)
python scripts/validate_grounding.py \
  --answer "..." \
  --chunks "..." \
  --allow-partial
```

**Output:**
- Sentence-level grounding analysis
- Supporting chunks for each sentence
- Grounding rate
- Refusal recommendation

## Evaluation Metrics

### Retrieval Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Recall@k** | Fraction of expected chunks retrieved | TP / (TP + FN) |
| **Precision@k** | Fraction of retrieved chunks that are relevant | TP / (TP + FP) |
| **F1 Score** | Harmonic mean of precision and recall | 2 × (P × R) / (P + R) |

Where:
- TP = True Positives (expected chunks retrieved)
- FP = False Positives (unexpected chunks retrieved)
- FN = False Negatives (expected chunks not retrieved)

### Grounding Metrics

| Metric | Description |
|--------|-------------|
| **Grounding Rate** | % of sentences with supporting evidence |
| **Ungrounded Sentences** | Count of sentences lacking evidence |
| **Should Refuse** | Whether answer should be refused |

## Evaluation Data Format

### queries.jsonl

Each line is a JSON object:

```json
{
  "query_id": "q1",
  "query": "What is the definition of employer?",
  "expected_chunks": ["section_2_employer"],
  "expected_acts": ["Minimum Wages Act"],
  "category": "definition"
}
```

Fields:
- `query_id`: Unique identifier
- `query`: Question text
- `expected_chunks`: List of expected chunk ID patterns (fuzzy matched)
- `expected_acts`: Acts that should appear in results
- `category`: Query type (definition, procedure, scope, etc.)

### expected_sources.jsonl

Reference data for expected evidence:

```json
{
  "chunk_id": "section_2_employer",
  "act": "Minimum Wages Act",
  "section": "2",
  "text_snippet": "employer means any person who employs",
  "relevance": "definition of employer"
}
```

## Grounded Generation Validation

### Rules

1. **Every factual sentence must have supporting evidence**
2. **Meta-statements don't require evidence** (e.g., "Based on the evidence...")
3. **Minimum word overlap** required (default: 30%)
4. **Refusal enforced** if evidence is missing

### Sentence Classification

**Requires Evidence:**
- "The Minimum Wages Act extends to the whole of India."
- "An employer is defined as any person who employs workers."
- "The review period is five years."

**Meta-statements (No Evidence Required):**
- "Based on the evidence provided..."
- "According to the retrieved documents..."
- "I cannot answer this question because..."
- "The evidence does not contain information about..."

### Example Validation

```python
# Good: All sentences grounded
answer = """
Based on the evidence, the Minimum Wages Act extends to the whole of India.
An employer is defined as any person who employs workers.
"""
# Result: PASS (meta-statement + grounded sentence)

# Bad: Ungrounded claim
answer = """
The Minimum Wages Act was enacted in 1950.
"""
# Result: FAIL (no evidence for 1950, actual year is 1948)
```

## CI Integration

### GitHub Actions Example

```yaml
name: RAG Evaluation

on: [push, pull_request]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run Phase-1 Pipeline
        run: |
          python scripts/ingest.py
          python scripts/chunk.py
          python scripts/index.py
      
      - name: Run Evaluation
        run: |
          python scripts/eval_rag.py --pass-threshold 0.7
      
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: eval-results
          path: eval/results.json
```

### Exit Codes

- `0`: Evaluation passed (recall >= threshold)
- `1`: Evaluation failed (recall < threshold)

## Adding New Evaluation Queries

1. **Identify the question** you want to test
2. **Manually find expected evidence** in your documents
3. **Add to queries.jsonl**:

```bash
echo '{"query_id": "q11", "query": "Your question?", "expected_chunks": ["pattern"], "expected_acts": ["Act Name"], "category": "type"}' >> eval/queries.jsonl
```

4. **Run evaluation** to verify

## Fuzzy Matching

By default, fuzzy matching is enabled to handle:
- Chunk ID variations
- Act-level matching when exact IDs unknown
- Text snippet matching

**Disable fuzzy matching:**
```bash
python scripts/eval_rag.py --no-fuzzy
```

## Configuration

All evaluation uses the same config as Phase-1:

```yaml
# configs/phase1_rag.yaml
encoder:
  model_name: "BAAI/bge-large-en-v1.5"

retrieval:
  top_k: 5
  collection_name: "legal_chunks"
```

## Example Evaluation Run

```bash
$ python scripts/eval_rag.py --verbose

======================================================================
Phase-2 RAG: Retrieval Evaluation
======================================================================
[2025-01-15 10:00:00] Config: configs/phase1_rag.yaml
[2025-01-15 10:00:00] Version: 1.0.0
[2025-01-15 10:00:00] Queries file: eval/queries.jsonl
[2025-01-15 10:00:00] Top-K: 5
[2025-01-15 10:00:00] Pass threshold: 0.7

[2025-01-15 10:00:01] Loading evaluation queries...
[2025-01-15 10:00:01] Loaded 10 evaluation queries

[2025-01-15 10:00:02] Initializing ChromaDB...
[2025-01-15 10:00:10] Embedding model loaded
[2025-01-15 10:00:10] Loaded collection with 43 chunks

[2025-01-15 10:00:10] Running evaluation...
[2025-01-15 10:00:11] ✓ [1/10] q1: R=1.00 P=0.80
[2025-01-15 10:00:12] ✓ [2/10] q2: R=1.00 P=0.80
[2025-01-15 10:00:13] ✓ [3/10] q3: R=1.00 P=1.00
...

======================================================================
RETRIEVAL EVALUATION REPORT
======================================================================
Timestamp: 2025-01-15T10:00:20.123456
Total Queries: 10
Top-K: 5

Mean Recall@5:    0.9000
Mean Precision@5: 0.8500
Mean F1 Score:    0.8750

Perfect Recall (1.0): 8/10
Zero Recall (0.0):    0/10

Pass Threshold: 0.70
Status: ✓ PASS
======================================================================
```

## Example Grounding Validation

```bash
$ python scripts/validate_grounding.py \
  --answer "The Minimum Wages Act extends to the whole of India." \
  --chunks 893e4fa2a3c8605d \
  --verbose

======================================================================
Phase-2 RAG: Grounded Generation Validator
======================================================================
[2025-01-15 10:05:00] Answer length: 58 characters
[2025-01-15 10:05:00] Loaded 1 chunks

[2025-01-15 10:05:00] Validating grounding...

======================================================================
GROUNDED GENERATION VALIDATION REPORT
======================================================================
Timestamp: 2025-01-15T10:05:01.123456

Total Sentences: 1
Sentences Requiring Evidence: 1
Grounded Sentences: 1
Ungrounded Sentences: 0
Grounding Rate: 100.00%

Valid: True
Should Refuse: False

Status: ✓ PASS
======================================================================

SENTENCE-LEVEL RESULTS:
----------------------------------------------------------------------

✓ Sentence 1:
   The Minimum Wages Act extends to the whole of India.
   Grounded: True
   Supporting: 893e4fa2a3c8605d
   Reason: Found 1 supporting chunk(s) with 45% overlap
```

## Troubleshooting

### Low Recall Scores

**Causes:**
- Expected chunk IDs don't match actual IDs
- Insufficient retrieval (increase top-k)
- Poor embedding quality

**Solutions:**
1. Use fuzzy matching (enabled by default)
2. Specify `expected_acts` instead of exact chunk IDs
3. Increase `--top-k` parameter
4. Review actual chunk IDs: `ls data/rag/chunks/`

### False Positives in Grounding

**Causes:**
- Overlap threshold too low
- Common words causing false matches

**Solutions:**
1. Increase `--min-overlap` (default: 0.3)
2. Use `require_all_grounded` mode (default)
3. Review sentence-level results with `--verbose`

### Evaluation Takes Too Long

**Causes:**
- Large number of queries
- Slow embedding model

**Solutions:**
1. Reduce number of queries in eval set
2. Use smaller embedding model (update config)
3. Run on GPU if available

## Best Practices

1. **Start with small eval set** (5-10 queries) and expand
2. **Use fuzzy matching** for robustness
3. **Set realistic thresholds** (0.7 is reasonable for most systems)
4. **Run evaluation in CI** to catch regressions
5. **Validate grounding** before deploying answers
6. **Review failures** to improve retrieval

## Grounded Generation (C3)

Phase-2 includes the **C3 (Contract for Correct Citations)** grounded generation system that ensures LLM answers are strictly evidence-based.

### C3 Quick Start

```bash
# Generate grounded answer (mock mode)
python scripts/c3_generate.py "What is Section 420 IPC?"

# With real LLM (requires OpenAI API key)
export OPENAI_API_KEY="your-key"
python scripts/c3_generate.py "What is Section 420 IPC?" --use-llm

# JSON output
python scripts/c3_generate.py "What is cheating?" --json
```

### C3 Contract Rules

1. **Answer ONLY from evidence** - No prior knowledge allowed
2. **Cite all sources** - Format: `[semantic_id]` (e.g., `[IPC_420_0]`)
3. **Hard refusal** - If evidence insufficient, return exact refusal message
4. **Post-generation validation** - All answers validated against retrieved chunks

### Example C3 Output

```
Query: What is Section 420 IPC?

Answer:
According to [IPC_420_0], Section 420 IPC deals with cheating and 
dishonestly inducing delivery of property.

VALIDATION
----------
Grounded: ✓ YES
Citations: 1
Cited Sources: IPC_420_0
```

### C3 Testing

```bash
# Run all C3 tests (17 tests)
python tests/test_c3_grounding.py
```

**Test Coverage:**
- ✓ Valid answers with citations
- ✓ Refusal messages
- ✓ Invalid citation detection
- ✓ Hallucination prevention
- ✓ No-citation rejection

See `C3_GROUNDED_GENERATION.md` for complete C3 documentation.

## Phase-2 Scope

**Included:**
- Retrieval evaluation (Recall@k, Precision@k)
- Grounded generation validation
- C3 grounded answer generation
- CI-ready pass/fail reports
- Deterministic, script-based evaluation

**Not Included (Future Phases):**
- Human evaluation
- A/B testing
- Production monitoring
