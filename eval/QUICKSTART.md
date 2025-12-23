# Phase-2 Evaluation Quickstart

## What is Phase-2?

Phase-2 adds **evaluation and safety checks** to the Phase-1 RAG system without modifying any ingestion, chunking, or indexing logic.

## Quick Demo

### 1. Run Retrieval Evaluation

```bash
python scripts/eval_rag.py --verbose
```

This evaluates retrieval quality using 10 test queries in `eval/queries.jsonl`.

**Metrics:**
- Recall@5: % of expected evidence retrieved
- Precision@5: % of retrieved chunks that are relevant
- F1 Score: Harmonic mean of precision and recall

**Pass/Fail:** Exit code 0 if mean recall ≥ 0.7, otherwise 1

### 2. Validate Grounded Generation

```bash
python scripts/validate_grounding.py \
  --answer-file eval/example_answer.txt \
  --chunks 893e4fa2a3c8605d,37f9eb52f8611b85,251098e502a594d7 \
  --verbose
```

This checks that every sentence in the answer has supporting evidence.

**Rules:**
- ✓ Factual claims must have evidence
- ✓ Meta-statements don't need evidence ("Based on the evidence...")
- ✗ Ungrounded sentences trigger refusal

## File Structure

```
eval/
├── queries.jsonl              # 10 test queries with expected evidence
├── expected_sources.jsonl     # Reference evidence chunks
├── example_answer.txt         # Sample answer for grounding validation
├── README.md                  # Data format documentation
└── QUICKSTART.md             # This file

scripts/
├── eval_rag.py               # Retrieval evaluation script
└── validate_grounding.py     # Grounding validation script
```

## Example Output

### Retrieval Evaluation

```
======================================================================
RETRIEVAL EVALUATION REPORT
======================================================================
Total Queries: 10
Top-K: 5

Mean Recall@5:    0.0000
Mean Precision@5: 0.0000
Mean F1 Score:    0.0000

Perfect Recall (1.0): 0/10
Zero Recall (0.0):    10/10

Pass Threshold: 0.70
Status: ✗ FAIL
======================================================================
```

**Note:** Low recall is expected initially because expected chunk IDs are patterns, not exact matches. The fuzzy matching helps but you may need to update `eval/queries.jsonl` with actual chunk IDs from your index.

### Grounding Validation

```
======================================================================
GROUNDED GENERATION VALIDATION REPORT
======================================================================
Total Sentences: 4
Sentences Requiring Evidence: 3
Grounded Sentences: 2
Ungrounded Sentences: 1
Grounding Rate: 66.67%

Valid: False
Should Refuse: True
Refusal Reason: 1 sentence(s) lack supporting evidence

Status: ✗ FAIL
======================================================================
```

## Customization

### Add More Evaluation Queries

```bash
echo '{"query_id": "q11", "query": "What is the short title of the Act?", "expected_chunks": ["section_1"], "expected_acts": ["Minimum Wages Act"], "category": "scope"}' >> eval/queries.jsonl
```

### Adjust Pass Threshold

```bash
python scripts/eval_rag.py --pass-threshold 0.8
```

### Change Top-K

```bash
python scripts/eval_rag.py --top-k 10
```

### Allow Partial Grounding

```bash
python scripts/validate_grounding.py \
  --answer "..." \
  --chunks "..." \
  --allow-partial  # 80% threshold instead of 100%
```

## CI Integration

Both scripts return exit code 0 on pass, 1 on fail - perfect for CI pipelines:

```bash
# In your CI script
python scripts/eval_rag.py --pass-threshold 0.7 || exit 1
```

## Next Steps

1. **Update eval queries** with actual chunk IDs from your index
2. **Set realistic thresholds** based on your use case
3. **Add to CI pipeline** to catch regressions
4. **Expand eval set** as you add more documents

## Full Documentation

See `PHASE2_EVALUATION_README.md` for complete documentation.
