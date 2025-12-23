# Semantic ID Implementation - Complete

## Problem Statement

Evaluation recall was 0% because:
- Expected chunk IDs in queries were semantic patterns (e.g., "section_2_employer")
- Retrieved chunks used hash-based IDs (e.g., "893e4fa2a3c8605d")
- No mapping between the two

## Solution

Introduced a **stable semantic identifier layer** for chunks without replacing hash-based chunk IDs.

## Implementation

### 1. Chunk Generation (`scripts/chunk.py`)

Added `generate_semantic_id()` function that creates human-readable IDs:

**Format:** `<ACT>_<SECTION>_<INDEX>`

**Examples:**
- `IPC_420_0` - IPC Section 420, chunk 0
- `MinimumWagesAct_2_1` - Minimum Wages Act Section 2, chunk 1
- `SupremeCourt_3_2020_2` - Supreme Court case, Section 3, Year 2020, chunk 2

**Key Features:**
- Deterministic (same input → same ID)
- Human-readable
- Sanitized (removes special characters)
- Includes year for case law

**Chunk Data Structure:**
```json
{
  "chunk_id": "893e4fa2a3c8605d",      // Hash ID (unchanged)
  "semantic_id": "MinimumWagesAct_2_1", // NEW: Semantic ID
  "act": "Minimum Wages Act",
  "section": "2",
  "text": "...",
  ...
}
```

### 2. Vector Store (`scripts/index.py`)

Updated ChromaDB metadata to include `semantic_id`:

```python
batch_metadatas.append({
    "chunk_id": chunk_id,
    "semantic_id": chunk.get('semantic_id') or "",  # NEW
    "act": chunk.get('act') or "",
    "section": chunk.get('section') or "",
    ...
})
```

### 3. Query Results (`scripts/query.py`)

Updated `QueryResult` dataclass to expose `semantic_id`:

```python
@dataclass
class QueryResult:
    chunk_id: str
    semantic_id: str  # NEW
    text: str
    score: float
    ...
```

### 4. Evaluation Logic (`scripts/eval_rag.py`)

**REMOVED:** All fuzzy matching logic
**ADDED:** Exact semantic_id matching

```python
def evaluate_query(eval_query, retrieved):
    # Get retrieved semantic IDs (NOT chunk_ids)
    retrieved_ids = {r.semantic_id for r in retrieved if r.semantic_id}
    expected_ids = set(eval_query.expected_chunks)
    
    # EXACT matching only
    true_positives = len(retrieved_ids & expected_ids)
    false_positives = len(retrieved_ids - expected_ids)
    false_negatives = len(expected_ids - retrieved_ids)
    
    recall = true_positives / len(expected_ids) if expected_ids else 0.0
    precision = true_positives / len(retrieved_ids) if retrieved_ids else 0.0
    ...
```

### 5. Evaluation Queries (`eval/queries.jsonl`)

Updated with actual semantic IDs from the system:

**Before:**
```json
{"query_id": "q1", "expected_chunks": ["section_2_employer"], ...}
```

**After:**
```json
{"query_id": "q1", "expected_chunks": ["MinimumWagesAct_2_1"], ...}
```

## Results

### Before Implementation
```
Mean Recall@5:    0.0000
Mean Precision@5: 0.0000
Perfect Recall (1.0): 0/10
Status: ✗ FAIL
```

### After Implementation
```
Mean Recall@5:    1.0000
Mean Precision@5: 0.2400
Perfect Recall (1.0): 10/10
Status: ✓ PASS
```

## Current Semantic IDs in System

```
IPC_420_0                      - IPC Section 420
MinimumWagesAct_1_0            - Minimum Wages Act Section 1
MinimumWagesAct_2_1            - Minimum Wages Act Section 2
MinimumWagesAct_3_2            - Minimum Wages Act Section 3
SupremeCourt_1_2020_0          - Supreme Court case, Section 1
SupremeCourt_2_2020_1          - Supreme Court case, Section 2
SupremeCourt_3_2020_2          - Supreme Court case, Section 3
SupremeCourt_5_2020_3          - Supreme Court case, Section 5
SupremeCourt_6_2020_4          - Supreme Court case, Section 6
SupremeCourt_7_2020_5          - Supreme Court case, Section 7
SupremeCourtofIndia_1_2012_0   - Supreme Court of India case
```

## Key Design Decisions

### ✓ What We Did

1. **Kept hash-based chunk_id** - No breaking changes to existing system
2. **Added semantic_id as separate field** - Clean separation of concerns
3. **Removed fuzzy matching** - Deterministic, predictable evaluation
4. **Used exact matching** - No ambiguity, clear pass/fail
5. **Made semantic IDs human-readable** - Easy to understand and debug

### ✗ What We Avoided

1. **Replacing chunk_id** - Would break existing references
2. **Fuzzy matching hacks** - Non-deterministic, hard to debug
3. **Hardcoding hash IDs** - Not maintainable
4. **Changing retrieval scoring** - Kept Phase-1 logic intact
5. **Regenerating embeddings** - Only rebuilt index with new metadata

## Files Modified

1. `scripts/chunk.py` - Added `generate_semantic_id()` function
2. `scripts/index.py` - Added `semantic_id` to metadata
3. `scripts/query.py` - Added `semantic_id` to QueryResult
4. `scripts/eval_rag.py` - Removed fuzzy matching, use exact semantic_id
5. `eval/queries.jsonl` - Updated with actual semantic IDs

## Usage

### Generate Chunks with Semantic IDs
```bash
python scripts/chunk.py
```

### Index with Semantic IDs
```bash
python scripts/index.py --rebuild
```

### Query (returns semantic_id)
```bash
python scripts/query.py "What is the definition of employer?" --json
```

### Evaluate (uses semantic_id)
```bash
python scripts/eval_rag.py --verbose
```

## Verification

All evaluation queries now show **100% recall**:

```
✓ [1/10] q1: R=1.00 P=0.25
✓ [2/10] q2: R=1.00 P=0.25
✓ [3/10] q3: R=1.00 P=0.25
✓ [4/10] q4: R=1.00 P=0.25
✓ [5/10] q5: R=1.00 P=0.25
✓ [6/10] q6: R=1.00 P=0.20
✓ [7/10] q7: R=1.00 P=0.20
✓ [8/10] q8: R=1.00 P=0.25
✓ [9/10] q9: R=1.00 P=0.25
✓ [10/10] q10: R=1.00 P=0.25
```

## Benefits

1. **Deterministic** - Same input always produces same semantic_id
2. **Human-readable** - Easy to understand what chunk represents
3. **Maintainable** - Easy to update evaluation queries
4. **CI-ready** - Predictable pass/fail behavior
5. **Backward compatible** - Hash-based chunk_id still works

## Future Improvements

1. Add semantic_id to grounding validation
2. Create semantic_id lookup tool
3. Add semantic_id to query results display
4. Document semantic_id format in API responses
