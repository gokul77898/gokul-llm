# Model Routing Documentation

## Overview

The MARK system automatically routes queries between **Mamba** (State Space Model for long contexts) and **Transformer** (BERT/GPT-style for short queries) based on content characteristics.

## Architecture

### Routing Pipeline

```
User Query → Retrieval → Context Analysis → Model Selection → Generation → Response
                              ↓
                    ┌─────────┴──────────┐
                    │  Routing Heuristics │
                    └────────────────────┘
                           ↓
                    ┌──────┴───────┐
                    │    Mamba     │  Long context, multi-page docs
                    │              │  Legal judgments, orders
                    └──────────────┘
                           ↓
                    ┌──────┴───────┐
                    │ Transformer  │  Short queries, single-page
                    │              │  Definitions, simple Q&A
                    └──────────────┘
```

## Routing Heuristics

The system uses three main heuristics to select the appropriate model:

### 1. Token Count Threshold

- **Rule**: If estimated token count ≥ `mamba_threshold_tokens`, use Mamba
- **Default**: 4096 tokens
- **Rationale**: Mamba's state space mechanism handles long sequences efficiently

Example:
```python
# Long multi-page judgment (>4096 tokens) → Mamba
query = "Summarize the Supreme Court judgment on XYZ case"
context = "... 50-page judgment text ..."
# Selected: Mamba
```

### 2. Page Count

- **Rule**: If document spans ≥ `mamba_min_pages`, use Mamba
- **Default**: 3 pages
- **Rationale**: Multi-page documents benefit from Mamba's long-range dependencies

Example:
```python
# 5-page legal order → Mamba
retrieved_docs = [
    {'content': '...', 'metadata': {'page': 1}},
    {'content': '...', 'metadata': {'page': 2}},
    {'content': '...', 'metadata': {'page': 3}},
    # ... pages 4-5
]
# Selected: Mamba
```

### 3. Keyword Matching

- **Rule**: If query contains legal keywords, use Mamba
- **Default Keywords**:
  - judgment
  - order
  - case
  - verdict
  - appellate
  - supreme court

Example:
```python
query = "What was the supreme court verdict?"
# Selected: Mamba (keyword match)
```

## Configuration

### Config File: `configs/model_routing.yaml`

```yaml
model_routing:
  enable_mamba: true              # Master switch
  mamba_threshold_tokens: 4096    # Token threshold
  mamba_min_pages: 3              # Minimum pages for Mamba
  mamba_keywords:                 # Legal keywords
    - judgment
    - order
    - case
    - verdict
    - appellate
    - "supreme court"
  default_model: transformer      # Fallback model
  fallback_to_transformer: true   # Enable fallback on error
```

### Tuning Parameters

#### Increase Mamba Usage

```yaml
mamba_threshold_tokens: 2048  # Lower threshold
mamba_min_pages: 2            # Fewer pages needed
mamba_keywords:               # More keywords
  - judgment
  - order
  - case
  - verdict
  - section
  - act
  - statute
```

#### Decrease Mamba Usage

```yaml
mamba_threshold_tokens: 8192  # Higher threshold
mamba_min_pages: 5            # More pages needed
mamba_keywords: []            # No keyword matching
```

#### Disable Mamba Completely

```yaml
enable_mamba: false
```

Or via environment variable:
```bash
export ENABLE_MAMBA=false
```

## Fallback Handling

### Automatic Fallback

When Mamba is unavailable or fails:

1. System detects Mamba unavailability via `is_model_available('mamba')`
2. If `fallback_to_transformer: true`, routes to Transformer
3. Logs fallback event with reason
4. Returns response indicating fallback occurred

### Mamba Unavailability Scenarios

- Mamba package not installed (`mamba-ssm`)
- Custom Mamba implementation missing
- GPU kernels incompatible (Mac MPS limitations)
- Environment variable `ENABLE_MAMBA=false`
- Config setting `enable_mamba: false`

## Telemetry & Logging

### Routing Log

**Location**: `logs/model_routing.json`

**Format**:
```json
{
  "timestamp": "2025-11-19T11:05:00",
  "query_len_tokens": 5200,
  "retrieved_docs_count": 8,
  "selected_model": "mamba",
  "fallback_used": false,
  "reason": "token_count=5200 >= 4096",
  "page_count": 4,
  "mamba_available": true
}
```

### API Endpoint

**GET** `/api/v1/model/routing_log?limit=100`

Returns recent routing decisions with statistics.

Example:
```bash
curl http://localhost:8000/api/v1/model/routing_log?limit=50
```

Response:
```json
{
  "count": 50,
  "total_logged": 243,
  "entries": [
    {
      "timestamp": "2025-11-19T11:05:00",
      "selected_model": "mamba",
      "reason": "page_count=5 >= 3",
      ...
    }
  ]
}
```

## Training & Checkpoints

### Per-Model LoRA Training

Each model maintains separate LoRA checkpoints:

```
checkpoints/lora/
├── mamba_lora/
│   ├── final/
│   └── checkpoint-500/
└── transformer_lora/
    ├── final/
    └── checkpoint-500/
```

### Training Commands

**Mamba LoRA**:
```bash
python -m src.training.lora_trainer \
  --model mamba \
  --config configs/lora_mamba.yaml \
  --dry-run
```

**Transformer LoRA**:
```bash
python -m src.training.lora_trainer \
  --model transformer \
  --config configs/lora_transformer.yaml \
  --dry-run
```

## Performance Tuning

### Mac MPS Optimization

Mamba on Mac MPS has limitations:
- No custom CUDA kernels
- Fallback to CPU for certain ops
- Slower than CUDA but faster than pure CPU

**Recommendation**: 
- Use Transformer for interactive queries on Mac
- Use Mamba for batch processing or Linux/CUDA systems

### Production Deployment

**Recommended Setup**:
1. Enable Mamba on GPU servers (CUDA)
2. Use Transformer on CPU instances
3. Load balance based on query type
4. Cache long-context Mamba results

## Monitoring

### Key Metrics

- **Routing distribution**: Mamba vs Transformer usage %
- **Fallback rate**: How often Mamba fails
- **Latency by model**: Avg response time
- **Accuracy by model**: User feedback per model

### Dashboard Query

```bash
# Get routing stats
jq '[.[] | .selected_model] | group_by(.) | map({model: .[0], count: length})' \
  logs/model_routing.json
```

Output:
```json
[
  {"model": "mamba", "count": 156},
  {"model": "transformer", "count": 87}
]
```

## Troubleshooting

### Mamba Never Selected

**Check**:
1. `enable_mamba: true` in config
2. `ENABLE_MAMBA` env var not set to false
3. Mamba package installed: `pip install mamba-ssm`
4. No errors in logs: `tail -f logs/auto_selection.json`

### Always Falling Back to Transformer

**Likely Causes**:
1. Mamba import failing
2. GPU unavailable (CUDA required)
3. Checkpoint missing

**Solution**:
- Check `logs/model_routing.json` for `"mamba_available": false`
- Verify installation: `python -c "from src.core.mamba_loader import is_mamba_available; print(is_mamba_available())"`

### High Latency on Mac

**Expected**: Mamba on Mac MPS is slower than CUDA

**Solutions**:
1. Increase `mamba_threshold_tokens` to 8192 (use Mamba less)
2. Use Transformer by default on Mac
3. Deploy Mamba on Linux GPU server

## Best Practices

1. **Start Conservative**: Use default thresholds, monitor routing decisions
2. **Tune Gradually**: Adjust one parameter at a time based on telemetry
3. **Test Fallbacks**: Periodically disable Mamba to verify fallback works
4. **Monitor Performance**: Track latency and accuracy by model
5. **Update Checkpoints**: Retrain LoRA adapters as legal corpus evolves

## Examples

### Short Definition Query

```python
query = "What is a plaint?"
# Token count: ~5 tokens
# Page count: 0
# Keywords: None
# Selected: Transformer ✓
```

### Multi-Page Judgment

```python
query = "Summarize the Supreme Court judgment"
# Token count: ~8000 tokens
# Page count: 6
# Keywords: ["Supreme Court", "judgment"]
# Selected: Mamba ✓
```

### Medium Query (Edge Case)

```python
query = "Explain Section 138 of Negotiable Instruments Act"
# Token count: ~1200 tokens
# Page count: 0
# Keywords: None
# Selected: Transformer (default)
```

## API Integration

No UI changes required. The routing is transparent:

```bash
# Same API endpoint, auto-routing happens internally
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the judgment on XYZ case?",
    "model": "auto",
    "top_k": 5
  }'
```

Response includes routing metadata:
```json
{
  "answer": "...",
  "model": "auto",
  "auto_model_used": "Mamba Hierarchical Attention",
  "metadata": {
    "selected_model": "mamba",
    "word_count": 8,
    "grounded_score": 0.85
  }
}
```

## References

- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [LoRA Fine-Tuning](https://arxiv.org/abs/2106.09685)
- ChromaDB Integration: `docs/CHROMADB_SYSTEM_COMPLETE.md`
- Training Pipeline: `docs/TRAINING_IMPLEMENTATION_SUMMARY.md`
