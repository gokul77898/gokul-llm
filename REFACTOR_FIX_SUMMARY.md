# Refactor Fix Summary: Local-Only Model Loading

## Overview

This document summarizes the refactor that permanently removed the non-existent `load_model` API and replaced it with a strict, explicit, local-only model loading interface.

**This system performs 100% local inference.**

---

## What Was Broken

### The Problem
Multiple files imported `load_model` from `src.core`, but this function **never existed**. This was a bug caused by a half-finished refactor that left stale, broken references throughout the codebase.

### Affected Files (Before Fix)
| File | Issue |
|------|-------|
| `src/api/main.py` | Imported non-existent `load_model`, used in 4 locations |
| `src/integrations/langchain_graph.py` | Imported non-existent `load_model`, used in 3 locations |
| `src/pipelines/fusion_pipeline.py` | Imported non-existent `load_model`, used in 3 locations |
| `src/core/__init__.py` | Exported HF-related symbols that should not exist |
| `src/core/model_registry.py` | Contained `HFInferenceClient` remote inference code |

---

## Why `load_model` Was Invalid

1. **No Definition Existed**: The function was never implemented in `src.core`
2. **Stale Reference**: It was a remnant from an incomplete refactor
3. **Would Cause ImportError**: Any attempt to run the code would fail immediately
4. **Mixed Concepts**: The old design conflated local and remote model loading

---

## Why HF Paths Were Removed

1. **This Repository is LOCAL-ONLY**: No Hugging Face Inference API, no remote inference, no API keys
2. **Security**: No external API calls means no credential exposure risks
3. **Determinism**: Local inference is predictable and auditable
4. **Offline Capability**: System works without internet connectivity
5. **Performance**: No network latency for model inference

### Removed Components
- `HFInferenceClient` class
- `get_hf_client()` function
- `HF_API_URL` constant
- `_get_hf_token()` function
- All `requests` imports for HF API calls

---

## Canonical Local-Only Model Loading

### Single Source of Truth

All models are now loaded locally via:

```python
from src.inference.local_models import LocalModelRegistry
```

### Standard Usage Pattern

```python
from src.inference.model_loader import ModelLoader

# Initialize with explicit device
loader = ModelLoader(device="cuda")  # or "cpu"

# Load encoder explicitly
encoder = loader.load_encoder("model-name")

# Load decoder explicitly  
decoder = loader.load_decoder("model-name")
```

### Rules
- Encoder and decoder are loaded **explicitly**
- No magic abstraction
- No global state
- No shared singleton
- Device must be passed **explicitly**

---

## Files Modified

### `src/inference/model_loader.py` (CREATED)
Canonical local-only model loader abstraction. This is the ONLY entry point for model loading.

### `src/core/__init__.py`
- Removed: `get_hf_client`, `HFInferenceClient`, `load_model`
- Kept: `ModelRegistry`, `get_registry`, `ExpertInfo`

### `src/core/model_registry.py`
- Removed: `HFInferenceClient` class (150+ lines)
- Removed: `get_hf_client()` function
- Removed: `HF_API_URL` constant
- Removed: `_get_hf_token()` function
- Removed: `requests` import

### `src/api/main.py`
- Replaced `load_model` import with `ModelLoader`
- Fixed all call sites to use local-only pattern
- Direct `/generate` endpoint now returns 501 (use RAG pipeline instead)

### `src/integrations/langchain_graph.py`
- Replaced `load_model` import with `ModelLoader`
- Updated `MARKLLMWrapper` for local-only operation
- Updated `MARKRetrieverWrapper` for local-only operation

### `src/pipelines/fusion_pipeline.py`
- Replaced `load_model` import with `ModelLoader`
- Pipeline now initializes with lazy model loading
- Models loaded on-demand when GPU available

---

## Validation Results

After refactor:
- ✅ Zero references to `load_model`
- ✅ Zero references to `get_hf_client`
- ✅ Zero references to `HFInferenceClient`
- ✅ Zero references to `HF_API_URL`
- ✅ No ImportError on module load
- ✅ No ModuleNotFoundError
- ✅ No hidden remote calls

---

## Architecture After Refactor

```
src/
├── inference/
│   ├── local_models.py      # LocalModelRegistry (GPU-ready local execution)
│   └── model_loader.py      # ModelLoader (canonical abstraction)
├── core/
│   ├── __init__.py          # Clean exports (no HF)
│   └── model_registry.py    # Expert registry (no HF client)
├── api/
│   └── main.py              # Uses ModelLoader
├── pipelines/
│   └── fusion_pipeline.py   # Uses ModelLoader
└── integrations/
    └── langchain_graph.py   # Uses ModelLoader
```

---

## Constraints Maintained

- ❌ Did NOT touch RAG logic
- ❌ Did NOT touch C3 logic
- ❌ Did NOT modify configs
- ❌ Did NOT add dependencies
- ❌ Did NOT hide errors
- ❌ Did NOT create compatibility wrappers
- ❌ Did NOT add fallback logic

---

## Summary

**Before**: Broken imports, HF remote inference code, non-existent `load_model` function

**After**: Clean imports, 100% local inference, deterministic model loading via `LocalModelRegistry`

This system is now production-safe and auditable with no dead abstractions and no HF dependency anywhere.
