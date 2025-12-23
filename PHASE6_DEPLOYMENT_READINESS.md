# Phase-6: Deployment Readiness

## Overview

Phase-6 implements **deployment readiness** features including config loading with environment overrides, system health checks, and startup validation. The system can boot **without GPU or models**.

**Key Principle:** Validate system health on startup before processing requests.

## Architecture

### Startup Flow

```
Script Start
    ↓
Load Config (YAML + ENV overrides)
    ↓
Validate Config (required keys)
    ↓
Health Check
    ├─ Config loaded ✓
    ├─ Retriever ready ✓
    ├─ Decoder registered ✓
    └─ Inference status ✓
    ↓
Health OK? → Continue
Health FAIL? → Exit with error
```

## Implementation

### 1. Config Loader (`src/runtime/config_loader.py`)

Loads configuration from YAML with environment variable overrides.

#### Priority Order

1. **Environment variables** (highest)
2. **YAML file**
3. **Defaults** (lowest)

#### Environment Variable Mappings

| Environment Variable | Config Path | Type |
|---------------------|-------------|------|
| `RAG_ENCODER_MODEL` | `encoder.model_name` | string |
| `RAG_DECODER_TYPE` | `decoder.type` | string |
| `RAG_DECODER_MODEL` | `decoder.model_name` | string |
| `RAG_DECODER_DEVICE` | `decoder.device` | string |
| `RAG_ENABLE_INFERENCE` | `decoder.enable_inference` | boolean |
| `RAG_COLLECTION_NAME` | `retrieval.collection_name` | string |
| `RAG_TOP_K` | `retrieval.top_k` | integer |
| `CHROMADB_DIR` | `paths.chromadb_dir` | string |

#### Usage

```python
from src.runtime import load_config

# Load with env overrides and validation
config = load_config('configs/phase1_rag.yaml')

# Load without validation (for testing)
config = load_config('configs/phase1_rag.yaml', validate=False)

# Load without env overrides
config = load_config('configs/phase1_rag.yaml', apply_env=False)
```

#### Config Validation

**Required Top-Level Keys:**
- `paths`
- `encoder`
- `decoder`
- `retrieval`

**Required Nested Keys:**
- `paths`: `raw_dir`, `documents_dir`, `chunks_dir`, `chromadb_dir`
- `encoder`: `model_name`, `embedding_dim`
- `decoder`: `type`, `enable_inference`
- `retrieval`: `top_k`, `collection_name`

**Validation Example:**

```python
from src.runtime import ConfigLoader

errors = ConfigLoader.validate_config(config)

if errors:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
```

### 2. Health Check (`src/runtime/health.py`)

Comprehensive system health check for deployment readiness.

#### Health Checks

**Check 1: Config Loaded**
- Validates config is not None
- Validates config is a dictionary
- Validates config is not empty

**Check 2: Retriever Ready**
- Validates ChromaDB directory configured
- Checks if directory path is valid

**Check 3: Decoder Registered**
- Validates decoder configured
- Validates decoder type supported
- Attempts decoder instantiation

**Check 4: Inference Status**
- Reports inference enabled/disabled
- Warns if guards will block inference
- Warns if large model on CPU

#### Usage

```python
from src.runtime import health_check, print_health_check

# Run health check
result = health_check(config)

# Print result
print_health_check(result, verbose=True)

# Check if healthy
if result.healthy:
    print("System ready")
else:
    print("System not ready")
    for error in result.errors:
        print(f"  Error: {error}")
```

#### HealthCheckResult

```python
@dataclass
class HealthCheckResult:
    healthy: bool                    # Overall health status
    checks: Dict[str, bool]          # Individual check results
    errors: List[str]                # Error messages
    warnings: List[str]              # Warning messages
    info: Dict[str, Any]             # System info
```

### 3. Startup Integration

Both entrypoint scripts (`c3_generate.py` and `c3_synthesize.py`) now run health checks on startup.

**Startup Sequence:**

1. Load config from YAML
2. Apply environment overrides
3. Validate config
4. Run health check
5. Exit if unhealthy
6. Continue if healthy

**Example Output:**

```
[2025-12-21 18:44:00] Running system health check...
[2025-12-21 18:44:00] Health check passed
======================================================================
Phase-2 RAG: C3 Grounded Generation
======================================================================
```

## Environment Variable Overrides

### Example: Override Decoder Type

```bash
# Config file has decoder.type = "qwen"
# Override to stub mode
export RAG_DECODER_TYPE=stub

python scripts/c3_generate.py "query"
# Uses stub decoder instead of qwen
```

### Example: Enable Inference via Environment

```bash
# Config file has enable_inference = false
# Override to enable
export RAG_ENABLE_INFERENCE=true
export ALLOW_LLM_INFERENCE=1

python scripts/c3_generate.py "query" --use-llm
# Inference enabled (but still blocked by Phase-5 guards if other checks fail)
```

### Example: Override Collection Name

```bash
# Use different ChromaDB collection
export RAG_COLLECTION_NAME=test_collection

python scripts/c3_generate.py "query"
# Uses test_collection instead of default
```

## Health Check Scenarios

### Scenario 1: Healthy System

```yaml
# Config
decoder:
  type: qwen
  enable_inference: false
```

**Health Check Output:**
```
======================================================================
SYSTEM HEALTH CHECK
======================================================================

Overall Status: ✓ HEALTHY

Info:
  collection_name: legal_chunks
  decoder_type: qwen
  inference_status: DISABLED (safe stub mode)
```

### Scenario 2: Missing Config Keys

```yaml
# Config missing decoder section
encoder:
  model_name: "BAAI/bge-large-en-v1.5"
```

**Health Check Output:**
```
======================================================================
SYSTEM HEALTH CHECK
======================================================================

Overall Status: ✗ UNHEALTHY

Checks:
  ✗ config_loaded

Errors:
  ✗ Config: Missing required top-level key: decoder
```

### Scenario 3: Inference Enabled with Warnings

```yaml
# Config
decoder:
  type: qwen
  model_name: "Qwen/Qwen2.5-32B-Instruct"
  device: cpu
  enable_inference: true
```

**Without ALLOW_LLM_INFERENCE:**
```
======================================================================
SYSTEM HEALTH CHECK
======================================================================

Overall Status: ✓ HEALTHY

Info:
  inference_status: ENABLED

Warnings:
  ⚠ Inference enabled in config but ALLOW_LLM_INFERENCE env var not set.
    Runtime guards will block inference.
  ⚠ Large model Qwen/Qwen2.5-32B-Instruct configured with device=cpu.
    Runtime guards will block inference. Use device=cuda or device=mps.
```

### Scenario 4: Decoder Not Available

```yaml
# Config
decoder:
  type: invalid_type
```

**Health Check Output:**
```
======================================================================
SYSTEM HEALTH CHECK
======================================================================

Overall Status: ✗ UNHEALTHY

Checks:
  ✓ config_loaded
  ✓ retriever_ready
  ✗ decoder_registered

Errors:
  ✗ Decoder: Unsupported decoder type: invalid_type
```

## Boot Without GPU or Models

Phase-6 ensures the system can boot **without GPU or models** by:

1. **Stub mode by default** - No model loading
2. **Config validation only** - No actual model instantiation
3. **Health checks** - Validate configuration, not hardware
4. **Graceful degradation** - Warnings instead of errors

### Example: Boot on CPU-Only Machine

```yaml
# Config
decoder:
  type: qwen
  enable_inference: false  # Stub mode
  device: cpu
```

```bash
# Boot on machine without GPU
python scripts/c3_generate.py "query"
```

**Result:** ✓ Boots successfully, returns C3 refusal (stub mode)

### Example: Boot Without Models Downloaded

```yaml
# Config references model not downloaded
decoder:
  type: qwen
  model_name: "Qwen/Qwen2.5-32B-Instruct"
  enable_inference: false  # Stub mode
```

```bash
python scripts/c3_generate.py "query"
```

**Result:** ✓ Boots successfully (no model loading in stub mode)

## Configuration Examples

### Production Config (Stub Mode)

```yaml
# configs/production.yaml
decoder:
  type: qwen
  model_name: "Qwen/Qwen2.5-32B-Instruct"
  device: cpu
  enable_inference: false  # Safe stub mode
  
retrieval:
  collection_name: production_chunks
  top_k: 5
```

### Development Config (Override via ENV)

```yaml
# configs/development.yaml
decoder:
  type: qwen
  enable_inference: false
  
retrieval:
  collection_name: dev_chunks
  top_k: 3
```

```bash
# Override for testing
export RAG_DECODER_TYPE=stub
export RAG_COLLECTION_NAME=test_chunks

python scripts/c3_generate.py "query"
```

### Staging Config (Inference Enabled)

```yaml
# configs/staging.yaml
decoder:
  type: qwen
  model_name: "Qwen/Qwen2.5-7B-Instruct"  # Small model
  device: cuda
  enable_inference: true
  
retrieval:
  collection_name: staging_chunks
  top_k: 5
```

```bash
# Enable inference
export ALLOW_LLM_INFERENCE=1

python scripts/c3_generate.py "query" --use-llm
```

## Files Created/Modified

### Created Files

1. **`src/runtime/config_loader.py`** (200 lines)
   - ConfigLoader class
   - Environment variable mappings
   - Config validation
   - load_config() function

2. **`src/runtime/health.py`** (280 lines)
   - HealthChecker class
   - 4 health checks
   - HealthCheckResult dataclass
   - health_check() function
   - print_health_check() function

### Modified Files

1. **`src/runtime/__init__.py`** - Added Phase-6 exports
2. **`scripts/c3_generate.py`** - Added health check on startup
3. **`scripts/c3_synthesize.py`** - Added health check on startup

## API Reference

### ConfigLoader.load_config()

```python
def load_config(
    config_path: str,
    apply_env: bool = True,
    validate: bool = True
) -> Dict[str, Any]:
    """Load, override, and validate configuration.
    
    Args:
        config_path: Path to YAML config file
        apply_env: Apply environment variable overrides
        validate: Validate required keys
    
    Returns:
        Validated configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigValidationError: If validation fails
    """
```

### health_check()

```python
def health_check(config: Optional[Dict[str, Any]]) -> HealthCheckResult:
    """Perform comprehensive system health check.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        HealthCheckResult with detailed status
    """
```

### print_health_check()

```python
def print_health_check(result: HealthCheckResult, verbose: bool = False) -> None:
    """Print health check result in human-readable format.
    
    Args:
        result: HealthCheckResult to print
        verbose: Show detailed check results
    """
```

## Deployment Checklist

### Pre-Deployment

- [ ] Config file exists and is valid
- [ ] Required keys present in config
- [ ] Environment variables set (if needed)
- [ ] ChromaDB directory configured
- [ ] Collection name matches indexed data

### Startup Validation

- [ ] Config loads successfully
- [ ] Environment overrides applied
- [ ] Config validation passes
- [ ] Health check passes
- [ ] No critical errors

### Runtime Checks

- [ ] Inference status correct (enabled/disabled)
- [ ] Decoder type supported
- [ ] No warnings (or warnings acknowledged)
- [ ] System boots without GPU (if needed)
- [ ] System boots without models (if needed)

## Troubleshooting

### Config Validation Failed

**Error:** `Missing required key: decoder.type`

**Solution:**
```yaml
# Add missing key to config
decoder:
  type: qwen  # Add this
```

### Health Check Failed: Decoder Not Registered

**Error:** `Decoder: Unsupported decoder type: invalid`

**Solution:**
```yaml
# Use supported decoder type
decoder:
  type: qwen  # or llama, stub
```

### Environment Override Not Applied

**Issue:** Environment variable set but not used

**Solution:**
```bash
# Check variable name matches mapping
export RAG_DECODER_TYPE=stub  # Correct
export DECODER_TYPE=stub      # Wrong - not in mapping
```

### Health Check Warnings

**Warning:** `Inference enabled but ALLOW_LLM_INFERENCE not set`

**Solution:**
```bash
# Set environment variable
export ALLOW_LLM_INFERENCE=1
```

## Design Principles

### Fail Fast

- Validate config on startup
- Exit immediately if unhealthy
- Don't process requests with bad config

### Environment-Aware

- Support environment overrides
- Enable 12-factor app deployment
- Allow runtime configuration

### Boot Without Resources

- No GPU required for boot
- No models required for boot
- Stub mode works everywhere

### Clear Feedback

- Detailed health check output
- Actionable error messages
- Warnings for potential issues

## Summary

### Delivered

✓ **Config loader** - YAML + ENV overrides  
✓ **Config validation** - Required keys checked  
✓ **Health check** - 4 comprehensive checks  
✓ **Startup integration** - Both entrypoints  
✓ **Boot without GPU** - Stub mode works  
✓ **Boot without models** - No loading required  

### Health Checks

✓ **Config loaded** - Valid configuration  
✓ **Retriever ready** - ChromaDB configured  
✓ **Decoder registered** - Decoder available  
✓ **Inference status** - Enabled/disabled reported  

### Environment Overrides

✓ **8 environment variables** - Config overrides  
✓ **Type conversion** - Boolean, integer support  
✓ **Priority order** - ENV > YAML > defaults  

Phase-6 Deployment Readiness is complete and production-ready with comprehensive config loading, validation, and health checks.
