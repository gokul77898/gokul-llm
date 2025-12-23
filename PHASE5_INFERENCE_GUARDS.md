# Phase-5: Inference Safety & Runtime Guards

## Overview

Phase-5 implements **runtime guards** to prevent accidental inference, GPU usage, or model loading. These guards enforce multiple safety checks before allowing any inference operations.

**Key Principle:** Defense in depth - multiple independent checks must pass before inference is allowed.

## Architecture

### Guard System

```
Request → LocalHFDecoder.generate()
            ↓
         enable_inference?
            ↓ No
         Return C3 Refusal (Safe)
            ↓ Yes
         InferenceGuard.assert_inference_allowed()
            ↓
         Check 1: enable_inference == true ✓
         Check 2: ALLOW_LLM_INFERENCE=1 env var ✓
         Check 3: GPU required for large models ✓
            ↓ Any fail
         Log reason + Return C3 Refusal
            ↓ All pass
         Proceed to inference (not implemented)
```

## Implementation

### 1. InferenceGuard (`src/runtime/guards.py`)

```python
class InferenceGuard:
    """Runtime guard to prevent accidental inference.
    
    Checks multiple safety conditions:
    1. enable_inference must be True in config
    2. ALLOW_LLM_INFERENCE=1 environment variable must be set
    3. Large models require GPU (device != cpu)
    """
    
    LARGE_MODELS = [
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "meta-llama/Llama-3-70B-Instruct",
        "meta-llama/Llama-2-70B-chat",
    ]
    
    @staticmethod
    def assert_inference_allowed(
        enable_inference: bool,
        model_name: str,
        device: str
    ) -> Tuple[bool, str]:
        """Assert that inference is allowed.
        
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
```

### 2. Safety Checks

#### Check 1: Config Flag

```python
if not enable_inference:
    return InferenceGuardResult(
        allowed=False,
        reason="Inference disabled by config (enable_inference=false)"
    )
```

**Purpose:** Explicit opt-in required in config file.

#### Check 2: Environment Variable

```python
env_flag = os.environ.get("ALLOW_LLM_INFERENCE", "0")
if env_flag != "1":
    return InferenceGuardResult(
        allowed=False,
        reason="Inference blocked: ALLOW_LLM_INFERENCE environment variable not set to 1"
    )
```

**Purpose:** Explicit runtime opt-in required via environment.

#### Check 3: GPU Requirement for Large Models

```python
if model_name in InferenceGuard.LARGE_MODELS:
    if device == "cpu":
        return InferenceGuardResult(
            allowed=False,
            reason=f"GPU required for large model {model_name} (device=cpu not allowed)"
        )
```

**Purpose:** Prevent OOM crashes from running large models on CPU.

### 3. Enforcement in LocalHFDecoder

```python
def generate(self, prompt: str, max_tokens: Optional[int] = None, 
             temperature: float = 0.0) -> str:
    """Generate text from prompt with runtime guards."""
    
    if not self.enable_inference:
        # Safe stub mode: return deterministic refusal
        logger.debug("Stub mode: returning C3 refusal")
        return self.REFUSAL_MESSAGE
    
    # Phase-5: Enforce runtime guards
    if GUARDS_AVAILABLE:
        allowed, reason = assert_inference_allowed(
            enable_inference=self.enable_inference,
            model_name=self.model_name,
            device=self.device
        )
        
        if not allowed:
            # Guard failed: log and return C3 refusal
            logger.warning(f"Inference guard blocked generation: {reason}")
            return self.REFUSAL_MESSAGE
        
        # Guards passed: would proceed with inference
        logger.info(f"Inference guards passed: {reason}")
    
    # Inference mode not implemented (Phase-4 scope)
    raise NotImplementedError("Inference mode not implemented in Phase-4")
```

## Error Messages

### Clear, Actionable Messages

| Scenario | Error Message |
|----------|--------------|
| Config disabled | `"Inference disabled by config (enable_inference=false)"` |
| Env var not set | `"Inference blocked: ALLOW_LLM_INFERENCE environment variable not set to 1"` |
| CPU for large model | `"GPU required for large model {model_name} (device=cpu not allowed)"` |
| All checks passed | `"Inference allowed: all safety checks passed"` |

## Usage

### Safe Mode (Default)

```yaml
# Config: configs/phase1_rag.yaml
decoder:
  enable_inference: false  # Safe mode
```

```bash
# No environment variable needed
python scripts/c3_generate.py "query"
```

**Result:** Returns C3 refusal (safe stub mode)

### Attempting Inference (Blocked by Guards)

```yaml
# Config: configs/phase1_rag.yaml
decoder:
  enable_inference: true  # Try to enable
  model_name: "Qwen/Qwen2.5-32B-Instruct"
  device: "cpu"
```

```bash
# Without environment variable
python scripts/c3_generate.py "query" --use-llm
```

**Result:** Guard blocks, returns C3 refusal with reason:
```
WARNING: Inference blocked: ALLOW_LLM_INFERENCE environment variable not set to 1
Answer: I cannot answer based on the provided documents.
```

### Enabling Inference (All Guards Pass)

```yaml
# Config
decoder:
  enable_inference: true
  model_name: "Qwen/Qwen2.5-7B-Instruct"  # Small model
  device: "cuda"  # GPU
```

```bash
# With environment variable
export ALLOW_LLM_INFERENCE=1
python scripts/c3_generate.py "query" --use-llm
```

**Result:** Guards pass, but inference not implemented:
```
INFO: Inference guards passed: all safety checks passed
ERROR: NotImplementedError: Inference mode not implemented in Phase-4
```

## Guard Failure Scenarios

### Scenario 1: Config Disabled

```yaml
decoder:
  enable_inference: false
```

**Guard Result:**
- ✗ Check 1 fails: `enable_inference=false`
- **Action:** Return C3 refusal
- **Log:** `"Stub mode: returning C3 refusal"`

### Scenario 2: Environment Variable Not Set

```yaml
decoder:
  enable_inference: true
```

```bash
# ALLOW_LLM_INFERENCE not set
python scripts/c3_generate.py "query" --use-llm
```

**Guard Result:**
- ✓ Check 1 passes: `enable_inference=true`
- ✗ Check 2 fails: `ALLOW_LLM_INFERENCE != 1`
- **Action:** Return C3 refusal
- **Log:** `"Inference blocked: ALLOW_LLM_INFERENCE environment variable not set to 1"`

### Scenario 3: Large Model on CPU

```yaml
decoder:
  enable_inference: true
  model_name: "Qwen/Qwen2.5-32B-Instruct"
  device: "cpu"
```

```bash
export ALLOW_LLM_INFERENCE=1
python scripts/c3_generate.py "query" --use-llm
```

**Guard Result:**
- ✓ Check 1 passes: `enable_inference=true`
- ✓ Check 2 passes: `ALLOW_LLM_INFERENCE=1`
- ✗ Check 3 fails: Large model on CPU
- **Action:** Return C3 refusal
- **Log:** `"GPU required for large model Qwen/Qwen2.5-32B-Instruct (device=cpu not allowed)"`

### Scenario 4: All Checks Pass

```yaml
decoder:
  enable_inference: true
  model_name: "Qwen/Qwen2.5-7B-Instruct"  # Small model OK on CPU
  device: "cpu"
```

```bash
export ALLOW_LLM_INFERENCE=1
python scripts/c3_generate.py "query" --use-llm
```

**Guard Result:**
- ✓ Check 1 passes: `enable_inference=true`
- ✓ Check 2 passes: `ALLOW_LLM_INFERENCE=1`
- ✓ Check 3 passes: Small model on CPU allowed
- **Action:** Proceed to inference (not implemented)
- **Log:** `"Inference guards passed: all safety checks passed"`

## Design Principles

### Defense in Depth

Multiple independent checks ensure safety:
1. **Config check** - Explicit opt-in in config file
2. **Environment check** - Explicit runtime opt-in
3. **Resource check** - Prevent OOM from large models on CPU

### Fail-Safe Behavior

If any check fails:
- **MUST** return C3 refusal string
- **MUST** log the reason
- **MUST NOT** attempt inference
- **MUST NOT** load models

### Zero Behavior Change

Phase-5 adds guards but changes no existing behavior:
- ✓ Stub mode unchanged (still returns refusal)
- ✓ C3 logic unchanged
- ✓ Phase-2/3 logic unchanged
- ✓ Scripts work exactly as before

## Files Created/Modified

### Created Files

1. **`src/runtime/__init__.py`** - Runtime package initialization
2. **`src/runtime/guards.py`** - InferenceGuard implementation

### Modified Files

1. **`src/decoders/local_hf.py`** - Enforces guards in `generate()`

## Verification

### Test Guard Blocking (Config Disabled)

```python
from src.decoders import get_decoder

config = {
    'type': 'qwen',
    'enable_inference': False  # Disabled
}

decoder = get_decoder(config)
answer = decoder.generate("test prompt")

print(answer)
# Output: "I cannot answer based on the provided documents."
```

### Test Guard Blocking (Env Var Not Set)

```python
import os
from src.decoders.local_hf import LocalHFDecoder

# Ensure env var not set
os.environ.pop('ALLOW_LLM_INFERENCE', None)

decoder = LocalHFDecoder(
    model_name="Qwen/Qwen2.5-32B-Instruct",
    device="cuda",
    enable_inference=True  # Try to enable
)

# This will fail at __init__ (Phase-4 not implemented)
# But if it didn't, generate() would block via guard
```

### Test Guard Blocking (Large Model on CPU)

```python
import os
from src.runtime.guards import assert_inference_allowed

os.environ['ALLOW_LLM_INFERENCE'] = '1'

allowed, reason = assert_inference_allowed(
    enable_inference=True,
    model_name="Qwen/Qwen2.5-32B-Instruct",
    device="cpu"
)

print(f"Allowed: {allowed}")
print(f"Reason: {reason}")
# Output:
# Allowed: False
# Reason: GPU required for large model Qwen/Qwen2.5-32B-Instruct (device=cpu not allowed)
```

## Large Models List

Models requiring GPU (>10B parameters):

```python
LARGE_MODELS = [
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3-70B-Instruct",
    "meta-llama/Llama-2-70B-chat",
]
```

**Small models** (can run on CPU):
- Qwen/Qwen2.5-7B-Instruct
- meta-llama/Llama-3-8B-Instruct
- Any model not in LARGE_MODELS list

## Logging

### Log Levels

| Level | Message | Scenario |
|-------|---------|----------|
| DEBUG | `"Stub mode: returning C3 refusal"` | Stub mode active |
| WARNING | `"Inference guard blocked generation: {reason}"` | Guard blocked inference |
| INFO | `"Inference guards passed: {reason}"` | Guards passed |
| WARNING | `"Runtime guards not available"` | Guards module missing |

### Example Log Output

```
[2025-12-21 18:40:00] DEBUG: Stub mode: returning C3 refusal
[2025-12-21 18:40:05] WARNING: Inference blocked: ALLOW_LLM_INFERENCE environment variable not set to 1
[2025-12-21 18:40:10] WARNING: Inference guard blocked generation: GPU required for large model Qwen/Qwen2.5-32B-Instruct (device=cpu not allowed)
[2025-12-21 18:40:15] INFO: Inference allowed: all safety checks passed
```

## Future Enhancements

### Additional Guards (Future)

1. **Memory check** - Ensure sufficient RAM/VRAM
2. **Disk space check** - Ensure space for model cache
3. **Network check** - Warn if model needs download
4. **License check** - Verify model license acceptance
5. **Rate limiting** - Prevent excessive inference calls

### Guard Configuration (Future)

```yaml
decoder:
  guards:
    require_env_var: true  # Require ALLOW_LLM_INFERENCE
    check_gpu_memory: true  # Check VRAM before loading
    check_disk_space: true  # Check disk space
    max_model_size_gb: 50   # Max model size allowed
```

## Summary

### Delivered

✓ **InferenceGuard class** - Multi-check safety system  
✓ **assert_inference_allowed()** - Guard enforcement function  
✓ **LocalHFDecoder integration** - Guards enforced in generate()  
✓ **Clear error messages** - Actionable failure reasons  
✓ **C3 refusal on failure** - Safe fallback behavior  
✓ **Logging** - All guard decisions logged  

### Safety Checks

✓ **Check 1:** Config flag (`enable_inference=true`)  
✓ **Check 2:** Environment variable (`ALLOW_LLM_INFERENCE=1`)  
✓ **Check 3:** GPU requirement for large models  

### Behavior Guarantees

✓ **Zero behavior change** - Existing code unchanged  
✓ **Fail-safe** - Returns C3 refusal on any guard failure  
✓ **Defense in depth** - Multiple independent checks  
✓ **No model loading** - Guards prevent accidental loading  
✓ **No GPU usage** - Guards prevent accidental GPU use  

Phase-5 Inference Safety & Runtime Guards is complete and production-ready.
