# Phase-4: Local Decoder Abstraction

## Overview

Phase-4 replaces all external LLM calls (OpenAI) with a **swappable local decoder interface** without changing any C3, Phase-2, or Phase-3 logic.

**Key Principle:** Decoder is swappable via config, with **safe stub mode** as the default (no inference, no model loading).

## Architecture

### Decoder Interface Hierarchy

```
DecoderInterface (Abstract Base)
    ↓
LocalHFDecoder (HuggingFace Implementation)
    ↓
Registry (get_decoder)
```

### Safe Stub Mode (Default)

```python
decoder:
  type: "qwen"
  model_name: "Qwen/Qwen2.5-32B-Instruct"
  enable_inference: false  # SAFE MODE: No model loading, no GPU
```

**Stub Behavior:**
- Returns deterministic refusal: `"I cannot answer based on the provided documents."`
- No model loading
- No GPU usage
- No inference

## Implementation

### 1. Decoder Base Interface (`src/decoders/base.py`)

```python
class DecoderInterface(ABC):
    """Abstract base class for all decoder implementations."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                 temperature: float = 0.0) -> str:
        """Generate text from prompt."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return decoder name."""
        pass
```

**Contract:**
- All decoders implement `generate()` and `name` property
- Deterministic by default (temperature=0.0)
- Optional max_tokens parameter

### 2. Local HuggingFace Decoder (`src/decoders/local_hf.py`)

```python
class LocalHFDecoder(DecoderInterface):
    """Local HuggingFace decoder with safe stub mode."""
    
    def __init__(self, model_name: str, device: str = "cpu", 
                 dtype: str = "float32", enable_inference: bool = False):
        self.enable_inference = enable_inference
        
        if self.enable_inference:
            raise NotImplementedError("Inference mode not implemented in Phase-4")
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                 temperature: float = 0.0) -> str:
        if not self.enable_inference:
            # Safe stub mode: return deterministic refusal
            return "I cannot answer based on the provided documents."
        
        raise NotImplementedError("Inference mode not implemented")
```

**Features:**
- **Safe by default:** `enable_inference=False`
- **No model loading** in stub mode
- **Deterministic refusal** in stub mode
- **Inference mode not implemented** (Phase-4 scope)

### 3. Decoder Registry (`src/decoders/registry.py`)

```python
def get_decoder(config: Dict[str, Any]) -> DecoderInterface:
    """Get decoder instance from configuration.
    
    Supported types:
    - "qwen": Qwen models (e.g., Qwen2.5-32B-Instruct)
    - "llama": Llama models (e.g., Llama-3-8B-Instruct)
    - "stub": Safe stub decoder (always returns refusal)
    """
    
    decoder_type = config.get("type", "stub")
    model_name = config.get("model_name", "Qwen/Qwen2.5-32B-Instruct")
    device = config.get("device", "cpu")
    dtype = config.get("dtype", "float32")
    enable_inference = config.get("enable_inference", False)
    
    if decoder_type == "qwen":
        return LocalHFDecoder(model_name, device, dtype, enable_inference)
    elif decoder_type == "llama":
        return LocalHFDecoder(model_name, device, dtype, enable_inference)
    elif decoder_type == "stub":
        return LocalHFDecoder("stub", "cpu", "float32", False)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")
```

**Registry Features:**
- Central decoder selection
- Type-based instantiation
- Default safe stub mode
- Extensible for new decoder types

## Configuration

### Config File (`configs/phase1_rag.yaml`)

```yaml
# Decoder Model (generation) - Phase-4 Local Decoder Abstraction
decoder:
  type: "qwen"  # Options: qwen, llama, stub
  model_name: "Qwen/Qwen2.5-32B-Instruct"
  device: "cpu"  # Options: cpu, cuda, mps
  dtype: "float32"  # Options: float32, float16, bfloat16
  enable_inference: false  # SAFE MODE: false = stub, true = actual inference
  max_context_length: 2048
  max_generation_length: 512
  temperature: 0.0  # Deterministic generation
```

**Config Options:**

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `type` | qwen, llama, stub | stub | Decoder type |
| `model_name` | HF model name | Qwen/Qwen2.5-32B-Instruct | Model identifier |
| `device` | cpu, cuda, mps | cpu | Device for inference |
| `dtype` | float32, float16, bfloat16 | float32 | Data type |
| `enable_inference` | true, false | **false** | Enable actual inference |

## Integration

### Before (Phase-2/3): OpenAI

```python
# Old OpenAI-based generation
def generate_answer_openai(prompt: str, model: str = "gpt-3.5-turbo", 
                          temperature: float = 0.0) -> str:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[...],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()
```

### After (Phase-4): Decoder Interface

```python
# New decoder-based generation
def generate_answer_decoder(prompt: str, config: dict, 
                           temperature: float = 0.0) -> str:
    decoder = get_decoder(config.get('decoder', {}))
    answer = decoder.generate(prompt, max_tokens=500, temperature=temperature)
    return answer.strip()
```

### Updated Scripts

**c3_generate.py:**
```python
# Before
from openai import ...
answer = generate_answer_openai(prompt, model="gpt-3.5-turbo")

# After
from src.decoders import get_decoder
answer = generate_answer_decoder(prompt, config)
```

**c3_synthesize.py:**
```python
# Before
answer = generate_answer_openai(prompt, model="gpt-3.5-turbo", temperature=0.0)

# After
answer = generate_answer_decoder(prompt, config, temperature=0.0)
```

## Usage

### Mock Mode (Default)

```bash
# Uses mock generator (no decoder)
python scripts/c3_generate.py "What is Section 420 IPC?"
python scripts/c3_synthesize.py "What is the minimum wage?"
```

### Stub Mode (Decoder with Safe Stub)

```bash
# Uses decoder in stub mode (returns refusal)
python scripts/c3_generate.py "What is Section 420 IPC?" --use-llm
python scripts/c3_synthesize.py "What is the minimum wage?" --use-llm
```

**Output (Stub Mode):**
```
Answer: I cannot answer based on the provided documents.
```

### Inference Mode (Not Implemented)

```yaml
# To enable inference (NOT IMPLEMENTED in Phase-4)
decoder:
  enable_inference: true  # Will raise NotImplementedError
```

## Strict Rules Enforced

### ✓ What Phase-4 Does

1. **Swappable decoder interface** - Config-driven decoder selection
2. **Safe stub mode by default** - No model loading, no inference
3. **No changes to C3 logic** - All validation logic unchanged
4. **No changes to Phase-2/3 logic** - Grounding, sufficiency, coverage unchanged
5. **Deterministic stub behavior** - Always returns exact refusal message

### ✗ What Phase-4 Does NOT Do

1. **No model downloading** - Stub mode only
2. **No model loading** - No weights loaded
3. **No GPU usage** - CPU only, no inference
4. **No actual inference** - Stub returns refusal
5. **No logic changes** - C3, Phase-2, Phase-3 logic untouched

## Files Created/Modified

### Created Files

1. **`src/decoders/__init__.py`** - Package initialization
2. **`src/decoders/base.py`** - Abstract decoder interface
3. **`src/decoders/local_hf.py`** - HuggingFace decoder with stub mode
4. **`src/decoders/registry.py`** - Decoder registry and selection

### Modified Files

1. **`configs/phase1_rag.yaml`** - Added decoder configuration
2. **`scripts/c3_generate.py`** - Replaced OpenAI with decoder interface
3. **`scripts/c3_synthesize.py`** - Replaced OpenAI with decoder interface

## Verification

### Test Decoder Interface

```bash
python -c "
from src.decoders import get_decoder

# Test stub decoder
config = {'type': 'stub'}
decoder = get_decoder(config)
print(f'Decoder: {decoder.name}')

answer = decoder.generate('test prompt')
print(f'Answer: {answer}')
"
```

**Expected Output:**
```
Decoder: LocalHF-Stub(stub)
Answer: I cannot answer based on the provided documents.
```

### Test Scripts in Stub Mode

```bash
# Test c3_generate.py
python scripts/c3_generate.py "What is Section 420 IPC?"

# Test c3_synthesize.py
python scripts/c3_synthesize.py "What is the minimum wage?"
```

**Both should run successfully in mock mode (no decoder).**

## Design Decisions

### Why Stub Mode by Default?

1. **Safety** - No accidental model loading or GPU usage
2. **Testing** - Can test integration without models
3. **Development** - Fast iteration without inference overhead
4. **Determinism** - Predictable behavior for testing

### Why Not Implement Inference?

Phase-4 scope is **wiring only**:
- Establish decoder interface
- Replace OpenAI calls
- Ensure scripts run in stub mode
- **NOT** implement actual inference

Inference implementation is future work.

### Why Abstract Interface?

1. **Extensibility** - Easy to add new decoder types
2. **Testability** - Can mock decoders for testing
3. **Flexibility** - Swap decoders without code changes
4. **Maintainability** - Single interface for all decoders

## Future Work

### Phase-5: Actual Inference (Future)

To enable actual inference:

1. **Implement model loading** in `LocalHFDecoder`
2. **Add GPU support** (cuda, mps)
3. **Implement generation** with transformers
4. **Add quantization** (int8, int4)
5. **Add batching** for efficiency

**Example (Future):**
```python
class LocalHFDecoder(DecoderInterface):
    def __init__(self, model_name, device, dtype, enable_inference):
        if enable_inference:
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model.to(device)
    
    def generate(self, prompt, max_tokens, temperature):
        if self.enable_inference:
            # Actual inference
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
            return self.tokenizer.decode(outputs[0])
        else:
            # Stub mode
            return REFUSAL_MESSAGE
```

## Summary

### Delivered

✓ **Decoder interface** - Abstract base class  
✓ **Local HF decoder** - With safe stub mode  
✓ **Decoder registry** - Config-driven selection  
✓ **Config integration** - Decoder settings in YAML  
✓ **Script updates** - c3_generate.py and c3_synthesize.py  
✓ **Verification** - All scripts run in stub mode  

### Constraints Maintained

✓ **No model loading** - Stub mode only  
✓ **No GPU usage** - CPU only  
✓ **No inference** - Returns refusal  
✓ **No logic changes** - C3, Phase-2, Phase-3 unchanged  
✓ **Safe by default** - enable_inference=false  

### Ready for Future

✓ **Extensible interface** - Easy to add inference  
✓ **Config-driven** - No code changes needed  
✓ **Swappable decoders** - qwen, llama, stub  
✓ **Deterministic** - Predictable stub behavior  

Phase-4 Local Decoder Abstraction is complete and production-ready in stub mode.
