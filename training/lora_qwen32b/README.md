# Production LoRA Training for Qwen2.5-32B-Instruct

## Overview

Production-grade LoRA fine-tuning pipeline for the Qwen2.5-32B-Instruct model on Indian legal text corpus.

**This is REAL training with REAL data on REAL GPUs.**

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPUs | 4× A100 80GB | 8× A100 80GB or 4× H100 80GB |
| VRAM per GPU | 80 GB | 80 GB |
| System RAM | 256 GB | 512 GB |
| Storage | 1 TB NVMe | 2 TB NVMe |
| CUDA | 11.8+ | 12.1+ |

## Directory Structure

```
training/lora_qwen32b/
├── train.py                 # Main training script
├── lora_config.py           # LoRA and training configuration
├── deepspeed_config.json    # DeepSpeed ZeRO-2 configuration
├── data/
│   ├── train.jsonl          # Training data
│   └── val.jsonl            # Validation data (optional)
├── outputs/
│   └── adapters/
│       ├── checkpoint-1000/
│       ├── checkpoint-2000/
│       └── final/
└── README.md
```

## Data Format

Training data must be in JSONL format with a single `text` field:

```jsonl
{"text": "Section 420 of the Indian Penal Code deals with cheating..."}
{"text": "The Supreme Court held in the case of..."}
```

**Requirements:**
- UTF-8 encoding
- Plain text only (no HTML, markdown, or markup)
- No labels, citations, or metadata
- Maximum sequence length: 2048 tokens

## LoRA Configuration

| Parameter | Value |
|-----------|-------|
| r | 16 |
| lora_alpha | 32 |
| lora_dropout | 0.05 |
| target_modules | q_proj, v_proj |
| bias | none |
| task_type | CAUSAL_LM |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Micro batch size | 1 |
| Gradient accumulation | 16 |
| Effective batch size | num_gpus × 16 |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Warmup steps | 1000 |
| LR scheduler | cosine |
| Save steps | 1000 |
| Logging steps | 50 |

## DeepSpeed Configuration

- **ZeRO Stage**: 2
- **Optimizer offload**: CPU
- **Gradient checkpointing**: Enabled
- **BF16**: Enabled (required)

## Usage

### Single Node Multi-GPU

```bash
deepspeed --num_gpus=8 train.py \
    --data_path data/train.jsonl \
    --val_path data/val.jsonl
```

### Multi-Node Training

```bash
deepspeed --hostfile=hostfile.txt \
    --num_nodes=2 \
    --num_gpus=8 \
    train.py \
    --data_path data/train.jsonl
```

### Custom Max Steps

```bash
deepspeed --num_gpus=8 train.py \
    --data_path data/train.jsonl \
    --max_steps 50000
```

## Output

Training produces **LoRA adapters only** in Hugging Face PEFT format:

```
outputs/adapters/final/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer_config.json
├── tokenizer.json
└── special_tokens_map.json
```

**No base model weights are saved.**

## Loading Trained Adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "outputs/adapters/final",
)

# Inference
tokenizer = AutoTokenizer.from_pretrained("outputs/adapters/final")
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
```

## Validation

The training script automatically runs post-training validation:

1. Loads fresh base model
2. Runs deterministic test prompt
3. Loads base + LoRA adapter
4. Runs same prompt
5. Confirms behavioral difference

## Fail Conditions

Training will **ABORT** if:

- CUDA is not available
- BF16 is not supported
- Dataset is missing or empty
- Base model parameters have `requires_grad=True`
- LoRA adapters are merged into base model
- Any RAG/Graph/ChromaDB imports are detected

## Monitoring

During training, the following are logged:

- Loss curve (every 50 steps)
- GPU memory usage
- Step time
- Learning rate schedule

## Troubleshooting

### Out of Memory

1. Reduce `gradient_accumulation_steps` in `deepspeed_config.json`
2. Enable CPU offload for parameters (ZeRO-3)
3. Reduce `max_seq_length` in `lora_config.py`

### NaN/Inf Loss

1. Reduce learning rate
2. Increase warmup steps
3. Check data for invalid samples

### Slow Training

1. Verify NCCL backend is working
2. Check GPU interconnect (NVLink/InfiniBand)
3. Increase `dataloader_num_workers`

## License

This training pipeline is part of the Omilos Legal LLM project.
