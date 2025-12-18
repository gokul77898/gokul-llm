# Phase 3.5: Training Infrastructure

Production-grade training infrastructure for the MoE Legal AI system.

**⚠️ DO NOT EXECUTE TRAINING - Infrastructure setup only.**

## Authorized Models (Phase 3.6)

| Role | Model ID | Description |
|------|----------|-------------|
| **Encoder** | `law-ai/Indian-Legal-Assistant-8B` | Indian Legal domain encoder |
| **Decoder** | `Qwen/Qwen2.5-32B-Instruct` | Instruction-following decoder |

These are the **ONLY** authorized models. No other models are permitted.

Training is **DISABLED** in Phase 3.5 - infrastructure setup only.

## Directory Structure

```
src/training/
├── __init__.py
├── configs/
│   ├── encoder_sft.yaml      # Encoder (NER) training config
│   └── decoder_sft.yaml      # Decoder (SFT) training config
├── datasets/
│   ├── encoder/              # Encoder training data (JSONL)
│   │   ├── train.jsonl
│   │   └── eval.jsonl
│   └── decoder/              # Decoder training data (JSONL)
│       ├── train.jsonl
│       └── eval.jsonl
├── trainers/
│   ├── __init__.py
│   ├── encoder_trainer.py    # HuggingFace Trainer for NER
│   └── decoder_trainer.py    # SFT trainer with LoRA support
├── eval/
│   ├── __init__.py
│   ├── encoder_eval.py       # Encoder metrics (recall, FPR)
│   └── decoder_eval.py       # Decoder metrics (fact-adherence, refusal)
├── utils/
│   ├── __init__.py
│   ├── data_validation.py    # Data contract validation
│   └── metrics.py            # Shared metrics utilities
└── README.md
```

## Data Contracts

### Encoder Data (JSONL)

```json
{
  "text": "The accused was charged under Section 420 IPC.",
  "entities": [
    {"start": 32, "end": 43, "label": "SECTION"}
  ],
  "task": "ner"
}
```

**Required fields:**
- `text`: Input text (string, min 10 chars)
- `entities`: List of entity spans with `start`, `end`, `label`
- `task`: Task type (`ner` or `classification`)

**Valid labels:**
- `SECTION`, `ACT`, `PARTY`, `DATE`, `COURT`, `JUDGE`, `OFFENSE`, `PENALTY`, `CITATION`, `MISC`

### Decoder Data (JSONL)

```json
{
  "prompt": "ENCODER_FACTS:\n- Section 420\n\nQUESTION:\nWhat is Section 420 IPC?",
  "response": "Section 420 IPC deals with cheating and dishonestly inducing delivery of property.",
  "refusal_allowed": true
}
```

**Required fields:**
- `prompt`: Input prompt with ENCODER_FACTS block
- `response`: Expected response

**Optional fields:**
- `refusal_allowed`: Whether refusal is acceptable (default: true)

**Refusal samples:**
```json
{
  "prompt": "ENCODER_FACTS:\n(none)\n\nQUESTION:\nWhat is the law?",
  "response": "REFUSE: Missing facts.",
  "refusal_allowed": true
}
```

## Usage

### Validate Data

```python
from src.training.utils import validate_all_datasets

results = validate_all_datasets()
for name, result in results.items():
    print(f"{name}: {'VALID' if result.valid else 'INVALID'}")
    print(f"  Stats: {result.stats}")
```

### Setup Encoder Trainer

```python
from src.training.trainers import create_encoder_trainer

trainer = create_encoder_trainer("src/training/configs/encoder_sft.yaml")
print(trainer.get_training_args())
print(trainer.validate_ready_to_train())
```

### Setup Decoder Trainer

```python
from src.training.trainers import create_decoder_trainer

trainer = create_decoder_trainer("src/training/configs/decoder_sft.yaml")
print(trainer.get_training_args())
print(trainer.get_peft_config())
```

### Evaluate Encoder

```python
from src.training.eval import create_encoder_evaluator

evaluator = create_encoder_evaluator(["O", "B-SECTION", "I-SECTION", ...])
result = evaluator.evaluate(predictions, references)
print(f"Recall: {result.recall}, FPR: {result.false_positive_rate}")
```

### Evaluate Decoder

```python
from src.training.eval import create_decoder_evaluator

evaluator = create_decoder_evaluator()
result = evaluator.evaluate(outputs, prompts, should_refuse)
print(f"Fact Adherence: {result.fact_adherence}")
print(f"Refusal F1: {result.refusal_f1}")
```

## Evaluation Metrics

### Encoder Metrics
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1**: Harmonic mean of precision and recall
- **False Positive Rate**: FP / (FP + TN)
- **Entity-level metrics**: Exact span matching

### Decoder Metrics
- **Fact Adherence**: % of non-refusal outputs that mention encoder facts
- **Refusal Precision**: True refusals / All refusals
- **Refusal Recall**: True refusals / Expected refusals
- **Overall Accuracy**: (True refusals + Correct answers) / Total

## Hard Rules

1. **Training code NEVER imported by inference code**
2. **No model weights downloaded or trained**
3. **Config-driven design (YAML)**
4. **Refusal-aware data supported**
5. **Evaluation defined before training**

## Next Steps (Phase 4)

1. Create training datasets
2. Run data validation
3. Execute encoder training
4. Execute decoder training
5. Evaluate trained models
6. Deploy to inference
