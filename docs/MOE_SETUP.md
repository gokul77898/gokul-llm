# MARK MoE Architecture Setup

## Overview

The MARK system has been migrated to a **Pure Mixture-of-Experts (MoE)** architecture using Hugging Face models. All Mamba, RWKV, and custom kernel dependencies have been removed.

## Experts

The system uses a set of specialized experts defined in `configs/moe_experts.yaml`:

- **InLegalLLaMA**: QA, Summarization, Reasoning (Long context)
- **InLegalBERT**: Classification, NER
- **Paramanu-Ayn**: Translation
- **Aalap**: Simplification
- **OpenNyAI**: Specific legal tasks

## Components

1. **Registry**: `src/core/model_registry.py` - Manages expert metadata and loading.
2. **Router**: `src/inference/moe_router.py` - Routes queries based on task, length, and keywords.
3. **Trainer**: `src/training/expert_trainer.py` - LoRA/Full fine-tuning for experts.
4. **Server**: `src/inference/server.py` - REST API for the MoE system.

## Usage

### 1. Route a Query
```bash
python -m src.inference.moe_router --text "What is the penalty for murder under Section 302?" --task qa
```

### 2. Train an Expert (LoRA)
```bash
python -m src.training.expert_trainer --expert inlegalllama --mode lora --confirm-run
```

### 3. Start Inference Server
```bash
python -m src.inference.server
```

## Logs

- Routing decisions: `logs/model_routing.json`
- Training runs: `logs/training_runs.json`

## Deployment

Use `deploy/triton_recipe.sh` to scaffold a Triton Inference Server model repository.

## 🖥️ UI Integration Setup

To prepare the backend for UI MoE testing:

### 1. Download Experts
```bash
python scripts/download_experts.py --all
# Or for a single expert:
# python scripts/download_experts.py --expert inlegalllama
```

### 2. Start Backend Server
```bash
python -m src.inference.server
# Server runs on http://0.0.0.0:8000
```

### 3. UI API Endpoints
The UI can now use these endpoints:

- `GET /ready` - Check readiness & loaded experts
- `POST /moe-test` - Test routing (no model load)
- `POST /moe-generate` - Full MoE generation

### 4. Verify Pipeline
Run the test script to verify routing and loading:
```bash
python scripts/test_moe_pipeline.py
```

## 🔬 Testing the MoE System

Run:

```bash
python scripts/test_moe.py
```

This will:
- Load the MoE router
- Select the best expert per task
- Run inference using each expert
- Print router decisions + model output

