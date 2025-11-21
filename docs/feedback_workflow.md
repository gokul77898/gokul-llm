# Feedback & Continuous Improvement Workflow

## System Components

1. **Grounded Answer Generator** - Reduces hallucinations by grounding answers in retrieved documents
2. **Feedback Worker** - Processes user feedback and builds SFT training buffer
3. **Safe SFT Training** - Validates before deployment
4. **Checkpoint Promotion** - Manual approval required

## User Flow

1. User asks question
2. System retrieves docs, reranks, generates grounded answer
3. If grounding_score < 0.30, uses extractive fallback
4. User can mark incorrect and provide correction
5. Feedback saved to `feedback/incoming.jsonl`
6. Worker routes to SFT buffer or review queue
7. When 20 examples collected, creates SFT bundle
8. Operator runs training manually
9. Validation required before promotion
10. Manual checkpoint promotion to production

## CLI Commands

### View Buffer
```bash
wc -l feedback/sft_buffer.jsonl
```

### Trigger SFT Bundle
```bash
curl -X POST http://localhost:8000/retrain/trigger
```

### Run SFT Training
```bash
python -m src.training.sft_train \
  --data feedback/sft_bundle_TIMESTAMP.jsonl \
  --out checkpoints/rlhf/sft/sft_incremental_TIMESTAMP.pt \
  --epochs 2 --batch_size 4
```

### Promote Checkpoint
```bash
python -m src.core.promote_checkpoint \
  --from checkpoints/rlhf/sft/sft_incremental_TIMESTAMP.pt \
  --to rl_trained
```

## Safety Features

- ✅ Grounding score prevents hallucinations
- ✅ Extractive fallback when confidence low
- ✅ Human review queue for ambiguous feedback
- ✅ Manual SFT trigger only
- ✅ Validation before deployment
- ✅ Operator approval required for production

## API Endpoints

- `POST /feedback` - Submit user feedback
- `GET /feedback/review` - View review queue
- `POST /retrain/trigger` - Create SFT bundle

## Monitoring

Logs at:
- `logs/qa_events.log` - All QA interactions
- `logs/auto_retrain.log` - Training events
- `feedback/` - All feedback data
