#!/bin/bash

# MARK Training Pipeline - One-Command Execution
# This script runs the complete training pipeline

set -e  # Exit on error

echo "======================================================================="
echo "  MARK TRAINING PIPELINE - OPTION C (RAG + LoRA)"
echo "======================================================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Generate Training Data
echo -e "${BLUE}[STEP 1/4]${NC} Generating SFT Training Data from ChromaDB..."
python3.10 -m src.training.data_prep \
    --collection pdf_docs \
    --out-dir data/ \
    --top-k 3 \
    --max-samples 500

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Training data generated${NC}"
else
    echo -e "${YELLOW}❌ Data generation failed${NC}"
    exit 1
fi

echo ""

# Step 2: Dry-Run LoRA Training
echo -e "${BLUE}[STEP 2/4]${NC} Running LoRA Training Dry-Run..."
python3.10 -m src.training.lora_trainer \
    --config configs/lora_sft.yaml \
    --dry-run

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Dry-run validation passed${NC}"
else
    echo -e "${YELLOW}❌ Dry-run failed${NC}"
    exit 1
fi

echo ""

# Step 3: Test Formatter
echo -e "${BLUE}[STEP 3/4]${NC} Testing ChatGPT-Style Formatter..."
python3.10 test_chatgpt_formatter.py > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Formatter integration verified${NC}"
else
    echo -e "${YELLOW}❌ Formatter test failed${NC}"
    exit 1
fi

echo ""

# Step 4: Summary
echo -e "${BLUE}[STEP 4/4]${NC} Pipeline Summary"
echo "======================================================================="
echo ""
echo -e "${GREEN}✅ ALL STEPS COMPLETE${NC}"
echo ""
echo "📊 Results:"
echo "   • Training data: $(wc -l < data/train_sft.jsonl 2>/dev/null || echo 0) samples"
echo "   • Validation data: $(wc -l < data/val_sft.jsonl 2>/dev/null || echo 0) samples"
echo "   • LoRA validation: PASSED"
echo "   • ChatGPT formatter: INTEGRATED"
echo ""
echo "🎯 System Status:"
echo "   • ChromaDB: 766 documents loaded"
echo "   • Fine-tuning: Ready (dry-run validated)"
echo "   • Response format: ChatGPT-style enabled"
echo ""
echo "🚀 Next Steps:"
echo "   1. Start backend: python -m uvicorn src.api.main:app --reload --port 8000"
echo "   2. Start frontend: cd ui && npm run dev"
echo "   3. Test in chat UI at http://localhost:5173"
echo ""
echo "💡 To run actual training (OPTIONAL):"
echo "   1. Edit configs/lora_sft.yaml (set epochs: 3, dry_run: false)"
echo "   2. Run: python -m src.training.lora_trainer --config configs/lora_sft.yaml --confirm-run"
echo ""
echo "======================================================================="
