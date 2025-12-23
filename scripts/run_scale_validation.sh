#!/bin/bash
# Master Orchestration Script for Scale Validation Pipeline
# Runs complete ingestion pipeline: Reset → Prep → Ingest → Chunk → Index → Verify

set -e  # Exit on any error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "RAG SCALE VALIDATION PIPELINE"
echo "========================================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# ============================================================================
# PHASE A: HARD RESET
# ============================================================================

echo "========================================================================"
echo "PHASE A: HARD RESET"
echo "========================================================================"
echo ""

bash scripts/hard_reset.sh
if [ $? -ne 0 ]; then
    echo "ERROR: Hard reset failed"
    exit 1
fi

echo ""

# ============================================================================
# PHASE B: DATASET PREPARATION
# ============================================================================

echo "========================================================================"
echo "PHASE B: DATASET PREPARATION"
echo "========================================================================"
echo ""

python3 scripts/prepare_dataset.py
if [ $? -ne 0 ]; then
    echo "ERROR: Dataset preparation failed"
    echo ""
    echo "Please follow the instructions above to prepare the dataset."
    exit 1
fi

echo ""

# ============================================================================
# PHASE C: INGESTION PIPELINE
# ============================================================================

echo "========================================================================"
echo "PHASE C: INGESTION PIPELINE"
echo "========================================================================"
echo ""

# Step 1: Ingest
echo "Step 1: Ingesting documents..."
echo "----------------------------------------------------------------------"
python3 scripts/ingest.py
if [ $? -ne 0 ]; then
    echo "ERROR: Ingestion failed"
    exit 1
fi
echo ""
echo "✓ Ingestion complete"
echo ""

# Step 2: Chunk
echo "Step 2: Chunking documents..."
echo "----------------------------------------------------------------------"
python3 scripts/chunk.py
if [ $? -ne 0 ]; then
    echo "ERROR: Chunking failed"
    exit 1
fi
echo ""
echo "✓ Chunking complete"
echo ""

# Step 3: Index
echo "Step 3: Indexing chunks..."
echo "----------------------------------------------------------------------"
python3 scripts/index.py
if [ $? -ne 0 ]; then
    echo "ERROR: Indexing failed"
    exit 1
fi
echo ""
echo "✓ Indexing complete"
echo ""

# ============================================================================
# PHASE D: VERIFICATION
# ============================================================================

echo "========================================================================"
echo "PHASE D: VERIFICATION AT SCALE"
echo "========================================================================"
echo ""

python3 scripts/validate_scale.py
if [ $? -ne 0 ]; then
    echo "ERROR: Validation failed"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✓ PIPELINE COMPLETE"
echo "========================================================================"
echo ""

exit 0
