#!/bin/bash
set -e

echo "========================================================================"
echo "RAG PIPELINE VALIDATION - 50MB DATASET"
echo "========================================================================"
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "========================================================================"
echo "PHASE A: HARD RESET"
echo "========================================================================"
echo ""

# Kill any running processes
pkill -f "ingest.py" || true
pkill -f "chunk.py" || true
pkill -f "index.py" || true
sleep 2

# Clean directories using find (handles large file counts)
echo "Cleaning data directories..."
find data/rag/raw -type f -delete 2>/dev/null || true
rm -rf data/rag/documents/* 2>/dev/null || true
rm -rf data/rag/chunks/* 2>/dev/null || true
rm -rf data/rag/chroma/* 2>/dev/null || true

# Recreate directories
mkdir -p data/rag/raw
mkdir -p data/rag/documents
mkdir -p data/rag/chunks
mkdir -p data/rag/chroma

echo "✓ Hard reset complete"
echo ""

echo "========================================================================"
echo "PHASE B: DATASET GENERATION (50MB)"
echo "========================================================================"
echo ""

python scripts/generate_50mb_dataset.py

if [ $? -ne 0 ]; then
    echo "✗ Dataset generation failed"
    exit 1
fi

echo ""
echo "========================================================================"
echo "PHASE C: INGESTION PIPELINE"
echo "========================================================================"
echo ""

echo "Step 1: Ingesting documents..."
python scripts/ingest.py

if [ $? -ne 0 ]; then
    echo "✗ Ingestion failed"
    exit 1
fi

echo ""
echo "Step 2: Chunking documents..."
python scripts/chunk.py

if [ $? -ne 0 ]; then
    echo "✗ Chunking failed"
    exit 1
fi

echo ""
echo "Step 3: Indexing chunks..."
python scripts/index.py

if [ $? -ne 0 ]; then
    echo "✗ Indexing failed"
    exit 1
fi

echo ""
echo "========================================================================"
echo "PHASE D: VERIFICATION"
echo "========================================================================"
echo ""

python scripts/verify_50mb.py

if [ $? -ne 0 ]; then
    echo "✗ Verification failed"
    exit 1
fi

echo ""
echo "========================================================================"
echo "PIPELINE COMPLETE"
echo "========================================================================"
