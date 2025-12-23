#!/bin/bash
# PHASE A: Hard Reset - Delete ALL previous state

set -e  # Exit on any error

echo "========================================================================"
echo "PHASE A: HARD RESET"
echo "========================================================================"
echo ""

# Define directories to clean
DIRS_TO_CLEAN=(
    "data/rag/raw"
    "data/rag/documents"
    "data/rag/chunks"
    "chromadb"
    "db_store"
    "cache"
)

# Find and remove all __pycache__ directories
echo "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "✓ __pycache__ directories removed"
echo ""

# Clean each directory
for dir in "${DIRS_TO_CLEAN[@]}"; do
    if [ -d "$dir" ]; then
        echo "Cleaning: $dir"
        rm -rf "$dir"/*
        echo "✓ $dir cleaned"
    else
        echo "⚠ Directory does not exist: $dir (will be created)"
    fi
done

echo ""
echo "Verifying directories are empty..."
echo ""

# Verify each directory is empty
ALL_CLEAN=true
for dir in "${DIRS_TO_CLEAN[@]}"; do
    if [ -d "$dir" ]; then
        FILE_COUNT=$(find "$dir" -type f 2>/dev/null | wc -l | tr -d ' ')
        if [ "$FILE_COUNT" -gt 0 ]; then
            echo "✗ ERROR: $dir still contains $FILE_COUNT files"
            ALL_CLEAN=false
        else
            echo "✓ $dir is empty"
        fi
    fi
done

echo ""

if [ "$ALL_CLEAN" = false ]; then
    echo "========================================================================"
    echo "ERROR: Hard reset failed - some directories still contain files"
    echo "========================================================================"
    exit 1
fi

echo "========================================================================"
echo "✓ HARD RESET COMPLETE - All directories cleaned"
echo "========================================================================"
echo ""

exit 0
