# RAG Scale Validation Guide

## Overview

Complete pipeline for validating RAG ingestion at scale with ~5GB of Indian Supreme Court judgments.

**Scope:** Ingestion + Chunking + Indexing + Verification ONLY  
**No:** Training, model changes, LLM usage, or C3 evaluation

## Pipeline Phases

### Phase A: Hard Reset

**Purpose:** Clean all previous state

**Script:** `scripts/hard_reset.sh`

**Actions:**
- Deletes all files in:
  - `data/rag/raw`
  - `data/rag/documents`
  - `data/rag/chunks`
  - `chromadb`
  - `db_store`
  - `cache`
  - All `__pycache__` directories
- Verifies directories are empty
- Aborts if any files remain

**Run:**
```bash
bash scripts/hard_reset.sh
```

### Phase B: Dataset Preparation

**Purpose:** Prepare and validate ~5GB dataset

**Script:** `scripts/prepare_dataset.py`

**Requirements:**

1. **Dataset:** Indian Kanoon Supreme Court Judgments (plain text)
2. **Size:** 4.5 GB - 5.5 GB
3. **Format:** UTF-8 encoded `.txt` files only
4. **Location:** `data/rag/raw/`

**File Format (MANDATORY):**

Each file MUST begin with:
```
ACT: IPC | CrPC | CPC | Evidence Act | Other
SECTION: <section number or NA>
TYPE: case_law
COURT: Supreme Court of India
YEAR: <YYYY>

<blank line>
<judgment text>
```

**Actions:**
- Creates `data/rag/raw/` directory
- Validates dataset size (4.5-5.5 GB)
- Validates file format:
  - UTF-8 encoding
  - `.txt` extension
  - Required metadata headers
  - No HTML tags
  - No empty files
- Removes invalid files
- Reports statistics

**Run:**
```bash
python3 scripts/prepare_dataset.py
```

**Output:**
```
Valid files: 10,234
Total size: 5.12 GB
Average file size: 512 KB
✓ Dataset ready for ingestion
```

### Phase C: Ingestion Pipeline

**Purpose:** Ingest, chunk, and index documents

**Scripts:**
1. `scripts/ingest.py` - Document ingestion
2. `scripts/chunk.py` - Legal-aware chunking
3. `scripts/index.py` - Vector indexing

**Step 1: Ingest**

```bash
python3 scripts/ingest.py
```

**Requirements:**
- Non-empty `raw_text`
- Document version = 1
- Deterministic `doc_id`
- `semantic_id` created
- Zero failures allowed

**Step 2: Chunk**

```bash
python3 scripts/chunk.py
```

**Requirements:**
- SectionParser applied
- Legal-aware chunking
- `semantic_id` format: `<ACT>_<SECTION>_<INDEX>`
- No duplicate `chunk_id`
- No duplicate `semantic_id`

**Step 3: Index**

```bash
python3 scripts/index.py
```

**Requirements:**
- All chunks indexed
- ChromaDB metadata includes `semantic_id`
- Indexed vector count == chunk count

### Phase D: Verification

**Purpose:** Validate pipeline at scale

**Script:** `scripts/validate_scale.py`

**Actions:**

1. **Collect Statistics:**
   - Raw documents count
   - Canonical documents count
   - Total chunks
   - Total indexed vectors
   - Disk usage (GB)

2. **Run Sample Queries (Retrieval Only):**
   - "punishment under section 420 IPC"
   - "definition of employer minimum wages act"
   - "supreme court cheating offence"
   - "procedure under crpc"
   - "labour law employer liability"

3. **For Each Query, Print:**
   - `semantic_id`
   - `act`
   - `section`
   - `source_doc`
   - Distance score

4. **Verify:**
   - No hallucinated metadata
   - No missing act/section
   - No cross-document leakage
   - Deterministic results on rerun

**Run:**
```bash
python3 scripts/validate_scale.py
```

**Output:**

**Summary Table:**
```
======================================================================
SUMMARY TABLE
======================================================================
Raw documents:        10,234
Canonical documents:  10,234
Total chunks:         45,678
Indexed vectors:      45,678
Raw data size:        5.12 GB
Total disk usage:     8.45 GB
======================================================================
```

**Verification Checklist:**
```
======================================================================
VERIFICATION CHECKLIST
======================================================================
✓ Ingestion.................... PASS
✓ Chunking..................... PASS
✓ Indexing..................... PASS
✓ Metadata integrity........... PASS
✓ Scale handling............... PASS
======================================================================
```

**Final Line:**
```
======================================================================
RAG INGESTION PIPELINE VALIDATED FOR SCALE
======================================================================
```

## Master Orchestration Script

**Purpose:** Run complete pipeline end-to-end

**Script:** `scripts/run_scale_validation.sh`

**Run:**
```bash
bash scripts/run_scale_validation.sh
```

**Pipeline Flow:**
```
Phase A: Hard Reset
    ↓
Phase B: Dataset Preparation
    ↓
Phase C: Ingestion Pipeline
    ├─ Step 1: Ingest
    ├─ Step 2: Chunk
    └─ Step 3: Index
    ↓
Phase D: Verification
    ↓
✓ PIPELINE COMPLETE
```

## Dataset Preparation Instructions

### Step 1: Obtain Dataset

**Source:** Indian Kanoon Supreme Court Judgments

**Options:**
1. Download from Indian Kanoon API/website
2. Use public judicial records archive
3. Extract from legal databases

**Format:** Plain text (`.txt` files)

### Step 2: Prepare Files

For each judgment file, add metadata header:

```
ACT: IPC
SECTION: 420
TYPE: case_law
COURT: Supreme Court of India
YEAR: 2020

State of Maharashtra vs. Rajesh Kumar

[Judgment text begins here...]
```

**Metadata Fields:**

- **ACT:** Primary act referenced (IPC, CrPC, CPC, Evidence Act, Other)
- **SECTION:** Section number or "NA" if not applicable
- **TYPE:** Always "case_law" for judgments
- **COURT:** Always "Supreme Court of India"
- **YEAR:** Year of judgment (YYYY)

### Step 3: Place Files

```bash
# Create directory
mkdir -p data/rag/raw

# Copy files
cp /path/to/judgments/*.txt data/rag/raw/

# Check size
du -sh data/rag/raw
# Target: 4.5G - 5.5G
```

### Step 4: Subset to ~5GB

If dataset is too large:

```bash
# Count files
ls data/rag/raw/*.txt | wc -l

# Check size
du -sh data/rag/raw

# Remove excess files if needed
cd data/rag/raw
ls -S *.txt | tail -n 1000 | xargs rm  # Remove 1000 largest files

# Verify size
du -sh .
```

## Verification Criteria

### Ingestion: PASS

- ✓ Raw documents count > 0
- ✓ Canonical documents count == raw documents count
- ✓ All documents have valid metadata
- ✓ No ingestion failures

### Chunking: PASS

- ✓ Total chunks > canonical documents
- ✓ All chunks have `semantic_id`
- ✓ `semantic_id` format: `<ACT>_<SECTION>_<INDEX>`
- ✓ No duplicate chunk IDs
- ✓ No duplicate semantic IDs

### Indexing: PASS

- ✓ Indexed vectors count > 0
- ✓ Indexed vectors count == total chunks
- ✓ All metadata preserved in ChromaDB
- ✓ Retrieval returns valid results

### Metadata Integrity: PASS

- ✓ No missing `semantic_id`
- ✓ No missing `act`
- ✓ No missing `section`
- ✓ No missing `source_doc`
- ✓ No hallucinated metadata

### Scale Handling: PASS

- ✓ Raw data size: 4.5 GB - 5.5 GB
- ✓ Total chunks > 10,000
- ✓ Pipeline completes without OOM
- ✓ Retrieval performance acceptable

## Troubleshooting

### Dataset Preparation Fails

**Issue:** "No dataset found in data/rag/raw/"

**Solution:**
1. Download Indian Kanoon Supreme Court judgments
2. Add metadata headers to each file
3. Place files in `data/rag/raw/`
4. Re-run `python3 scripts/prepare_dataset.py`

**Issue:** "Size outside target range"

**Solution:**
```bash
# Check current size
du -sh data/rag/raw

# If too large, remove files
cd data/rag/raw
ls -S *.txt | tail -n 500 | xargs rm

# If too small, add more files
# Re-run preparation script
```

### Ingestion Fails

**Issue:** "Missing metadata headers"

**Solution:**
- Ensure all files have required headers:
  - ACT:
  - SECTION:
  - TYPE:
  - COURT:
  - YEAR:

**Issue:** "Not UTF-8 encoded"

**Solution:**
```bash
# Convert to UTF-8
iconv -f ISO-8859-1 -t UTF-8 file.txt > file_utf8.txt
```

### Chunking Fails

**Issue:** "Duplicate semantic_id"

**Solution:**
- Check for duplicate files in raw directory
- Ensure unique section/index combinations

### Indexing Fails

**Issue:** "ChromaDB collection not found"

**Solution:**
```bash
# Verify ChromaDB directory
ls -la chromadb/

# Re-run indexing
python3 scripts/index.py
```

### Verification Fails

**Issue:** "Metadata integrity: FAIL"

**Solution:**
- Check ingestion logs for metadata extraction errors
- Verify file format compliance
- Re-run pipeline from Phase A

## Performance Expectations

### Dataset Size: ~5 GB

**Expected Metrics:**
- Raw documents: 8,000 - 15,000
- Total chunks: 30,000 - 60,000
- Indexed vectors: 30,000 - 60,000
- Total disk usage: 7 - 10 GB
- Processing time: 30 - 90 minutes (depending on hardware)

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Disk: 15 GB free space

**Recommended:**
- CPU: 8+ cores
- RAM: 16+ GB
- Disk: 20+ GB free space
- SSD for faster I/O

## Files Created

1. **`scripts/hard_reset.sh`** - Phase A: Hard reset script
2. **`scripts/prepare_dataset.py`** - Phase B: Dataset preparation
3. **`scripts/validate_scale.py`** - Phase D: Verification script
4. **`scripts/run_scale_validation.sh`** - Master orchestration script
5. **`SCALE_VALIDATION_GUIDE.md`** - This documentation

## Quick Start

```bash
# 1. Prepare dataset (manual step)
# - Download Indian Kanoon Supreme Court judgments
# - Add metadata headers
# - Place in data/rag/raw/
# - Ensure size is 4.5-5.5 GB

# 2. Run complete pipeline
bash scripts/run_scale_validation.sh

# Pipeline will:
# - Hard reset all state
# - Validate dataset
# - Ingest documents
# - Chunk documents
# - Index chunks
# - Verify at scale
# - Print summary and checklist
```

## Success Criteria

Pipeline succeeds if:

✓ All phases complete without errors  
✓ All verification checks PASS  
✓ Final line printed: "RAG INGESTION PIPELINE VALIDATED FOR SCALE"  

## Constraints

**DO NOT:**
- Train models
- Modify retrieval logic
- Use LLMs
- Add C3 or evaluation
- Change configs beyond `phase1_rag.yaml`

**ONLY:**
- Ingestion
- Chunking
- Indexing
- Verification

## Summary

This pipeline validates the RAG ingestion system at scale with ~5GB of real Indian legal text. It ensures:

- Correct metadata extraction
- Legal-aware chunking
- Proper indexing
- Metadata integrity
- Scale handling

All without training models or using LLMs.
