# Phase-1 RAG Operational Baseline

A clean, reproducible operational baseline for the Legal RAG system.

## Directory Contract

```
data/rag/
├── raw/           # Raw input documents (.txt with metadata headers)
├── documents/     # Canonical JSON documents (after ingestion)
├── chunks/        # Legal-structure-aware chunks (after chunking)
└── chromadb/      # Vector index (after indexing)

configs/
└── phase1_rag.yaml   # All configuration values

scripts/
├── ingest.py      # Step 1: Raw → Documents
├── chunk.py       # Step 2: Documents → Chunks
├── index.py       # Step 3: Chunks → ChromaDB
└── query.py       # Step 4: Query the index
```

## Configuration

All hardcoded values are centralized in `configs/phase1_rag.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `encoder.model_name` | `BAAI/bge-large-en-v1.5` | Embedding model |
| `decoder.model_name` | `Qwen/Qwen2.5-32B-Instruct` | Generation model |
| `retrieval.top_k` | `5` | Default retrieval count |
| `retrieval.collection_name` | `legal_chunks` | ChromaDB collection |

## Quick Start

### 1. Prepare Raw Documents

Place `.txt` files in `data/rag/raw/` with metadata headers:

```
ACT: Minimum Wages Act
YEAR: 1948
TYPE: bare_act

Section 1. Short title, extent and commencement.
(1) This Act may be called the Minimum Wages Act, 1948.
(2) It extends to the whole of India.
...
```

Supported document types:
- `bare_act` - Legislation text
- `case_law` - Court judgments
- `amendment` - Amendment acts
- `notification` - Government notifications

### 2. Run the Pipeline

```bash
# Step 1: Ingest raw documents
python scripts/ingest.py

# Step 2: Chunk documents into legal units
python scripts/chunk.py

# Step 3: Index chunks into ChromaDB
python scripts/index.py

# Step 4: Query the index
python scripts/query.py "What is the minimum wage?"
```

### 3. Interactive Query Mode

```bash
python scripts/query.py --interactive
```

## Script Reference

### `scripts/ingest.py`

Ingests raw documents from `data/rag/raw/` into `data/rag/documents/`.

```bash
python scripts/ingest.py
python scripts/ingest.py --config configs/phase1_rag.yaml
```

**Output:**
- Canonical JSON documents in `data/rag/documents/`
- Each document has a deterministic `doc_id`

### `scripts/chunk.py`

Chunks documents into legal-structure-aware units.

```bash
python scripts/chunk.py
python scripts/chunk.py --config configs/phase1_rag.yaml
```

**Output:**
- JSON chunks in `data/rag/chunks/`
- Each chunk = ONE legal unit (section, subsection, paragraph)
- Deterministic `chunk_id` for each chunk

### `scripts/index.py`

Creates vector embeddings and indexes into ChromaDB.

```bash
python scripts/index.py
python scripts/index.py --rebuild  # Recreate from scratch
```

**Output:**
- ChromaDB collection in `data/rag/chromadb/`
- Uses `BAAI/bge-large-en-v1.5` embeddings

### `scripts/query.py`

Queries the indexed chunks.

```bash
# Single query
python scripts/query.py "What is the definition of employer?"

# With options
python scripts/query.py --top-k 10 "minimum wages"
python scripts/query.py --filter-act "Minimum Wages Act" "employer definition"

# JSON output
python scripts/query.py --json "query text"

# Interactive mode
python scripts/query.py --interactive
```

## Determinism Guarantees

1. **Document IDs** - Generated from content hash + source
2. **Chunk IDs** - Generated from doc_id + section + offset
3. **Embeddings** - Same model produces same vectors
4. **No shared state** - Each script reads from disk, writes to disk

## Fresh Environment Test

To verify the pipeline works in a fresh environment:

```bash
# Clear all data
rm -rf data/rag/documents/* data/rag/chunks/* data/rag/chromadb/*

# Run full pipeline
python scripts/ingest.py && \
python scripts/chunk.py && \
python scripts/index.py && \
python scripts/query.py "test query"
```

## Logging

Each script outputs:
- Timestamps for all operations
- Counts (processed, succeeded, failed, skipped)
- Paths (input, output directories)
- Version from config

Example output:
```
============================================================
Phase-1 RAG: Document Ingestion
============================================================
[2025-01-15 10:30:00] Config: configs/phase1_rag.yaml
[2025-01-15 10:30:00] Version: 1.0.0
[2025-01-15 10:30:00] Raw directory: data/rag/raw
[2025-01-15 10:30:00] Documents directory: data/rag/documents
[2025-01-15 10:30:00] Found 5 raw files

[2025-01-15 10:30:01] ✓ Ingested: minimum_wages_act.txt -> a1b2c3d4e5f6...
...
============================================================
INGESTION SUMMARY
============================================================
[2025-01-15 10:30:02] Processed: 5
[2025-01-15 10:30:02] Succeeded: 5
[2025-01-15 10:30:02] Failed:    0
```

## Dependencies

```
pyyaml
chromadb
sentence-transformers
pydantic
```

## Troubleshooting

### "No raw files found"
Place `.txt` files with metadata headers in `data/rag/raw/`

### "No documents found"
Run `python scripts/ingest.py` first

### "Index is empty"
Run `python scripts/index.py` first

### "ChromaDB not found"
Run `python scripts/index.py` to create the index

## Phase-1 Scope

**Included:**
- Document ingestion with validation
- Legal-structure-aware chunking
- Dense vector indexing (ChromaDB)
- Semantic search queries

**Not Included (Phase-2+):**
- LLM generation
- BM25 sparse retrieval
- Hybrid fusion
- Validation/refusal logic
- Context assembly
