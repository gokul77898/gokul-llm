# RAG Module - Phase R1: Legal Chunking & Indexing

## Purpose

This module provides legal document ingestion and chunking infrastructure for the MoE Legal AI system.

**Phase R0** established:
- Canonical document schema for legal documents
- Text normalization for legal references
- Strict validation to ensure data quality
- Filesystem storage for auditable persistence

**Phase R1** adds:
- Legal-structure-aware chunking
- Section parsing for bare acts
- Paragraph parsing for case law
- Deterministic chunk IDs
- Chunk storage and index manifest

## What is NOT Implemented (Intentionally)

**No embeddings or retrieval yet.** These will be added in later phases.

| Feature | Status | Phase |
|---------|--------|-------|
| Embeddings | ❌ Not implemented | R2 |
| Vector databases (FAISS, ChromaDB) | ❌ Not implemented | R2 |
| Retrieval logic | ❌ Not implemented | R2 |
| LLM integration | ❌ Not implemented | R3 |
| BM25 / keyword search | ❌ Not implemented | R2 |
| Reranking | ❌ Not implemented | R3 |

## Directory Structure

```
src/rag/
├── __init__.py              # Module exports
├── README.md                # This file
├── schemas/
│   ├── __init__.py
│   ├── document.py          # Canonical LegalDocument schema
│   └── chunk.py             # LegalChunk schema (Phase R1)
├── ingestion/
│   ├── __init__.py
│   ├── loaders.py           # Document loading
│   ├── normalizer.py        # Legal text normalization
│   └── validator.py         # Strict validation
├── chunking/                # Phase R1
│   ├── __init__.py
│   ├── chunker.py           # Legal-structure-aware chunker
│   ├── section_parser.py    # Section extraction
│   └── id_generator.py      # Deterministic chunk IDs
├── storage/
│   ├── __init__.py
│   ├── filesystem.py        # Document JSON storage
│   └── chunk_storage.py     # Chunk JSON storage (Phase R1)
└── utils/
    ├── __init__.py
    └── text_cleaning.py     # Text processing utilities
```

## Canonical Document Schema

All legal documents must conform to the `LegalDocument` schema:

```python
from src.rag.schemas.document import LegalDocument, DocumentType

doc = LegalDocument(
    doc_id="a1b2c3d4e5f6g7h8",      # Deterministic (SHA256 hash)
    title="Indian Penal Code, 1860",
    doc_type=DocumentType.BARE_ACT,  # bare_act | case_law | amendment | notification
    act="IPC",                        # Required for bare_act, amendment
    court=None,                       # Required for case_law
    year=1860,
    citation="Act No. 45 of 1860",
    raw_text="Section 1. Title and extent...",
    source="ipc_1860.pdf",
    version=1,
    created_at="2024-01-01T00:00:00"
)
```

### Document Types

| Type | Description | Required Fields |
|------|-------------|-----------------|
| `bare_act` | Primary legislation | `act` |
| `case_law` | Court judgments | `court` |
| `amendment` | Legislative amendments | `act` |
| `notification` | Government notifications | - |

## Normalization Rules

The normalizer converts legal references to canonical forms:

### Section References
```
"Sec. 420"      → "Section 420"
"section 420"   → "Section 420"
"S.420"         → "Section 420"
"§ 420"         → "Section 420"
```

### Act Names
```
"Indian Penal Code"           → "IPC"
"Code of Criminal Procedure"  → "CrPC"
"Code of Civil Procedure"     → "CPC"
"Indian Evidence Act"         → "IEA"
```

## Validation Rules

Documents are **rejected** if:

1. `raw_text` is empty or < 50 characters
2. `title` is empty
3. `doc_type` is unknown
4. `year` is in the future or < 1800
5. `source` is empty
6. `doc_id` is not 16 hex characters
7. Section numbers are malformed (for bare_act)
8. `court` is missing (for case_law)
9. `act` is missing (for bare_act, amendment)

## Usage

### Ingest a Single Document

```python
from src.rag import load_document, DocumentType, FilesystemStorage

# Initialize storage
storage = FilesystemStorage("data/rag/documents")

# Load and persist a bare act
doc = load_document(
    source_path="data/raw/ipc_1860.txt",
    doc_type=DocumentType.BARE_ACT,
    title="Indian Penal Code, 1860",
    act="IPC",
    year=1860,
    storage=storage,
)

print(f"Ingested: {doc.doc_id}")
```

### Ingest from Raw Text

```python
from src.rag import load_from_text, DocumentType

doc = load_from_text(
    raw_text="Section 1. This Act shall be called...",
    source="manual_entry",
    doc_type=DocumentType.BARE_ACT,
    title="Sample Act",
    act="Sample Act",
)
```

### Batch Ingestion

```python
from src.rag.ingestion.loaders import ingest_directory
from src.rag import DocumentType, FilesystemStorage

storage = FilesystemStorage()
results = ingest_directory(
    directory="data/raw/bare_acts/",
    doc_type=DocumentType.BARE_ACT,
    storage=storage,
)

print(f"Processed: {results['processed']}")
print(f"Succeeded: {results['succeeded']}")
print(f"Failed: {results['failed']}")
```

## Storage

Documents are stored as JSON files:

```
data/rag/documents/
├── a1b2c3d4e5f6g7h8.json
├── b2c3d4e5f6g7h8i9.json
└── ...
```

### Versioning

- Documents cannot be overwritten without incrementing `version`
- Each save is atomic (write to temp file, then rename)
- Use `storage.load(doc_id)` to retrieve documents

## Chunk Schema (Phase R1)

Each chunk represents ONE legal unit:

```python
from src.rag import LegalChunk, DocumentType

chunk = LegalChunk(
    chunk_id="a1b2c3d4e5f6g7h8",      # Deterministic hash
    doc_id="parent_doc_id",
    act="IPC",
    section="420",
    subsection=None,                   # Or "(1)", "(a)", etc.
    doc_type=DocumentType.BARE_ACT,
    text="Whoever cheats and thereby...",
    citation="Section 420, IPC",
    court=None,
    year=1860,
    start_offset=1000,
    end_offset=1500,
    chunk_index=5,
)
```

### Chunk Fields

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | str | Deterministic hash (16 hex chars) |
| `doc_id` | str | Parent document ID |
| `act` | str \| null | Act name |
| `section` | str \| null | Section number |
| `subsection` | str \| null | Subsection identifier |
| `doc_type` | enum | Document type |
| `text` | str | Chunk text content |
| `citation` | str \| null | Legal citation |
| `court` | str \| null | Court name |
| `year` | int \| null | Year |
| `start_offset` | int | Start position in document |
| `end_offset` | int | End position in document |
| `chunk_index` | int | Position in document (0-indexed) |

## Chunking Rules

### Bare Acts
- Chunk by **Section**
- Subsections become separate chunks if present
- Explanations and Illustrations are separate chunks
- **NEVER** chunk by token length
- **NEVER** merge sections

### Case Law
- Chunk by **numbered paragraphs**
- Major headings (JUDGMENT, HELD, etc.) start new chunks
- Falls back to paragraph boundaries

### Example

```python
from src.rag import load_from_text, chunk_document, ChunkStorage, DocumentType

# Load document
doc = load_from_text(
    raw_text='''
    Section 420. Cheating and dishonestly inducing delivery of property.
    Whoever cheats and thereby dishonestly induces the person deceived...
    
    Section 421. Dishonest or fraudulent removal or concealment of property.
    Whoever dishonestly or fraudulently removes...
    ''',
    source="ipc_extract.txt",
    doc_type=DocumentType.BARE_ACT,
    title="IPC Extract",
    act="IPC",
)

# Chunk document
chunks = chunk_document(doc)
print(f"Created {len(chunks)} chunks")

# Store chunks
storage = ChunkStorage()
storage.save_many(chunks)
```

## Chunk Storage

Chunks are stored as JSON files:

```
data/rag/chunks/
├── a1b2c3d4e5f6g7h8.json
├── b2c3d4e5f6g7h8i9.json
├── ...
└── index.json              # Manifest file
```

### Index File

The `index.json` contains lightweight metadata for all chunks:

```json
{
  "version": 1,
  "updated_at": "2024-01-01T00:00:00",
  "chunk_count": 100,
  "chunks": {
    "a1b2c3d4e5f6g7h8": {
      "chunk_id": "a1b2c3d4e5f6g7h8",
      "doc_id": "parent_doc_id",
      "act": "IPC",
      "section": "420",
      "doc_type": "bare_act",
      "year": 1860
    }
  }
}
```

**Note:** This is NOT retrieval — just a manifest for chunk lookup.

## Phase R2: Dual Retrieval Engine

Phase R2 implements real retrieval using:
1. **BM25 (Sparse)** - Keyword-based retrieval with legal term boosting
2. **ChromaDB (Dense)** - Semantic similarity using sentence-transformers
3. **Fusion** - Deterministic combination of both methods

**No LLMs are used in Phase R2** - only embedding models for dense retrieval.

### BM25 Retrieval

```python
from src.rag import BM25Retriever

bm25 = BM25Retriever()
bm25.load()

results = bm25.query("Section 420 IPC", top_k=5)
for r in results:
    print(f"{r.section}: {r.score:.3f}")
```

Features:
- Prioritizes exact section numbers
- Boosts statute name matches
- Tokenizes legal terms specially

### Dense Retrieval (ChromaDB)

```python
from src.rag import DenseRetriever

dense = DenseRetriever()
dense.index_chunks()

results = dense.query("cheating offence punishment", top_k=5)
for r in results:
    print(f"{r.section}: {r.score:.3f}")
```

Features:
- Uses sentence-transformers embeddings
- Persists to disk (`data/rag/chromadb/`)
- Supports metadata filtering

### Fusion Retrieval

```python
from src.rag import LegalRetriever

retriever = LegalRetriever()
retriever.initialize()

# Fused retrieval (default)
results = retriever.retrieve("Section 420 IPC cheating", top_k=5)

for r in results:
    print(f"{r.section} ({r.source}): {r.score:.3f}")
    print(f"  BM25: {r.bm25_score}, Dense: {r.dense_score}")
```

Fusion logic:
- Normalizes scores from both sources
- Weighted combination (default: 40% BM25, 60% Dense)
- Boosts chunks matching query section/act
- Penalizes wrong statute matches

### RetrievedChunk Schema

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | str | Chunk identifier |
| `text` | str | Chunk content |
| `act` | str \| null | Act name |
| `section` | str \| null | Section number |
| `doc_type` | str | Document type |
| `citation` | str \| null | Legal citation |
| `court` | str \| null | Court name |
| `year` | int \| null | Year |
| `score` | float | Retrieval score |
| `source` | str | "bm25", "dense", or "fused" |
| `bm25_score` | float \| null | BM25 component score |
| `dense_score` | float \| null | Dense component score |

## Phase R3: Retrieval Validation & Filtering

Phase R3 prevents incorrect, weak, or unsafe evidence from reaching generation.

**No LLM is ever called with unvalidated evidence.**

### Validation Rules

1. **Statute Validation**
   - If query mentions IPC → Reject CrPC-only chunks
   - If query mentions CrPC → Reject IPC-only chunks
   - Unknown statute → Allow with penalty

2. **Section Consistency**
   - If query mentions section number → Chunk MUST contain same section
   - Otherwise → Chunk is discarded

3. **Repealed/Invalid Law Guard**
   - Reject chunks marked: `repealed=true`, `invalid=true`, `superseded=true`

4. **Evidence Threshold**
   - At least 2 chunks OR
   - One chunk with confidence ≥ 0.75
   - If threshold not met → **REFUSE**

### Usage

```python
from src.rag import LegalRetriever, RetrievalValidator

# Retrieve chunks
retriever = LegalRetriever()
retriever.initialize()
chunks = retriever.retrieve("Section 420 IPC", top_k=10)

# Validate before using
validator = RetrievalValidator()
result = validator.validate(
    query="Section 420 IPC",
    retrieved_chunks=[c.to_dict() for c in chunks],
)

if result.status.value == "pass":
    # Safe to use accepted_chunks
    for chunk in result.accepted_chunks:
        print(f"{chunk.section}: {chunk.adjusted_score:.3f}")
else:
    # REFUSE - do not proceed
    print(f"Refused: {result.refusal_reason}")
    print(f"Message: {result.refusal_message}")
```

### ValidationResult Schema

| Field | Type | Description |
|-------|------|-------------|
| `status` | enum | "pass" or "refuse" |
| `accepted_chunks` | list | Chunks that passed validation |
| `rejected_chunks` | list | Chunks that failed with reasons |
| `refusal_reason` | enum | Machine-readable reason |
| `refusal_message` | str | Human-readable message |

### Refusal Reasons

| Reason | Description |
|--------|-------------|
| `insufficient_evidence` | Not enough chunks |
| `statute_mismatch` | Wrong statute (IPC vs CrPC) |
| `section_mismatch` | Section not in chunk |
| `repealed_law` | Law is repealed |
| `low_confidence` | Scores too low |
| `no_valid_chunks` | All chunks rejected |

### Logging

All validation runs are logged to `logs/rag_validation.jsonl`:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "query": "Section 420 IPC",
  "total_retrieved": 10,
  "accepted_count": 3,
  "rejected_count": 7,
  "status": "pass",
  "refusal_reason": null
}
```

## Phase R4: Context Assembler

Phase R4 assembles validated evidence into a STRICT, bounded, auditable context.

**The decoder never sees raw documents — only assembled evidence blocks.**

### Context Format

```
EVIDENCE_START
[1] (IPC, Section 420, 1860)
Whoever cheats and thereby dishonestly induces...
SOURCE: Bare Act

[2] (Supreme Court, 2012, XYZ v State)
The court held that...
SOURCE: Case Law
EVIDENCE_END
```

### Token Budget Rules

- Default max tokens: 2,500
- If overflow: drop lowest-ranked chunks first
- Never split a chunk mid-sentence
- Minimum: at least 1 full chunk

### Ordering Rules

Chunks ordered by:
1. Exact section match
2. Statute match
3. Court hierarchy (SC > HC > others)
4. Retrieval score (descending)

### Citation Rules

Each chunk MUST include:
- Act / Court
- Section (if applicable)
- Year
- Source type

**If any required citation metadata missing → DROP chunk**

### Usage

```python
from src.rag import ContextAssembler

# After validation (Phase R3)
assembler = ContextAssembler()
result = assembler.assemble(
    query="Section 420 IPC",
    validated_chunks=[
        {"chunk_id": "abc", "text": "...", "section": "420", "act": "IPC", "year": 1860, "doc_type": "bare_act", "score": 0.9},
    ],
)

if result.status.value == "assembled":
    print(result.context_text)
    print(f"Tokens: {result.token_count}")
else:
    print(f"Refused: {result.refusal_reason}")
```

### ContextResult Schema

| Field | Type | Description |
|-------|------|-------------|
| `status` | enum | "assembled" or "refuse" |
| `context_text` | str | Assembled context |
| `used_chunks` | list | Chunks included |
| `dropped_chunks` | list | Chunks dropped with reasons |
| `token_count` | int | Total tokens |
| `refusal_reason` | enum | Reason if refused |

### Refusal Reasons

| Reason | Description |
|--------|-------------|
| `no_valid_evidence` | No chunks provided |
| `token_budget_exceeded` | No chunks fit budget |
| `missing_citation` | All chunks missing metadata |
| `insufficient_context` | Below minimum viable |

### Logging

All assembly runs logged to `logs/rag_context.jsonl`:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "query": "Section 420 IPC",
  "input_chunk_count": 5,
  "used_chunk_count": 3,
  "dropped_chunk_count": 2,
  "token_count": 1200,
  "status": "assembled",
  "refusal_reason": null
}
```

## How This Feeds Later Phases

| Phase | Uses From R0/R1/R2/R3/R4 |
|-------|-------------------------|
| R5: RAG Answering | `ContextResult.context_text` → decoder input |
| R5: MoE Integration | Assembled context → decoder prompt |

## Non-Goals

This module does **NOT**:
- Connect to inference code
- Call LLMs or decoders
- Generate answers
- Modify MoE behavior

## Testing

```bash
# Test document ingestion and chunking
python -c "
from src.rag import load_from_text, chunk_document, ChunkStorage, DocumentType

doc = load_from_text(
    raw_text='''
    Section 1. Short title.
    This Act may be called the Test Act.
    
    Section 2. Definitions.
    In this Act, unless the context otherwise requires:
    (a) 'person' means any individual;
    (b) 'property' means movable or immovable property.
    ''',
    source='test.txt',
    doc_type=DocumentType.BARE_ACT,
    title='Test Act',
    act='Test Act',
)
print(f'Document: {doc.doc_id}')

chunks = chunk_document(doc)
print(f'Chunks: {len(chunks)}')

storage = ChunkStorage()
saved = storage.save_many(chunks)
print(f'Saved: {saved} chunks')
"

# Test retrieval (Phase R2)
python -c "
from src.rag import LegalRetriever

retriever = LegalRetriever()
stats = retriever.initialize()
print(f'Initialized: {stats}')

results = retriever.retrieve('Section 420 IPC', top_k=3)
for r in results:
    print(f'{r.section} ({r.source}): {r.score:.3f}')
"

# Test validation (Phase R3)
python -c "
from src.rag import LegalRetriever, RetrievalValidator

retriever = LegalRetriever()
retriever.initialize()

chunks = retriever.retrieve('Section 420 IPC', top_k=5)
validator = RetrievalValidator()
result = validator.validate(
    query='Section 420 IPC',
    retrieved_chunks=[{'chunk_id': c.chunk_id, 'text': c.text, 'section': c.section, 'act': c.act, 'score': c.score} for c in chunks],
)
print(f'Status: {result.status.value}')
print(f'Accepted: {len(result.accepted_chunks)}')
print(f'Rejected: {len(result.rejected_chunks)}')
"

# Test context assembly (Phase R4)
python -c "
from src.rag import ContextAssembler

assembler = ContextAssembler()
result = assembler.assemble(
    query='Section 420 IPC',
    validated_chunks=[
        {'chunk_id': 'test1', 'text': 'Whoever cheats...', 'section': '420', 'act': 'IPC', 'year': 1860, 'doc_type': 'bare_act', 'score': 0.9},
    ],
)
print(f'Status: {result.status.value}')
print(f'Tokens: {result.token_count}')
print(result.context_text)
"
```
