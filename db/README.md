# ChromaDB Database Layer

Complete vector database system for legal document ingestion and retrieval.

## Features

- ✅ Persistent ChromaDB storage
- ✅ Universal file ingestion (PDF, TXT, DOCX, HTML)
- ✅ Intelligent text chunking with overlap
- ✅ Sentence transformer embeddings
- ✅ Vector similarity search
- ✅ Metadata tracking

## Installation

```bash
pip install -r requirements_chroma.txt
```

## Quick Start

```python
from db.chroma import ingest_file, VectorRetriever

# Ingest documents
ingest_file("document.pdf", collection_name="legal_docs")

# Search
retriever = VectorRetriever("legal_docs")
results = retriever.query("What is appropriate government?", top_k=5)
```

## Test

```bash
python app_test.py
```
