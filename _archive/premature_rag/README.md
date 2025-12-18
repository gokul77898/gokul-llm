# Archived Premature RAG Code

**Phase R0: RAG Foundations**

This directory contains RAG implementation code that was prematurely added before Phase R0 foundations were established.

## Why Archived?

The existing RAG code included:
- FAISS vector store
- ChromaDB integration
- Embedding-based retrieval
- LLM-based generation

Phase R0 requires a clean foundation **without** embeddings, vector databases, or retrieval logic.

## Files Archived

- `__init__.py` - RAG module exports
- `document_store.py` - FAISS/ChromaDB stores
- `retriever.py` - Embedding-based retrieval
- `generator.py` - LLM generation
- `grounded_generator.py` - Grounded generation
- `pipeline.py` - Full RAG pipeline
- `reranker.py` - Document reranking
- `indexer.py` - Document indexing
- `eval.py` - RAG evaluation

## When to Restore

These files may be referenced when implementing:
- Phase R1: Chunking and embeddings
- Phase R2: Vector storage
- Phase R3: Retrieval logic

## DO NOT

- Import these files from active code
- Use as authoritative implementation
- Restore without Phase R1+ approval
