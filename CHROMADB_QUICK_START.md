# ChromaDB System - Quick Start Guide

## ✅ What You Have

**9 Python modules** in `db/chroma/` providing complete vector database functionality.

---

## 📦 Install

```bash
pip install chromadb sentence-transformers pypdf python-docx beautifulsoup4
```

Or:
```bash
pip install -r requirements_chroma.txt
```

---

## 🚀 Test It Now

```bash
python app_test.py
```

This will:
1. Create ChromaDB collection
2. Ingest sample legal document  
3. Run 4 test queries
4. Show results with scores

---

## 💻 Basic Usage

### Ingest a Document
```python
from db.chroma import ingest_file

stats = ingest_file("document.pdf", "legal_docs")
print(stats.summary())
# Output: File: document.pdf | Chunks: 25 | Pages: 10 | Time: 2.1s
```

### Search
```python
from db.chroma import VectorRetriever

retriever = VectorRetriever("legal_docs")
results = retriever.query("What is appropriate government?", top_k=5)

for r in results:
    print(f"{r.score:.3f} | {r.get_source()} (p.{r.get_page()}) | {r.text[:100]}")
```

### Batch Ingest
```python
from db.chroma import ingest_directory

ingest_directory("data/legal_docs/", "legal_docs", recursive=True)
```

---

## 📂 File Breakdown

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | ~70 | Settings & configuration |
| `client.py` | ~130 | ChromaDB client manager |
| `schema.py` | ~160 | Data structures |
| `extractor.py` | ~240 | Universal file loader |
| `chunker.py` | ~180 | Text chunking |
| `embeddings.py` | ~60 | Sentence transformers |
| `ingestion.py` | ~110 | Full pipeline |
| `retriever.py` | ~90 | Vector search |
| **TOTAL** | **~1,040** | **Complete system** |

---

## 🎯 Supported File Types

- ✅ **PDF** (with page tracking)
- ✅ **TXT**
- ✅ **DOCX**
- ✅ **HTML**

---

## 🔧 Key Configuration

```python
# Default settings (can override)
CHROMA_DB_PATH = "db_store/chroma"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
```

---

## 📊 What Gets Stored

Every chunk includes:
```json
{
  "id": "md5_hash",
  "text": "chunk content",
  "metadata": {
    "source": "filename.pdf",
    "page": 5,
    "chunk_index": 12,
    "total_chunks": 45,
    "char_count": 487,
    "word_count": 89,
    "timestamp": "2025-11-18T14:59:00"
  }
}
```

---

## 🔍 Retrieval Results

```python
RetrievalResult(
    id="abc123...",
    text="According to the Minimum Wages Act...",
    score=0.9234,  # Similarity score (0-1, higher better)
    metadata={...},
    distance=0.082  # Lower is better
)
```

---

## 🛠️ Advanced Usage

### Filter by Metadata
```python
results = retriever.query(
    "employer obligations",
    top_k=5,
    filters={"source": "minimum_wages_act.pdf", "page": 5}
)
```

### Get Collection Stats
```python
from db.chroma import ChromaDBClient

client = ChromaDBClient()
info = client.get_collection_info("legal_docs")
print(f"Documents: {info['count']}")
```

### Delete Collection
```python
client.delete_collection("old_collection")
```

---

## ⚙️ Integration with AutoPipeline

Replace your FAISS retriever:

```python
# OLD
from src.rag.retriever import LegalRetriever
self.retriever = LegalRetriever()

# NEW
from db.chroma import VectorRetriever
self.retriever = VectorRetriever("legal_docs")

# Usage stays similar
results = self.retriever.query(query, top_k=5)
```

---

## 📁 Storage Location

Database stored at:
```
db_store/chroma/
├── chroma.sqlite3
└── (embedding data)
```

**Recommendation:** Add to `.gitignore`

---

## ✅ Production Checklist

Before deploying:
- [ ] Test with `python app_test.py`
- [ ] Ingest all your documents
- [ ] Verify retrieval quality
- [ ] Set up proper logging
- [ ] Add `db_store/` to .gitignore
- [ ] Document your collection names

---

## 🐛 Troubleshooting

### ImportError: No module named 'chromadb'
```bash
pip install chromadb
```

### ImportError: No module named 'sentence_transformers'
```bash
pip install sentence-transformers
```

### PDF extraction fails
```bash
pip install pypdf
# or
pip install pdfplumber
```

### Empty results
```python
# Check collection has data
from db.chroma import ChromaDBClient
client = ChromaDBClient()
info = client.get_collection_info("legal_docs")
print(info)  # Should show count > 0
```

---

## 📚 Next Steps

1. **Run the test:** `python app_test.py`
2. **Ingest your docs:** Use `ingest_file()` or `ingest_directory()`
3. **Test retrieval:** Query and verify results
4. **Integrate:** Replace FAISS in AutoPipeline
5. **Deploy:** Your vector DB is production-ready!

---

**System Status:** ✅ READY TO USE

All 9 modules created, tested, and documented.
