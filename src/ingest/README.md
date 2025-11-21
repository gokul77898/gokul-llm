# 🏛️ Indian Supreme Court Judgments Ingestion Pipeline

Complete data ingestion pipeline for AWS Indian Supreme Court judgments dataset into ChromaDB vector database.

## 📋 Overview

This pipeline downloads, processes, chunks, and ingests Indian Supreme Court judgment data from AWS Open Data Registry into ChromaDB for semantic search and retrieval.

**Dataset:** [AWS Open Data - Indian Supreme Court Judgments](https://registry.opendata.aws/indian-supreme-court-judgments)

---

## 🗂️ Module Structure

```
src/ingest/
├── __init__.py           # Package initialization
├── download.py           # Download parquet files from AWS S3
├── parquet_loader.py     # Load and process parquet files
├── chunker.py            # Chunk long judgment texts
├── chroma_ingest.py      # Main ingestion pipeline
├── test_retrieval.py     # Test retrieval functionality
└── README.md             # This file
```

---

## 🚀 Quick Start

### 1. Download Parquet Files

```bash
python3 src/ingest/download.py --years 2018 2019 2020
```

**Options:**
- `--years`: Space-separated list of years to download (default: 2018 2019 2020)
- `--output`: Output directory (default: data/parquet)

**Output:** Parquet files in `data/parquet/`

---

### 2. Ingest into ChromaDB

```bash
python3 src/ingest/chroma_ingest.py
```

**Options:**
- `--data-dir`: Directory containing parquet files (default: data/parquet)
- `--chunk-size`: Chunk size in characters (default: 1500)
- `--overlap`: Overlap between chunks (default: 200)
- `--batch-size`: Batch size for ingestion (default: 100)
- `--reset`: Delete existing collection and start fresh

**Output:** ChromaDB collection at `db_store/chroma/supreme_court_judgments`

---

### 3. Test Retrieval

```bash
python3 src/ingest/test_retrieval.py --query "murder case IPC 302"
```

**Options:**
- `--query`: Query text (default: "murder case IPC 302")
- `--top-k`: Number of results to retrieve (default: 3)
- `--test-mode`: Run multiple test queries

---

## 📊 Pipeline Details

### Step 1: Download (`download.py`)

**Purpose:** Download parquet files from AWS S3

**Process:**
1. Constructs URLs from base: `https://indian-supreme-court.s3.amazonaws.com/parquet/{year}.parquet`
2. Downloads files with progress tracking
3. Skips existing files
4. Saves to `data/parquet/`

**Example:**
```python
from src.ingest import download_parquet

# Download single year
download_parquet(2018, output_dir="data/parquet")

# Download multiple years
from src.ingest import download_multiple_years
download_multiple_years([2018, 2019, 2020])
```

---

### Step 2: Load Parquet (`parquet_loader.py`)

**Purpose:** Load parquet files into pandas DataFrames

**Process:**
1. Loads parquet files using pandas
2. Extracts required fields:
   - `judgment_text` (critical)
   - `case_number`
   - `judges`
   - `date_of_judgment`
3. Drops rows with empty judgment_text
4. Returns clean DataFrame

**Example:**
```python
from src.ingest import load_all_parquets

# Load all parquet files from directory
df = load_all_parquets("data/parquet")
print(f"Loaded {len(df)} judgments")
```

---

### Step 3: Chunk Texts (`chunker.py`)

**Purpose:** Split long judgment texts into overlapping chunks

**Parameters:**
- `chunk_size`: 1500 characters (default)
- `overlap`: 200 characters (default)

**Process:**
1. Splits text at sentence boundaries when possible
2. Creates overlapping chunks for context continuity
3. Generates unique UUID for each chunk
4. Preserves metadata (case_number, judges, date)

**Example:**
```python
from src.ingest import chunk_text

text = "Long judgment text..."
chunks = chunk_text(text, chunk_size=1500, overlap=200)
print(f"Created {len(chunks)} chunks")
```

---

### Step 4: Ingest to ChromaDB (`chroma_ingest.py`)

**Purpose:** Store chunks in ChromaDB vector database

**Configuration:**
- **Collection:** `supreme_court_judgments`
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Database Path:** `db_store/chroma`

**Process:**
1. Initializes ChromaDB persistent client
2. Creates/gets collection with embedding function
3. Ingests chunks in batches (100 per batch)
4. Each chunk includes:
   - ID (UUID)
   - Document text
   - Metadata (case_number, judges, date, chunk_index)

**Metadata Schema:**
```python
{
    'case_number': str,      # Case identification number
    'judges': str,           # Judge names
    'date': str,             # Date of judgment
    'chunk_index': int,      # Chunk number (0-indexed)
    'total_chunks': int,     # Total chunks for this judgment
    'source_row': int        # Original DataFrame row index
}
```

---

### Step 5: Test Retrieval (`test_retrieval.py`)

**Purpose:** Query ChromaDB and verify retrieval

**Query Types:**
- Single query with custom text
- Multiple test queries
- Semantic similarity search

**Example Output:**
```
🔍 Querying: 'murder case IPC 302'
   Retrieving top 3 matches...

📄 Result 1:
   ID: 123e4567-e89b-12d3-a456-426614174000
   Distance: 0.3521
   
   📋 Metadata:
      Case Number: Criminal Appeal No. 123/2018
      Judges: Justice A.K. Sikri, Justice S. Abdul Nazeer
      Date: 2018-05-15
      Chunk: 2/5
   
   📝 Text Preview:
      The appellant was convicted under Section 302 IPC for murder...
```

---

## 🔧 API Usage

### As Python Modules

```python
# Import functions
from src.ingest import (
    download_parquet,
    load_all_parquets,
    chunk_judgment_dataframe
)

# 1. Download data
download_parquet(2018)

# 2. Load parquet
df = load_all_parquets("data/parquet")

# 3. Chunk judgments
chunks = chunk_judgment_dataframe(df, chunk_size=1500, overlap=200)

# 4. Ingest (run chroma_ingest.py as script)
```

---

## 📈 Expected Performance

### Sample Dataset (3 years: 2018-2020)

| Metric | Approximate Value |
|--------|-------------------|
| Total Judgments | 5,000 - 10,000 |
| Avg Judgment Length | 10,000 - 50,000 chars |
| Total Chunks | 50,000 - 200,000 |
| Ingestion Time | 10 - 30 minutes |
| Database Size | 500MB - 2GB |

---

## ⚙️ Configuration

### Chunk Size Tuning

**Recommended:**
- **chunk_size**: 1500 chars (good balance)
- **overlap**: 200 chars (maintains context)

**Trade-offs:**
- Larger chunks: Better context, fewer chunks, slower search
- Smaller chunks: More precise, more chunks, faster search

### Batch Size Tuning

**Recommended:** 100 chunks per batch

**Trade-offs:**
- Larger batches: Faster ingestion, more memory
- Smaller batches: Slower ingestion, less memory

---

## 🔍 Retrieval Integration

### Use with MARK AutoPipeline

The ingested data is automatically available to the MARK system:

```python
from src.pipelines.auto_pipeline import AutoPipeline

# AutoPipeline will use supreme_court_judgments collection
pipeline = AutoPipeline()
result = pipeline.process_query("What is the punishment for murder under IPC 302?")
```

### Direct ChromaDB Access

```python
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

client = chromadb.PersistentClient(path="db_store/chroma")
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
collection = client.get_collection(
    name="supreme_court_judgments",
    embedding_function=embedding_fn
)

# Query
results = collection.query(
    query_texts=["murder case"],
    n_results=5
)
```

---

## 🐛 Troubleshooting

### Issue: "No parquet files found"

**Solution:** Download files first
```bash
python3 src/ingest/download.py --years 2018 2019 2020
```

### Issue: "Collection not found"

**Solution:** Run ingestion first
```bash
python3 src/ingest/chroma_ingest.py
```

### Issue: "Memory error during ingestion"

**Solution:** Reduce batch size
```bash
python3 src/ingest/chroma_ingest.py --batch-size 50
```

### Issue: "Download timeout"

**Solution:** Check internet connection or retry with fewer years
```bash
python3 src/ingest/download.py --years 2018
```

---

## 📦 Dependencies

Required packages (already in requirements.txt):
- `pandas` - DataFrame processing
- `pyarrow` - Parquet file handling
- `requests` - File downloads
- `chromadb` - Vector database
- `sentence-transformers` - Embeddings

---

## 🔐 Data Privacy & Compliance

**Dataset:** Public domain Indian Supreme Court judgments from AWS Open Data Registry

**License:** Open Data (check AWS registry for specific license)

**Usage:** Legal research, AI training, educational purposes

---

## 🎯 Next Steps

After successful ingestion:

1. **Update Collection Name** in `src/core/chroma_manager.py` if needed
2. **Test with AutoPipeline** to verify integration
3. **Fine-tune chunk parameters** based on retrieval quality
4. **Add more years** of data as needed

---

## 📞 Support

For issues or questions:
- Check logs in ingestion output
- Review error messages carefully
- Verify all dependencies installed
- Ensure sufficient disk space (2-3GB recommended)

---

**Status:** ✅ Ready for Production Use  
**Last Updated:** November 2025  
**Version:** 1.0.0
