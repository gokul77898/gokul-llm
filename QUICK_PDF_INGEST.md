# Quick PDF Ingestion & Query Guide

## Overview

This script loads your PDF, indexes it into FAISS, and queries using the RLHF-trained model.

## Prerequisites

```bash
# Install required packages
pip install langchain langchain-community pypdf sentence-transformers faiss-cpu
```

## Your PDF Location

```
~/Documents/test documents.pdf
```

Make sure this file exists!

## Usage

### Method 1: Run with Default Query

```bash
cd /Users/gokul/Documents/MARK
python3.10 ingest_and_query.py
```

Default query: "What is this document about?"

### Method 2: Run with Custom Query

```bash
python3.10 ingest_and_query.py "What are the main legal points?"
```

Or:

```bash
python3.10 ingest_and_query.py Summarize the key findings
```

### Method 3: Make Executable and Run

```bash
chmod +x ingest_and_query.py
./ingest_and_query.py "Your custom question here"
```

## What the Script Does

### Step 1: Load PDF
- Reads `~/Documents/test documents.pdf`
- Loads all pages using LangChain's PyPDFLoader
- Shows page count and preview

### Step 2: Split Documents
- Breaks pages into 500-character chunks
- 50-character overlap for context preservation
- Optimized for semantic search

### Step 3: Create FAISS Index
- Embeds chunks using `sentence-transformers/all-MiniLM-L6-v2`
- Creates FAISS index (Flat type for accuracy)
- Saves to `checkpoints/rag/custom_faiss.index`

### Step 4: Load RLHF Model
- Loads `rl_trained` (PPO-optimized model)
- Initializes on available device (CPU/GPU)

### Step 5: Create Pipeline
- Sets up LegalRetriever with indexed documents
- Creates FusionPipeline with RLHF generator
- Ready for queries

### Step 6: Query & Results
- Retrieves top-5 relevant chunks
- Generates answer using RLHF model
- Measures latency
- Outputs results

## Expected Output

```
================================================================================
VAKEELS.AI - PDF INGESTION & QUERY SYSTEM
================================================================================

📄 PDF Path: /Users/gokul/Documents/test documents.pdf
💾 Index Path: checkpoints/rag/custom_faiss.index
🤖 Generator Model: rl_trained
🔍 Top-K: 5

STEP 1: Loading PDF...
--------------------------------------------------------------------------------
✅ Loaded 10 pages from PDF
📖 Preview: This is the beginning of the document...

STEP 2: Splitting documents into chunks...
--------------------------------------------------------------------------------
✅ Split into 45 chunks
   Average chunk size: 480 chars

STEP 3: Creating FAISS index...
--------------------------------------------------------------------------------
🔧 Embedding model: sentence-transformers/all-MiniLM-L6-v2
📊 Embedding dimension: 384
🔄 Indexing 45 text chunks...
✅ Index created and saved to checkpoints/rag/custom_faiss.index
   Total documents indexed: 45

STEP 4: Loading RLHF-trained model...
--------------------------------------------------------------------------------
🖥️  Device: cpu
⏳ Loading rl_trained model...
✅ Model loaded successfully

STEP 5: Creating Fusion Pipeline...
--------------------------------------------------------------------------------
✅ Pipeline created with rl_trained generator
   Retriever ready with 45 documents

STEP 6: Querying the system...
================================================================================

❓ Query: What is this document about?

================================================================================
RESULTS
================================================================================

📝 ANSWER:
--------------------------------------------------------------------------------
RLHF generated response (action: 42)

📊 METRICS:
--------------------------------------------------------------------------------
✓ Confidence:       85.0%
✓ Retrieved Docs:   5
✓ Latency:          1250 ms
✓ Model Used:       rl_trained
✓ Top-K Requested:  5

📚 RETRIEVED DOCUMENTS:
--------------------------------------------------------------------------------
1. Score: 0.923
   Preview: This document discusses legal principles...

2. Score: 0.891
   Preview: The key findings include...

3. Score: 0.845
   Preview: According to the statute...

📋 RAW JSON:
--------------------------------------------------------------------------------
{
  "query": "What is this document about?",
  "answer": "RLHF generated response (action: 42)",
  "confidence": 0.85,
  "retrieved_docs_count": 5,
  "latency_ms": 1250,
  "model": "rl_trained",
  "top_k": 5
}

================================================================================
✅ COMPLETE - PDF successfully indexed and queried!
================================================================================
```

## Output Explanation

### Metrics

- **Confidence**: Model's certainty in the answer (0-100%)
- **Retrieved Docs**: Number of relevant chunks found
- **Latency**: Total processing time in milliseconds
- **Model Used**: Which generator model was used
- **Top-K**: How many documents were retrieved

### Retrieved Documents

Shows the top 3 most relevant chunks with:
- Similarity score (higher = more relevant)
- Content preview (first 150 characters)

### Raw JSON

Complete response in JSON format for:
- API integration
- Logging
- Further processing

## Configuration

Edit `ingest_and_query.py` to customize:

```python
PDF_PATH = os.path.expanduser("~/Documents/test documents.pdf")  # Your PDF
INDEX_PATH = "checkpoints/rag/custom_faiss.index"                # Index location
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"       # Embedding model
GENERATOR_MODEL = "rl_trained"                                    # Generator model
TOP_K = 5                                                         # Retrieval count
QUERY = "What is this document about?"                           # Default query
```

## Troubleshooting

### PDF Not Found
```
❌ ERROR: PDF not found at /Users/gokul/Documents/test documents.pdf
```

**Solution**: Verify file exists and path is correct
```bash
ls -la ~/Documents/"test documents.pdf"
```

### Missing Dependencies
```
ModuleNotFoundError: No module named 'pypdf'
```

**Solution**: Install requirements
```bash
pip install langchain langchain-community pypdf sentence-transformers faiss-cpu
```

### Model Not Found
```
❌ ERROR loading model: Model 'rl_trained' not found
```

**Solution**: Ensure RLHF model checkpoint exists
```bash
ls -la checkpoints/rlhf/ppo/ppo_final.pt
```

### Low Confidence
```
✓ Confidence: 15.0%
```

**Solution**: 
- Try different query phrasing
- Increase top-k value
- Verify PDF content is relevant to query

## Advanced Usage

### Query Multiple Times

```bash
# First query
python3.10 ingest_and_query.py "What is the main topic?"

# Second query (reuses index)
python3.10 ingest_and_query.py "List the key points"

# Third query
python3.10 ingest_and_query.py "What are the conclusions?"
```

The index is saved, so subsequent queries are faster!

### Use Different Models

Edit script and change:
```python
GENERATOR_MODEL = "mamba"        # or
GENERATOR_MODEL = "transformer"  # or
GENERATOR_MODEL = "rl_trained"   # (default)
```

### Adjust Retrieval

```python
TOP_K = 10  # Retrieve more documents for more context
```

### Change Chunk Size

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Larger chunks
    chunk_overlap=100,  # More overlap
)
```

## Integration with UI

Once indexed, use the same query in the web UI:

1. Open http://localhost:3000
2. Select "RL Optimized" model
3. Set Top-K to 5
4. Enter your query
5. See results with same confidence/retrieval metrics

## Files Created

```
checkpoints/rag/custom_faiss.index      # FAISS vector index
ingest_and_query.py                     # This script
QUICK_PDF_INGEST.md                     # This guide
```

## Next Steps

1. ✅ Index your PDF (this script)
2. ✅ Query via command line
3. ✅ Test in web UI
4. 🔄 Fine-tune model on your data (optional)
5. 🚀 Deploy to production

---

**Ready to run!** Just ensure `test documents.pdf` exists in `~/Documents/` and execute:

```bash
python3.10 ingest_and_query.py
```

🚀 **Vakeels.AI - Legal Intelligence Made Simple**
