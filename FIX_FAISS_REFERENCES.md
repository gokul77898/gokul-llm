# 🔧 Fix FAISS References - Action Plan

## Issue
`src/api/main.py` still has FAISS imports and references that should be replaced with ChromaDB.

---

## Files to Modify

### 1. src/api/main.py

**Current (Lines 22-23, 116, 125-135):**
```python
from src.rag.document_store import FAISSStore
from src.rag.retriever import LegalRetriever

class APIState:
    document_store: Optional[FAISSStore] = None
    
FAISS_INDEX_PATH = "checkpoints/rag/custom_faiss.index"

def load_faiss_index():
    """Load FAISS index if it exists"""
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH}")
            doc_store = FAISSStore(
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                index_type="Flat"
            )
```

**Should be replaced with:**
```python
from src.core.chroma_manager import get_chroma_manager

class APIState:
    chroma_manager: Optional[Any] = None
    
def initialize_chromadb():
    """Initialize ChromaDB on startup"""
    try:
        logger.info("Initializing ChromaDB...")
        chroma_manager = get_chroma_manager()
        if chroma_manager.is_ready():
            logger.info(f"ChromaDB ready (collection: legal_docs)")
            state.chroma_manager = chroma_manager
            state.retriever = chroma_manager.get_retriever()
        else:
            logger.warning("ChromaDB not ready")
    except Exception as e:
        logger.error(f"ChromaDB initialization failed: {e}")
```

---

## Quick Fix Script

Run this to automatically fix the issue:

```bash
# TODO: Create automated patch script
python fix_faiss_to_chromadb.py
```

---

## Manual Fix Steps

1. **Open** `src/api/main.py`

2. **Remove imports** (line 22-23):
   - Remove: `from src.rag.document_store import FAISSStore`
   - Remove: `from src.rag.retriever import LegalRetriever`

3. **Add ChromaDB import**:
   ```python
   from src.core.chroma_manager import get_chroma_manager
   ```

4. **Update APIState** (line ~116):
   - Remove: `document_store: Optional[FAISSStore] = None`
   - Add: `chroma_manager: Optional[Any] = None`

5. **Replace load_faiss_index** (lines 125-150):
   - Delete the entire `load_faiss_index()` function
   - Add the `initialize_chromadb()` function (see above)

6. **Update startup** (wherever load_faiss_index is called):
   - Replace: `load_faiss_index()`
   - With: `initialize_chromadb()`

---

## Testing After Fix

```bash
# 1. Start backend
python3.10 -m uvicorn src.api.main:app --reload

# 2. Test health endpoint
curl http://localhost:8000/api/v1/system/health

# 3. Test query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"test","top_k":5}'

# 4. Verify no FAISS errors in logs
```

---

## Expected Outcome

✅ No FAISS imports  
✅ ChromaDB initialized on startup  
✅ Retriever available from ChromaManager  
✅ No errors in logs  
✅ System audit passes 7/7 tests  

---

## Priority

**Medium** - System works without this fix, but cleaning up reduces technical debt.

You can:
- ✅ Use the system as-is (AutoPipeline uses ChromaDB correctly)
- ⚠️ Fix now to avoid future confusion
- 🔄 Fix later before production deployment
