# 🚀 Quick Start Guide - MARK AI System

## ✅ What Was Built

A **complete end-to-end AI system** with:
- ✅ ChromaDB vector database (FAISS removed)
- ✅ Automatic model selection
- ✅ AutoPipeline with retrieval + generation
- ✅ Training architecture (SFT/RL/RLHF skeletons)
- ✅ Admin dashboard with monitoring
- ✅ Safety guards (no training, no data loading)

---

## 🏁 Quick Start (3 Steps)

### Step 1: Verify Installation

```bash
# Test all components
python3.10 test_system_integration.py
```

**Expected:** All 5 tests pass ✅

### Step 2: Start the API

```bash
# Start FastAPI server
python3.10 -m uvicorn src.api.main:app --reload
```

**API running at:** `http://localhost:8000`

### Step 3: Test the System

```bash
# Health check
curl http://localhost:8000/api/v1/system/health

# Query (no docs yet, but works)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is appropriate government?","top_k":5}'

# ChromaDB stats
curl http://localhost:8000/api/v1/chroma/stats
```

---

## 📊 Available Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/system/health` | GET | System status |
| `/api/v1/chroma/stats` | GET | ChromaDB statistics |
| `/api/v1/training/status` | GET | Training pipeline status |
| `/api/v1/model_selector/log` | GET | Model selection history |
| `/query` | POST | Main query endpoint |

---

## 🎯 What Works Now

✅ **ChromaDB Integration** - Empty collection ready  
✅ **Model Selector** - Intelligent query routing  
✅ **AutoPipeline** - End-to-end processing  
✅ **Health Monitoring** - Real-time status  
✅ **Admin Dashboard** - UI for monitoring  

---

## ⚠️ What's Blocked

❌ **Data Ingestion** - `SETUP_MODE=true`  
❌ **Training** - All raise errors  

---

## 🔓 To Enable Features

### Enable Data Loading:
```bash
export SETUP_MODE=false
python3.10 -c "from db.chroma import ingest_file; ingest_file('doc.pdf', 'legal_docs')"
```

### Test Query After Loading:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Your question here","top_k":5}'
```

---

## 📁 Key Files

```
✅ src/core/model_selector.py       - Auto model selection
✅ src/core/chroma_manager.py       - ChromaDB manager
✅ src/pipelines/auto_pipeline.py   - Main pipeline
✅ src/training/*_trainer.py        - Training skeletons
✅ src/api/v1_endpoints.py          - New API routes
✅ ui/src/components/AdminDashboard.jsx - Admin UI
```

---

## 🧪 Tests

```bash
# Integration test (all components)
python3.10 test_system_integration.py

# Ingestion blocker test
python3.10 test_ingestion_block.py

# ChromaDB verification
python3.10 test_chroma_mock.py
```

---

## 📖 Full Documentation

See `SYSTEM_COMPLETE_SUMMARY.md` for detailed architecture and implementation details.

---

## ✅ You're Ready!

The system is **fully integrated and tested**. When you're ready to go to production:

1. Set `SETUP_MODE=false`
2. Ingest your documents
3. Remove training blockers (if needed)
4. Deploy!

**Status:** 🎉 **ALL SYSTEMS GO**
