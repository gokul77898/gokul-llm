# OMILOS Legal AI System - Complete Code Analysis

## Executive Summary

**System:** Legal AI with RAG + MoE + Local Inference  
**Architecture:** Multi-phase pipeline with strict safety guarantees  
**Key Principle:** Decoder never sees raw queries - only validated, assembled evidence

---

## PART 1: CORE SYSTEM ARCHITECTURE

### 1.1 Model Registry (`src/core/model_registry.py` - 93 lines)

**Purpose:** Central registry for all MoE expert models

**Key Components:**
- `ExpertInfo` dataclass (lines 17-27): Stores model metadata
  - name, model_id, task_types, token_window, tuning, lora_config, priority_score, role
- `ModelRegistry` class (lines 30-80): Manages expert models
  - Loads from YAML config
  - Provides query methods: get_expert, list_experts, get_experts_by_task
  - Filters by role (encoder/decoder)
- Global singleton pattern (lines 83-92)

**Design Pattern:** Singleton with lazy initialization

---

### 1.2 Generator (`src/core/generator.py` - 204 lines)

**Purpose:** Unified generation interface with MoE routing

**Main Methods:**

1. **`generate()` (lines 22-159):** Main generation pipeline
   - Model selection: auto (MoE routing) or explicit
   - Model caching: Global LOADED_EXPERTS dict
   - Task-specific handling:
     - Classification (lines 64-75): Softmax over logits
     - NER (lines 77-100): Token-level or sequence classification
     - Generation (lines 102-119): Text generation with temperature
     - Fallback (lines 121-143): Tries generation, falls back to classification
   - Error handling: Never crashes, returns error message

2. **`generate_with_expert()` (lines 161-203):** UI-friendly wrapper
   - Simplified interface for UI
   - Loads model if not cached
   - Cleans prompt echo from output

**Key Features:**
- Global model cache prevents reloading
- Task-specific inference paths
- Graceful error handling

---

### 1.3 ChromaDB Manager (`src/core/chroma_manager.py` - 151 lines)

**Purpose:** Centralized ChromaDB initialization and management

**Architecture:**
- Singleton pattern (lines 27-36)
- Graceful degradation if ChromaDB unavailable (lines 14-24)
- Lazy initialization (lines 38-70)

**Key Methods:**
- `_initialize()`: Creates client, collection, retriever
- `get_collection_stats()`: Returns document count and status
- `list_documents()`: Lists unique sources
- `count_documents()`: Total document count

**Safety:** All methods handle uninitialized state gracefully

---

### 1.4 Model Selector (`src/core/model_selector.py` - 177 lines)

**Purpose:** Automatic model selection based on query complexity

**Selection Logic:**
1. Analyze query (lines 54-97):
   - Count words
   - Detect legal terms (20+ keywords)
   - Detect reasoning requirements
   - Classify complexity: SIMPLE/MODERATE/COMPLEX/LEGAL

2. Pick model (lines 99-155):
   - Simple (≤5 words) → RL trained (fast)
   - Legal terms → RL trained (domain-specific)
   - Complex/long (>12 words) → Transformer
   - Default → RL trained

**Features:**
- Selection logging for analysis
- Explainable decisions with reasons

---

## PART 2: INFERENCE SYSTEM

### 2.1 Inference Server (`src/inference/server.py` - 1501 lines)

**Critical Invariants (lines 1-21):**
1. Decoder NEVER sees raw user query
2. Decoder ONLY sees assembled context from RAG
3. If RAG refuses → decoder NOT called
4. MoE routing decides encoder + decoder
5. All refusals server-side enforced
6. No hallucinated answers possible

**Pipeline Order:**
Encoder → RAG Retrieval → RAG Validation → Context Assembly → Decoder

**Key Components:**

1. **Configuration Validation (lines 23-118):**
   - Validates config at startup (FAIL-FAST)
   - Checks HF_TOKEN (deprecated)
   - Initializes local model registry (lazy loading)

2. **Latency Tracking (lines 157-230):**
   - `LatencyTracker` class
   - MAX_CONCURRENT_REQUESTS = 2
   - TOTAL_REQUEST_BUDGET_MS = 30,000
   - Per-stage timing with budget enforcement

3. **Refusal System (lines 232-293):**
   - 30+ structured refusal reasons
   - Categories: encoder/decoder failures, RAG failures, backpressure, kill switches
   - `_make_refusal()`: Creates structured response with config_hash

4. **Production Logging (lines 295-327):**
   - Append-only JSONL logs
   - Records: request_id, timestamp, query, models, status, latency
   - Truncates query to 500 chars for safety

5. **Encoder Execution (lines 445-480):**
   - `_run_encoder()`: Runs local encoder with error handling
   - Phase P2 fault isolation: EncoderExecutionError
   - NEVER crashes server on encoder failure

6. **RAG Components (lines 539-602):**
   - Lazy-initialized: retriever, validator, assembler
   - Post-generation verifier
   - Replay store and drift logger
   - Token budget: 2500 tokens

7. **Health Endpoints:**
   - `/health`: Basic check with config_hash
   - `/health/gpu`: GPU status, VRAM, models loaded
   - `/health/full`: Comprehensive system state
   - `/ops/state`: Operational state

**Safety Features:**
- Structured refusals (never guess)
- Latency budgets (prevent hangs)
- Concurrency limits (backpressure)
- Kill switches (feature flags)
- Canary mode support

---

### 2.2 MoE Router (`src/inference/moe_router.py` - 160 lines)

**Purpose:** Routes queries to best expert (deterministic, no model loading)

**Routing Algorithm (lines 65-124):**

For each expert, calculate score:
- Base: priority_score × 10.0
- +5.0 if task matches
- +3.0 if doc >2048 words and expert has ≥4096 token window
- +1.0 if doc <512 words and expert has ≤512 token window
- +2.0 if "judgment"/"order" in text and expert does judgment-prediction
- +2.0 if "section"/"act" in text and expert does NER

Sort by score descending, return top_k

**Features:**
- Cache for UI routing (lines 31-63)
- Decision logging to JSON (lines 126-137)
- CLI interface (lines 139-157)

---

### 2.3 Model Loader (`src/inference/model_loader.py` - 42 lines)

**Purpose:** Canonical local-only model loader

**Rules:**
- NO remote inference
- NO HF API
- NO hosted models
- 100% local inference

**API:**
- `load_encoder(name)`: Load encoder locally
- `load_decoder(name)`: Load decoder locally

**Design:** Simple wrapper around LocalModelRegistry

---

## PART 3: RAG SYSTEM

### 3.1 RAG Module (`src/rag/__init__.py` - 114 lines)

**Phase R0:** Document schema, normalization, validation, storage  
**Phase R1:** Legal chunking, section parsing, chunk IDs  
**Phase R2:** BM25 + ChromaDB retrieval, fusion  
**Phase R3:** Validation, thresholds, statute checks  
**Phase R4:** Token budgets, citations, context assembly

**Critical Principles:**
- No LLM called with unvalidated evidence
- Decoder never sees raw documents
- Only assembled evidence blocks passed to decoder

---

### 3.2 Legal Retriever (`src/rag/retrieval/retriever.py` - 323 lines)

**Purpose:** Unified retrieval combining BM25 and dense (ChromaDB)

**Key Classes:**

1. **`RetrievedChunk` (lines 28-52):**
   - Canonical output format
   - Fields: chunk_id, text, act, section, doc_type, citation, court, year, score, source
   - Optional: bm25_score, dense_score (explainability)

2. **`LegalRetriever` (lines 55-322):**
   - Supports 3 methods: BM25, Dense, Fused
   - Lazy initialization of retrievers
   - Weights: BM25=0.4, Dense=0.6

**Main Methods:**
- `initialize()`: Loads BM25 index, indexes into ChromaDB
- `retrieve()`: Main API with method selection
- `_retrieve_fused()`: Fetches top_k×2 from both, fuses with weights
- `retrieve_by_section()`: Retrieves by act and section
- `explain_result()`: Explainable score breakdown

---

### 3.3 Retrieval Validator (`src/rag/validation/validator.py` - 293 lines)

**Purpose:** Validates retrieved chunks before context assembly

**Validation Pipeline (lines 129-237):**

1. **Statute Validation:**
   - Checks query statute matches chunk statute
   - Applies penalty if mismatch
   - Rejects if strict mode and no match

2. **Evidence Filtering:**
   - Section consistency checks
   - Repealed law guards
   - Filters invalid chunks

3. **Threshold Checking:**
   - Checks if enough valid chunks
   - Checks confidence scores
   - Refuses if below threshold

**Output:**
- `ValidationResult` with status (PASS/REFUSE)
- Lists of accepted and rejected chunks
- Refusal reason if refused
- Logs to JSONL

**Refusal Reasons:**
- INSUFFICIENT_EVIDENCE
- STATUTE_MISMATCH
- SECTION_MISMATCH
- REPEALED_LAW
- LOW_CONFIDENCE
- NO_VALID_CHUNKS

---

### 3.4 Context Assembler (`src/rag/context/assembler.py` - 361 lines)

**Purpose:** Assembles validated evidence into bounded, auditable context

**Assembly Pipeline (lines 126-255):**

1. **Check Input:**
   - Refuse if no validated chunks

2. **Order Chunks (lines 257-311):**
   - Priority: exact section match → statute match → court hierarchy → score
   - Court hierarchy: SC > HC > others

3. **Format Evidence:**
   - Format each chunk with citations
   - Drop chunks missing citation metadata

4. **Apply Token Budget:**
   - Allocate tokens to each chunk
   - Drop chunks that exceed budget
   - Refuse if no chunks fit

5. **Reindex and Assemble:**
   - Reindex citations [1], [2], [3]...
   - Assemble final context text

**Output:**
- `ContextResult` with status (ASSEMBLED/REFUSE)
- Context text (if successful)
- Lists of used and dropped chunks
- Token count
- Refusal reason if refused

**Features:**
- Token budget enforcement (default: 2500 tokens)
- Citation formatting
- Evidence completeness checks
- Logs to JSONL

---

## PART 4: INGESTION PIPELINE

### 4.1 Text Chunker (`src/ingest/chunker.py` - 197 lines)

**Purpose:** Chunks legal documents for vector embedding

**Main Functions:**

1. **`chunk_text()` (lines 16-72):**
   - Splits text into overlapping chunks
   - Default: chunk_size=1500, overlap=200
   - Tries to break at sentence boundaries (. ! ?)
   - Returns list of text chunks

2. **`chunk_judgment_dataframe()` (lines 75-132):**
   - Chunks all judgments in DataFrame
   - Creates metadata for each chunk
   - Returns list ready for ChromaDB ingestion
   - Progress indicators every 100 judgments

3. **`get_chunk_stats()` (lines 135-155):**
   - Calculates statistics: total, avg/max/min length, total chars

**Algorithm:**
- Sliding window with overlap
- Sentence-boundary aware
- Ensures progress (no infinite loops)

---

## PART 5: API LAYER

### 5.1 FastAPI Main (`src/api/main.py` - 583 lines)

**Purpose:** Production API server for MARK system

**Key Endpoints:**

1. **`/health` (lines 278-294):**
   - Returns status, version, models loaded
   - ChromaDB status and document count

2. **`/query` (lines 314-425):**
   - Main query endpoint with RAG + generation
   - Pipeline: Retrieval → Auto model selection → Generation
   - Returns formatted ChatGPT-style response
   - Fallback on pipeline errors

3. **`/rag-search` (lines 427-463):**
   - Search without generation
   - Returns relevant documents with scores

4. **`/generate` (lines 465-532):**
   - Direct generation (deprecated)
   - Recommends using /query with RAG

5. **`/feedback` (lines 206-229):**
   - Submit user feedback for SFT training
   - Atomic feedback saving
   - Buffer threshold tracking

6. **`/retrain/trigger` (lines 248-276):**
   - Operator-only endpoint
   - Creates SFT bundle from feedback
   - Returns training command

**Initialization:**
- ChromaDB auto-initialized on startup
- Fusion pipeline with ChromaDB retriever
- Auto pipeline for model selection

**State Management:**
- Global APIState class
- Caches: retriever, pipelines, models, feedback worker

---

## PART 6: SYSTEM DESIGN PATTERNS

### 6.1 Safety Guarantees

1. **No Hallucination:**
   - Decoder only sees validated evidence
   - Citations required in evidence
   - Answer validated against evidence

2. **Structured Refusals:**
   - Machine-readable reasons
   - Human-readable messages
   - Never guess or make up answers

3. **Fault Isolation:**
   - Encoder failures don't crash server
   - Decoder failures don't crash server
   - RAG failures trigger refusals

4. **Auditability:**
   - All requests logged to JSONL
   - Trace IDs for replay
   - Decision provenance tracked

### 6.2 Performance Patterns

1. **Lazy Initialization:**
   - Models loaded on first use
   - RAG components initialized when needed
   - ChromaDB lazy-loaded

2. **Caching:**
   - Model cache (LOADED_EXPERTS)
   - Routing cache (MoE router)
   - API state cache

3. **Backpressure:**
   - Concurrency limits (max 2 concurrent)
   - Latency budgets (30 seconds)
   - Token budgets (2500 tokens)

### 6.3 Observability

1. **Logging:**
   - Production requests (JSONL)
   - RAG+MoE pipeline (JSONL)
   - Validation logs (JSONL)
   - Context assembly logs (JSONL)

2. **Health Checks:**
   - Basic: /health
   - GPU: /health/gpu
   - Full: /health/full
   - Ops: /ops/state

3. **Tracing:**
   - Trace IDs for all requests
   - Replay artifacts stored
   - Drift signal logging

---

## PART 7: KEY ALGORITHMS

### 7.1 MoE Routing Algorithm

```
For each expert:
  score = priority_score × 10.0
  if task_matches: score += 5.0
  if long_doc and large_window: score += 3.0
  if short_doc and small_window: score += 1.0
  if judgment_keywords and qa_task: score += 2.0
  if legal_keywords and ner_task: score += 2.0
Sort by score descending
Return top_k
```

### 7.2 Fusion Retrieval Algorithm

```
1. Fetch top_k×2 from BM25
2. Fetch top_k×2 from Dense
3. For each unique chunk:
   fused_score = (bm25_score × 0.4) + (dense_score × 0.6)
4. Sort by fused_score descending
5. Return top_k
```

### 7.3 Context Assembly Algorithm

```
1. Check if chunks exist → refuse if empty
2. Order chunks:
   - Exact section match first
   - Then statute match
   - Then court hierarchy (SC > HC > others)
   - Then by score
3. Format each chunk with citations
4. Apply token budget:
   - Allocate tokens to each chunk
   - Drop if exceeds budget
5. Reindex citations [1], [2], [3]...
6. Assemble final context text
```

---

## PART 8: CONFIGURATION & DEPLOYMENT

### 8.1 Configuration Files

- `configs/moe_experts.yaml`: Expert model definitions
- `configs/rag_config.yaml`: RAG parameters
- `.env`: Environment variables (HF_TOKEN - deprecated)

### 8.2 Feature Flags

- `is_encoder_enabled()`: Kill switch for encoder
- `is_rag_enabled()`: Kill switch for RAG
- `is_decoder_enabled()`: Kill switch for decoder
- `is_canary_mode()`: Canary deployment mode
- `is_ops_override_allowed()`: Ops override permissions

### 8.3 Deployment Modes

1. **CPU Mode (Current):**
   - RAG retrieval works
   - Encoder/decoder fail gracefully
   - Zero safety regression

2. **GPU Mode (Future):**
   - Load models manually
   - Full MoE + RAG + decoder
   - Same safety guarantees

---

## PART 9: DATA FLOW

### 9.1 Query Processing Flow

```
User Query
    ↓
[MoE Router] → Select Encoder + Decoder
    ↓
[Encoder] → Extract facts (if loaded, else skip)
    ↓
[RAG Retrieval] → BM25 + Dense fusion
    ↓
[RAG Validation] → Statute + Section + Threshold checks
    ↓
[Context Assembly] → Token budget + Citations
    ↓
[Decoder] → Generate answer from evidence
    ↓
[Post-Gen Verification] → Validate citations
    ↓
[Response] → Structured answer or refusal
```

### 9.2 Refusal Points

1. **Encoder:** If model not loaded or execution fails
2. **RAG Retrieval:** If no documents found
3. **RAG Validation:** If evidence fails quality checks
4. **Context Assembly:** If token budget exceeded or missing citations
5. **Decoder:** If model not loaded or execution fails
6. **Post-Gen:** If answer doesn't cite evidence

---

## PART 10: TESTING & VERIFICATION

### 10.1 Test Files

- `test_all.py`: Full system integration tests
- `test_system_integration.py`: System-level tests
- `test_full_system_audit.py`: Comprehensive audit
- `verify_all_features.py`: Feature verification
- `verify_auto_detection.py`: Auto-detection tests

### 10.2 Verification Scripts

- `scripts/full_system_verification.py`: End-to-end verification
- `scripts/gpu_readiness_check.py`: GPU setup verification
- `run_tests.sh`: Test runner with coverage

---

## SUMMARY

**Total Lines Analyzed:** ~5000+ lines across 60+ files

**System Architecture:**
- **Core:** Model registry, generator, ChromaDB manager, model selector
- **Inference:** Server with safety guarantees, MoE router, model loader
- **RAG:** Retrieval (BM25 + Dense), validation, context assembly
- **Ingestion:** Text chunking, PDF processing, ChromaDB indexing
- **API:** FastAPI server with health checks and query endpoints

**Key Principles:**
1. Safety first: Structured refusals, no hallucinations
2. Local inference: No remote APIs, all models local
3. Grounded generation: Decoder only sees validated evidence
4. Observability: Comprehensive logging and tracing
5. Fault isolation: Component failures don't crash system

**Production Ready:**
- Latency budgets and backpressure
- Kill switches and canary mode
- GPU health monitoring
- Comprehensive error handling
- Audit logging

This system implements a production-grade legal AI with strict safety guarantees, explainable decisions, and comprehensive observability.
