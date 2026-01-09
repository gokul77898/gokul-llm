# 🏛️ NyayaMitra — Indian Legal AI (Encoder–Decoder Architecture)

NyayaMitra is a **production-grade Indian legal assistant** built using a **distributed encoder–decoder architecture** with Hugging Face Inference Endpoints and Spaces.

The system is designed for **accuracy, scalability, and cost efficiency**, following the same architectural principles used in real-world AI systems.

---

## 📐 High-Level Architecture

```
Frontend UI
   │
   ▼
Backend API (/query)
   │
   ├──► Encoder Service (HF Space — 8B)
   │        │
   │        └─► Embeddings (1024-dim)
   │
   └──► Decoder Service (HF Inference Endpoint — 32B + LoRA)
            │
            └─► Final Legal Answer
```

---

## 🧠 Core Design Principles

* **Separation of concerns**
* **Heavy generation on GPU (A100)**
* **Light semantic understanding on smaller GPU**
* **No model logic in frontend**
* **Stateless, scalable inference**

---

## 🔹 Components Breakdown

### 1️⃣ Frontend (UI)

* Collects user legal queries
* Sends requests to backend only
* Never talks directly to models
* Keeps UI lightweight and secure

**Example request:**

```json
{
  "query": "What are bail provisions under Section 436 CrPC?",
  "model": "auto",
  "top_k": 5
}
```

---

### 2️⃣ Backend API (Core Orchestrator)

The backend is the **brain of the system**.

Responsibilities:

* Input validation
* Encoder invocation (optional)
* Decoder invocation (mandatory)
* Response normalization
* Confidence estimation

The backend exposes:

```
POST /query
```

---

### 3️⃣ Encoder Service (HF Space — 8B)

**Purpose:**
Semantic understanding & representation.

**Why separate?**

* Encoder does **not need A100**
* Runs on smaller GPU (T4 / medium)
* Much cheaper
* Easily replaceable

**What it does:**

* Converts query → 1024-dim embedding
* Enables:
  * Semantic grounding
  * Future RAG expansion
  * Confidence boosting

**Endpoint:**

```
POST https://omilosaisolutions-indian-legal-encoder-8b.hf.space/encode
```

**Response:**

```json
{
  "embedding": [ ...1024 floats... ]
}
```

⚠️ If no documents exist, embeddings are still generated but **do not block inference**.

---

### 4️⃣ Decoder Service (HF Inference Endpoint — 32B + LoRA)

**Purpose:**
Authoritative legal answer generation.

**Model Stack:**

* Base: `Qwen/Qwen2.5-32B-Instruct` 
* Adapter: `nyayamitra` (LoRA)
* Engine: **vLLM**
* Hardware: **A100 80GB**

**Why HF Inference Endpoint?**

* Autoscaling
* Token-based auth
* Production reliability
* OpenAI-compatible API

---

## 🔥 Decoder Inference (Exact API Used)

The decoder uses **OpenAI-style chat completions**.

### Endpoint

```
POST /v1/chat/completions
```

### Request Payload

```json
{
  "model": "Qwen/Qwen2.5-32B-Instruct",
  "lora": "nyayamitra",
  "messages": [
    {
      "role": "system",
      "content": "You are an Indian legal assistant. Answer strictly according to Indian law."
    },
    {
      "role": "user",
      "content": "<user query>"
    }
  ],
  "temperature": 0.3,
  "max_tokens": 300
}
```

### Response Parsing

```json
response["choices"][0]["message"]["content"]
```

---

## 🔐 Authentication & Security

* **Frontend:** No tokens
* **Backend:** Uses HF token via environment variables
* **Encoder Space:** Public
* **Decoder Endpoint:** Private / authenticated

### Required Environment Variables (Backend Only)

```env
HF_ENDPOINT_URL=https://xxxxx.us-east-1.aws.endpoints.huggingface.cloud
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

---

## 🧪 End-to-End Verification

A dedicated test confirms encoder + decoder integration:

```bash
python test_e2e.py
```

Expected output:

```
Embedding: OK (1024 dim)
Decoder: OK (LoRA active)
Answer returned
```

---

## ⚖️ Legal Accuracy & Guarding

* LoRA currently trained with **~200 steps**
* Infrastructure is correct
* Legal hallucinations can still occur

### Mitigation Strategy

* Strong system prompt grounding
* Planned CrPC section disambiguation
* Future LoRA expansion with section-wise datasets

---

## 🚀 Why This Architecture Works

| Problem              | Solution                |
| -------------------- | ----------------------- |
| GPU memory explosion | Split encoder & decoder |
| High inference cost  | Small GPU encoder       |
| Vendor lock-in       | Standard APIs           |
| Scaling              | Independent services    |
| Debugging            | Clear boundaries        |

This is the **same pattern used by OpenAI, Anthropic, and large RAG systems**.

---

## 📌 Status

| Component      | Status     |
| -------------- | ---------- |
| Frontend       | ✅ Working  |
| Backend        | ✅ Working  |
| Encoder Space  | ✅ Working  |
| HF Endpoint    | ✅ Working  |
| LoRA Adapter   | ✅ Loaded   |
| UI Integration | ✅ Complete |

---

## 🧠 Final Note

NyayaMitra is **production-correct** at the infrastructure level.

Future improvements focus on:

* Legal accuracy
* Dataset quality
* Guardrails
* RAG document ingestion
