# MARK System Architecture

## Overview

MARK (Modular Architecture for Reinforced Knowledge) is a complete legal AI system integrating multiple architectures for document processing, retrieval, and generation.

## System Components

### 1. Core Models

#### Mamba Model
- **Architecture**: Hierarchical attention transformer
- **Purpose**: Long document processing with efficient memory usage
- **Features**:
  - Document chunking with overlap
  - Hierarchical attention mechanism
  - Support for classification and generation tasks
  - Custom tokenizer with legal vocabulary

#### Transformer Model
- **Architecture**: BERT-based transfer learning
- **Purpose**: Legal text classification and NER
- **Features**:
  - Fine-tuned on legal corpora
  - Legal entity extraction
  - Case citation normalization
  - Multi-task capabilities

#### RAG System
- **Architecture**: Retrieval-Augmented Generation
- **Purpose**: Document search and context retrieval
- **Components**:
  - FAISS vector store
  - Sentence-transformers embeddings
  - Metadata-aware retrieval
  - Reranking capabilities

#### RL System
- **Architecture**: PPO-based reinforcement learning
- **Purpose**: Policy optimization for generation tasks
- **Features**:
  - Custom legal task environment
  - Reward shaping for quality
  - RLHF integration
  - Continuous improvement loop

### 2. Integration Layer

#### Model Registry
- Central management of all models
- Automatic checkpoint loading
- Device management (CPU/GPU)
- Version control and metadata

#### Fusion Pipeline
- Combines RAG + Generation
- Document retrieval
- Context-aware generation
- Confidence scoring
- Multi-model support

### 3. Training Infrastructure

#### Training Orchestrator
- Unified CLI for all training tasks
- Sequential and parallel training
- Checkpoint management
- Logging and monitoring
- Full pipeline execution

#### RLHF Pipeline
- Supervised fine-tuning (SFT)
- Reward model training
- PPO/DPO optimization
- Evaluation and export

### 4. API Layer

#### FastAPI Server
- RESTful API endpoints
- Real-time inference
- Batch processing support
- Authentication ready
- CORS enabled

**Endpoints**:
- `GET /health` - Health check
- `POST /query` - RAG + Generation
- `POST /rag-search` - Document search
- `POST /generate` - Text generation
- `GET /models` - List models

### 5. External Integrations

#### LangChain Graph
- Document loaders (PDF, TXT)
- Conversational memory
- Tool integration
- MARK models as LLM backend
- Custom retrievers

## Data Flow

```
User Query
    ↓
[API Server]
    ↓
[Fusion Pipeline]
    ↓
[RAG Retrieval] → [Document Store]
    ↓
[Context Assembly]
    ↓
[Generator Model] → [Mamba/Transformer]
    ↓
[Response Generation]
    ↓
[Confidence Scoring]
    ↓
User Response
```

## Training Flow

```
Data Preparation
    ↓
[Mamba Training] → Checkpoint
    ↓
[Transformer Training] → Checkpoint
    ↓
[RAG Index Building] → FAISS Index
    ↓
[RL Training] → Policy
    ↓
[RLHF Pipeline] → Final Model
    ↓
Model Export
```

## Technology Stack

- **Deep Learning**: PyTorch, Transformers
- **Vector Search**: FAISS, Sentence-Transformers
- **RL**: Gymnasium, Custom PPO
- **API**: FastAPI, Pydantic
- **Integration**: LangChain
- **Data**: Datasets, NumPy, Pandas
- **Testing**: Pytest
- **Logging**: Python logging, WandB (optional)

## Deployment Architecture

### Development
```
[Local Machine]
  ├─ Training Scripts
  ├─ Model Development
  ├─ Testing
  └─ Experimentation
```

### Production
```
[API Server]
  ├─ Model Registry
  ├─ Fusion Pipeline
  ├─ Request Handling
  └─ Response Generation

[Vector Store]
  ├─ FAISS Index
  ├─ Document Metadata
  └─ Embeddings Cache

[Model Storage]
  ├─ Checkpoints
  ├─ Vocabularies
  └─ Configurations
```

## Scalability

- **Horizontal**: Multiple API instances
- **Vertical**: GPU acceleration
- **Caching**: Model and embedding caches
- **Async**: FastAPI async support
- **Batch**: Batch inference for efficiency

## Security Considerations

- Input validation
- Rate limiting (to be implemented)
- Authentication (to be implemented)
- Model access control
- Data privacy compliance

## Performance Metrics

- **Latency**: < 200ms for retrieval, < 1s for generation
- **Throughput**: 10-100 requests/second (hardware dependent)
- **Accuracy**: Task-specific (classification, generation, retrieval)
- **Resource**: ~2GB RAM for inference, 4GB+ for training

## Future Enhancements

1. Multi-modal support (PDF, images)
2. Advanced caching strategies
3. Distributed training
4. Model quantization
5. Continuous learning pipeline
6. Advanced monitoring and alerting
7. A/B testing framework
8. User feedback integration
