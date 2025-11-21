# Testing Guide - Legal AI System

## 📋 Test Coverage Overview

### Available Test Suites

1. **test_mamba.py** - Mamba Architecture Tests (15+ tests)
2. **test_transfer.py** - Transfer Learning Tests (12+ tests)
3. **test_rag.py** - RAG System Tests (10+ tests)
4. **test_rl.py** - Reinforcement Learning Tests (8+ tests)

**Total**: 45+ comprehensive test cases

---

## 🚀 Running Tests

### Run All Tests

```bash
# Simple run
pytest tests/ -v

# With detailed output
pytest tests/ -v -s

# With coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Using Make
make test
make test-cov
```

### Run Specific Component Tests

```bash
# Mamba tests only
pytest tests/test_mamba.py -v

# Transfer tests only
pytest tests/test_transfer.py -v

# RAG tests only
pytest tests/test_rag.py -v

# RL tests only
pytest tests/test_rl.py -v
```

### Run Specific Test Cases

```bash
# Run single test
pytest tests/test_mamba.py::TestMambaModel::test_model_initialization -v

# Run test class
pytest tests/test_mamba.py::TestMambaModel -v

# Run tests matching pattern
pytest tests/ -k "tokenizer" -v
```

---

## 📊 Test Details

### 1. Mamba Architecture Tests

**File**: `tests/test_mamba.py`

**Test Classes**:
- `TestDocumentTokenizer` - Tokenization and chunking
- `TestHierarchicalAttention` - Attention mechanisms
- `TestMambaModel` - Model forward pass and generation
- `TestPositionalEncoding` - Positional encoding

**Key Tests**:
- ✅ Tokenizer initialization and vocab building
- ✅ Document chunking with overlap
- ✅ Hierarchical attention (3 levels)
- ✅ Classification forward pass
- ✅ Generation forward pass
- ✅ Text generation
- ✅ Parameter counting
- ✅ End-to-end pipeline

**Run**:
```bash
pytest tests/test_mamba.py -v
```

---

### 2. Transfer Learning Tests

**File**: `tests/test_transfer.py`

**Test Classes**:
- `TestLegalTokenizer` - Legal-specific tokenization
- `TestLegalTransferModel` - Transfer model variants

**Key Tests**:
- ✅ Legal entity extraction
- ✅ Text preprocessing
- ✅ Classification model
- ✅ NER model
- ✅ QA model
- ✅ Model freezing
- ✅ Forward passes for all tasks
- ✅ End-to-end pipeline

**Run**:
```bash
pytest tests/test_transfer.py -v
```

---

### 3. RAG System Tests

**File**: `tests/test_rag.py`

**Test Classes**:
- `TestDocument` - Document container
- `TestFAISSStore` - FAISS document store
- `TestLegalRetriever` - Retrieval mechanisms
- `TestRAGGenerator` - Generation (requires GPU)
- `TestRAGPipeline` - End-to-end pipeline

**Key Tests**:
- ✅ Document creation and management
- ✅ FAISS store operations
- ✅ Document search and retrieval
- ✅ Metadata filtering
- ✅ Query expansion
- ✅ Pipeline integration
- ✅ Full RAG workflow

**Run**:
```bash
pytest tests/test_rag.py -v

# Skip GPU-dependent tests
pytest tests/test_rag.py -v -m "not gpu"
```

---

### 4. Reinforcement Learning Tests

**File**: `tests/test_rl.py`

**Test Classes**:
- `TestLegalTaskEnvironment` - RL environment
- `TestRewardCalculator` - Reward computation
- `TestCustomPolicyNetwork` - Policy networks
- `TestRLAgents` - PPO/DQN agents (requires GPU)

**Key Tests**:
- ✅ Environment initialization and reset
- ✅ Environment step function
- ✅ Reward calculation (summarization, QA, classification)
- ✅ ROUGE and F1 computation
- ✅ Policy network forward pass
- ✅ Action selection
- ✅ Agent creation

**Run**:
```bash
pytest tests/test_rl.py -v

# Skip GPU-dependent tests
pytest tests/test_rl.py -v -k "not agent"
```

---

## 🎯 Test Coverage

### Current Coverage

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# Open report
open htmlcov/index.html  # macOS
```

### Expected Coverage

- **Core Modules**: 80-90%
- **Utilities**: 90-95%
- **Training Scripts**: 60-70%
- **Overall**: 75-85%

---

## 🔧 Test Configuration

### pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers
markers =
    slow: marks tests as slow
    gpu: marks tests requiring GPU
    integration: marks integration tests
```

### Skipping Tests

```bash
# Skip slow tests
pytest tests/ -v -m "not slow"

# Skip GPU tests
pytest tests/ -v -m "not gpu"

# Skip integration tests
pytest tests/ -v -m "not integration"
```

---

## 🐛 Debugging Failed Tests

### Verbose Output

```bash
# Show print statements
pytest tests/ -v -s

# Show local variables on failure
pytest tests/ -v -l

# Stop on first failure
pytest tests/ -v -x

# Run last failed tests
pytest tests/ --lf
```

### Debugging Specific Test

```python
# Add breakpoint in test
def test_something():
    import pdb; pdb.set_trace()  # Breakpoint
    # ... test code
```

Run with:
```bash
pytest tests/test_file.py::test_something -v -s
```

---

## 📝 Writing New Tests

### Test Template

```python
import pytest
import torch

class TestNewComponent:
    """Test suite for new component"""
    
    def test_initialization(self):
        """Test component initialization"""
        component = NewComponent()
        assert component is not None
    
    def test_forward_pass(self):
        """Test forward pass"""
        component = NewComponent()
        input_data = torch.randn(2, 10)
        output = component(input_data)
        assert output.shape == (2, 10)
    
    @pytest.fixture
    def sample_data(self):
        """Fixture for sample data"""
        return torch.randn(4, 20)
    
    def test_with_fixture(self, sample_data):
        """Test using fixture"""
        assert sample_data.shape == (4, 20)
```

---

## 🚦 Continuous Integration

### GitHub Actions (Example)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest tests/ --cov=src
```

---

## 📊 Performance Tests

### Benchmarking

```python
# Using pytest-benchmark
def test_model_speed(benchmark):
    model = MambaModel(vocab_size=1000, d_model=256)
    input_ids = torch.randint(0, 1000, (1, 100))
    
    result = benchmark(model, input_ids=input_ids)
```

Run:
```bash
pytest tests/ --benchmark-only
```

---

## ✅ Test Checklist

Before committing code:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Coverage >= 75%: `pytest tests/ --cov=src`
- [ ] No warnings: `pytest tests/ -v --strict-warnings`
- [ ] Code formatted: `black src/ tests/`
- [ ] Linting passes: `flake8 src/ tests/`
- [ ] Documentation updated
- [ ] New tests added for new features

---

## 🔍 Common Issues

### Issue: Import Errors

**Solution**:
```bash
# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: CUDA Out of Memory

**Solution**: Skip GPU tests or use CPU
```bash
pytest tests/ -v -m "not gpu" --cpu
```

### Issue: Missing Dependencies

**Solution**:
```bash
pip install -r requirements.txt --upgrade
python -m spacy download en_core_web_sm
```

---

## 📈 Test Metrics

Track test metrics over time:

- **Test Count**: Number of test cases
- **Coverage**: Code coverage percentage
- **Duration**: Test execution time
- **Failure Rate**: Percentage of failing tests
- **Flakiness**: Tests that intermittently fail

---

## 🎓 Best Practices

1. **Write tests first** (TDD approach)
2. **Test edge cases** and error conditions
3. **Use fixtures** for common setup
4. **Mock external dependencies** (APIs, databases)
5. **Keep tests fast** (< 1s per test)
6. **Make tests independent** (no shared state)
7. **Use descriptive names** for test functions
8. **Add docstrings** to explain test purpose

---

## 📚 Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Coverage.py](https://coverage.readthedocs.io/)

---

**Last Updated**: January 2024
**Test Count**: 45+ tests
**Coverage Target**: 75-85%
