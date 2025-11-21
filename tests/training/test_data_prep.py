"""
Unit tests for data preparation module.
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.training.data_prep import SFTDataGenerator


def test_qa_pair_generation():
    """Test QA pair generation from sample text."""
    generator = SFTDataGenerator("chroma_db", "pdf_docs")
    
    sample_chunk = "The Supreme Court ruled that Section 302 IPC applies to murder cases."
    metadata = {"source": "test.pdf", "page": 1}
    
    qa_pairs = generator.generate_qa_pair(sample_chunk, metadata)
    
    assert len(qa_pairs) > 0
    assert all('instruction' in qa for qa in qa_pairs)
    assert all('input' in qa for qa in qa_pairs)
    assert all('output' in qa for qa in qa_pairs)


def test_jsonl_format():
    """Test JSONL output format."""
    generator = SFTDataGenerator("chroma_db", "pdf_docs")
    
    # Create sample data
    data = [
        {
            "instruction": "Answer the question",
            "input": "Context: test\n\nQuestion: What?",
            "output": "Test answer",
            "metadata": {}
        }
    ]
    
    # Save and read back
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        generator.save_jsonl(data, temp_path)
        
        # Verify format
        with open(temp_path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 1
        loaded = json.loads(lines[0])
        assert 'instruction' in loaded
        assert 'input' in loaded
        assert 'output' in loaded
        assert 'metadata' not in loaded  # Should be stripped
    
    finally:
        Path(temp_path).unlink()


def test_noun_phrase_extraction():
    """Test topic extraction."""
    generator = SFTDataGenerator("chroma_db", "pdf_docs")
    
    text = "The Supreme Court in Delhi ruled on Article 370 and fundamental rights."
    topics = generator.extract_noun_phrases(text)
    
    assert len(topics) > 0
    assert any('Supreme' in t or 'Court' in t or 'Delhi' in t for t in topics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
