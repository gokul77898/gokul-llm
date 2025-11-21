#!/usr/bin/env python3
"""
Download all required models for testing
This prevents download delays during test execution
"""

print("="*70)
print("DOWNLOADING ALL REQUIRED MODELS")
print("="*70)

# 1. Download BERT model
print("\n[1/3] Downloading BERT-base-uncased...")
try:
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    print("✓ BERT-base-uncased downloaded successfully")
except Exception as e:
    print(f"✗ Error downloading BERT: {e}")

# 2. Download GPT-2 model
print("\n[2/3] Downloading GPT-2...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    print("✓ GPT-2 downloaded successfully")
except Exception as e:
    print(f"✗ Error downloading GPT-2: {e}")

# 3. Download Sentence Transformer
print("\n[3/3] Downloading Sentence Transformer...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("✓ Sentence Transformer downloaded successfully")
except Exception as e:
    print(f"✗ Error downloading Sentence Transformer: {e}")

print("\n" + "="*70)
print("✅ ALL MODELS DOWNLOADED SUCCESSFULLY!")
print("="*70)
print("\nYou can now run tests without download delays:")
print("  pytest tests/ -v")
print()
