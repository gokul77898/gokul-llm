#!/usr/bin/env python3
"""
Minimal E2E test - Embedding Service + HF Inference Endpoint
Run: python test_e2e.py
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

EMBEDDING_SERVICE_URL = "https://omilosaisolutions-indian-legal-encoder-8b.hf.space/encode"
HF_ENDPOINT_URL = os.environ.get("HF_ENDPOINT_URL", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def test_embedding(query: str) -> dict:
    """Test embedding service"""
    print(f"🔍 Testing embedding: {EMBEDDING_SERVICE_URL}")
    try:
        response = requests.post(
            EMBEDDING_SERVICE_URL,
            json={"query": query},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        print(f"✅ Embedding OK - embedding_dim={data.get('embedding_dim', 0)}")
        return {"status": "ok", "data": data}
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return {"status": "error", "error": str(e)}

def test_decoder(query: str) -> dict:
    """Test HF Inference Endpoint with encoder integration"""
    print(f"🤖 Testing decoder: {HF_ENDPOINT_URL}")
    
    if not HF_ENDPOINT_URL:
        print("❌ HF_ENDPOINT_URL not set")
        return {"status": "error", "error": "HF_ENDPOINT_URL not set"}
    
    if not HF_TOKEN:
        print("❌ HF_TOKEN not set")
        return {"status": "error", "error": "HF_TOKEN not set"}
    
    try:
        # Get embedding from encoder
        from src.inference.embedding_service import get_embedding
        embedding = get_embedding(query)
        
        # Create debug context
        encoder_debug_context = f"""
[ENCODER DEBUG]
embedding_length={len(embedding)}
embedding_preview={embedding[:5]}
"""
        
        # Build messages with encoder context
        messages = [
            {"role": "system", "content": "You are an Indian legal assistant."},
            {"role": "system", "content": encoder_debug_context},
            {"role": "user", "content": query}
        ]
        
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "Qwen/Qwen2.5-32B-Instruct",
            "lora": "nyayamitra",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 200
        }
        response = requests.post(
            f"{HF_ENDPOINT_URL}/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract answer from OpenAI format
        answer = result["choices"][0]["message"]["content"]
        
        print(f"✅ Decoder OK (with encoder integration)")
        return {"status": "ok", "answer": answer}
    except Exception as e:
        print(f"❌ Decoder failed: {e}")
        return {"status": "error", "error": str(e)}

def main():
    print("=" * 60)
    print("E2E TEST: Embedding Service + HF Inference Endpoint")
    print("=" * 60)
    
    query = "What are the bail provisions under Section 436 of the CrPC?"
    print(f"\n📝 Query: {query}\n")
    
    # Test embedding
    embedding_result = test_embedding(query)
    
    # Test decoder
    decoder_result = test_decoder(query)
    
    # Print final answer
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if embedding_result["status"] == "ok":
        print(f"✅ Embedding: OK (dim={embedding_result['data'].get('embedding_dim', 0)})")
    else:
        print(f"❌ Embedding: {embedding_result.get('error', 'Unknown error')}")
    
    if decoder_result["status"] == "ok":
        print(f"✅ Decoder: OK")
        print(f"\n💬 Answer:\n{decoder_result.get('answer', 'No answer')}")
    else:
        print(f"❌ Decoder: {decoder_result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
