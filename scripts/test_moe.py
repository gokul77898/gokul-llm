import sys
import os
import json
import logging

# Fix import path
sys.path.append(os.getcwd())

from src.inference.moe_router import MoERouter
from src.core.model_registry import get_registry
from src.core.generator import Generator

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    print("🔍 Loading registry...")
    # Registry is loaded automatically by get_registry() / MoERouter
    registry = get_registry()
    print(f"   Loaded {len(registry.list_experts())} experts.")
    
    router = MoERouter()
    generator = Generator()

    tests = [
        {
            "text": "Extract all named entities from this legal sentence: The accused was charged under Section 302 of IPC.",
            "task": "ner"
        },
        {
            "text": "What is the punishment for murder under Section 302 IPC?",
            "task": "qa"
        },
        {
            "text": "Compare Section 304A and Section 302 IPC.",
            "task": "similarity"
        }
    ]

    for t in tests:
        print("\n==============================")
        print(f"📝 Input Text: {t['text']}")
        print(f"🎯 Task: {t['task']}")

        # Test Router
        route_results = router.route(text=t["text"], task_hint=t["task"])
        if not route_results:
            print("❌ Router failed to select an expert.")
            continue
            
        top_route = route_results[0]
        print("\n🤖 Router Output:")
        print(json.dumps(top_route, indent=2))

        # Test Generation
        # Note: Generator.generate uses router internally if model_key="auto"
        # But here we want to verifying routing matches or just use the generator flow.
        # We can pass model_key="auto" to let it route again, or pass the expert name.
        # The user's snippet implied passing text and task.
        # My generator.generate signature: generate(query, model_key="auto", top_k=5, max_length=256)
        
        print("\n⚡ Generating Output:")
        try:
            # We use the generate_with_expert for direct string output as per previous turn's "UI friendly" request
            # or the standard generate() which returns a dict.
            # Let's use standard generate() with model="auto" to test end-to-end
            
            output = generator.generate(
                query=t["text"],
                model_key="auto",
                max_length=128,
                task=t["task"]
            )
            print(f"🟢 Model Response:\n{output['answer']}")
            print(f"   (Used Model: {output['model']})")
            
        except Exception as e:
            print(f"🔴 Generation failed: {e}")

if __name__ == "__main__":
    main()
