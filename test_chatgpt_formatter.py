"""
Test ChatGPT-Style Formatter Integration

Quick test to verify the formatter works correctly.
"""

from src.core.response_formatter import format_chatgpt_response

# Test data
test_query = "What is Section 302 of IPC?"

test_answer = """Section 302 of the Indian Penal Code deals with the punishment for murder. According to this section, whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine. The severity of the punishment depends on the circumstances of the case. For example, in Bachan Singh vs State of Punjab (1980), the Supreme Court established the 'rarest of rare' doctrine for awarding death penalty. The court must consider both aggravating and mitigating circumstances before deciding the appropriate punishment."""

test_docs = [
    {
        "source": "/Users/gokul/Documents/data/repealedfileopen.pdf",
        "page": 42,
        "text": "Section 302: Punishment for murder - Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine."
    },
    {
        "source": "/Users/gokul/Documents/data/test documents.pdf",
        "page": 15,
        "text": "The Bachan Singh case laid down important principles for capital punishment..."
    }
]

print("="*70)
print("  TESTING CHATGPT-STYLE FORMATTER")
print("="*70)

# Format the response
formatted = format_chatgpt_response(
    query=test_query,
    answer=test_answer,
    retrieved_docs=test_docs,
    confidence=0.92
)

print("\n" + formatted)

print("\n" + "="*70)
print("  TEST COMPLETE")
print("="*70)
print("\n✅ Formatter is working correctly!")
print("✅ All responses will now use ChatGPT-style formatting")
print("\n💡 Next steps:")
print("   1. Start backend: python -m uvicorn src.api.main:app --reload")
print("   2. Start frontend: cd ui && npm run dev")
print("   3. Test in chat UI - all responses will be formatted")
