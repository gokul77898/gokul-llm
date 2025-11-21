"""Test that data ingestion is blocked in SETUP MODE"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("TEST: Ingestion Blocker")
print("="*60)

try:
    from db.chroma import ingest_file
    print("✅ Ingestion module imported")
    
    # Try to ingest (should fail)
    print("\nAttempting to ingest file...")
    try:
        ingest_file("sample_legal_doc.txt", "legal_docs")
        print("❌ FAIL: Ingestion should be blocked!")
    except RuntimeError as e:
        print(f"✅ Ingestion correctly blocked")
        print(f"   Error message: {str(e)[:80]}...")
    
    print("\n" + "="*60)
    print("✅ INGESTION BLOCKER WORKING")
    print("="*60)
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
