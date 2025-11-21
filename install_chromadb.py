#!/usr/bin/env python3
"""
ChromaDB Installation Helper

This script helps install ChromaDB and its dependencies.
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ SUCCESS: {cmd}")
            return True
        else:
            print(f"❌ FAILED: {cmd}")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ ERROR running {cmd}: {e}")
        return False

def main():
    print("🔧 ChromaDB Installation Helper")
    print("=" * 50)
    
    # Check current Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Try different installation methods
    installation_methods = [
        # Method 1: Try with --user flag
        "pip3 install --user chromadb sentence-transformers",
        
        # Method 2: Try with --break-system-packages (if user confirms)
        "pip3 install --break-system-packages chromadb sentence-transformers",
        
        # Method 3: Try with python3 -m pip
        "python3 -m pip install --user chromadb sentence-transformers",
    ]
    
    print("\n📦 Attempting to install ChromaDB...")
    
    for i, method in enumerate(installation_methods, 1):
        print(f"\n🔄 Method {i}: {method}")
        
        if i == 2:  # --break-system-packages method
            response = input("⚠️  This method uses --break-system-packages. Continue? (y/N): ")
            if response.lower() != 'y':
                print("   Skipped by user")
                continue
        
        if run_command(method):
            print("✅ ChromaDB installation successful!")
            break
    else:
        print("\n❌ All installation methods failed.")
        print("\n🔧 Manual installation options:")
        print("1. Create a virtual environment:")
        print("   python3 -m venv venv")
        print("   source venv/bin/activate")
        print("   pip install chromadb sentence-transformers")
        print("\n2. Use conda:")
        print("   conda install -c conda-forge chromadb")
        print("\n3. Use pipx:")
        print("   brew install pipx")
        print("   pipx install chromadb")
        return False
    
    # Test the installation
    print("\n🧪 Testing ChromaDB installation...")
    try:
        import chromadb
        print("✅ ChromaDB imported successfully!")
        print(f"   Version: {chromadb.__version__}")
        
        # Test ChromaManager
        print("\n🔍 Testing ChromaManager...")
        sys.path.append(os.getcwd())
        from src.core.chroma_manager import ChromaManager
        
        db = ChromaManager()
        if db.is_ready():
            print("✅ ChromaManager initialized successfully!")
            stats = db.stats()
            print(f"📊 Stats: {stats}")
        else:
            print("⚠️  ChromaManager created but not fully initialized")
            print("   This is normal if no documents have been ingested yet")
        
        return True
        
    except ImportError as e:
        print(f"❌ ChromaDB import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ ChromaManager test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 ChromaDB setup complete!")
        print("\n📋 Next steps:")
        print("1. Ingest some documents:")
        print("   python3 src/ingest/chroma_ingest.py --file your_document.pdf")
        print("2. Test the ChromaManager:")
        print("   python3 -c 'from src.core.chroma_manager import ChromaManager; print(ChromaManager().stats())'")
    else:
        print("\n❌ ChromaDB setup failed. Please try manual installation.")
    
    sys.exit(0 if success else 1)
