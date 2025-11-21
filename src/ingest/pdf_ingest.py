import os
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb import PersistentClient

DATA_DIR = "/Users/gokul/Documents/data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pdf_docs"

def load_pdfs():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    docs = []

    for file in files:
        pdf_path = os.path.join(DATA_DIR, file)
        print(f"📄 Loading: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        docs.extend(pages)

    return docs

def chunk_docs(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(pages)

def save_to_chroma(chunks):
    client = PersistentClient(CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"chunk_{i}"],
            documents=[chunk.page_content],
            metadatas=[chunk.metadata]
        )
    print(f"✅ Inserted {len(chunks)} chunks into ChromaDB.")

if __name__ == "__main__":
    print("🚀 Starting PDF ingestion...")

    pages = load_pdfs()
    print(f"📑 Loaded {len(pages)} total PDF pages.")

    chunks = chunk_docs(pages)
    print(f"🔪 Created {len(chunks)} text chunks.")

    save_to_chroma(chunks)
    print("🎉 DONE! PDFs stored in ChromaDB.")
