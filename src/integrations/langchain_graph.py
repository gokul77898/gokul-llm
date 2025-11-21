"""LangChain Integration for MARK System"""

import torch
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from langchain.schema import Document as LangChainDocument
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.document_loaders import TextLoader, DirectoryLoader
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Install with: pip install langchain")

from src.core import load_model
from src.rag.document_store import Document

logger = logging.getLogger(__name__)


class MARKLLMWrapper:
    """Wrapper to use MARK models as LangChain LLMs"""
    
    def __init__(self, model_name: str = "mamba", device: Optional[str] = None):
        self.model_name = model_name
        self.model, self.tokenizer, self.device = load_model(model_name, device)
        logger.info(f"Initialized MARK LLM wrapper with {model_name}")
    
    def __call__(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Generate response for a prompt"""
        return self.generate(prompt, **kwargs)
    
    def generate(self, prompt: str, max_length: int = 256, **kwargs) -> str:
        """Generate text response"""
        # Encode prompt
        if self.model_name == "mamba":
            encoding = self.tokenizer.encode(prompt, return_tensors=False)
            input_ids = torch.tensor([encoding['input_ids'][:max_length]]).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_ids, task="generation")
                logits = outputs['logits']
                predicted_ids = torch.argmax(logits, dim=-1)
                response = self.tokenizer.decode(predicted_ids[0].cpu().tolist())
            
        elif self.model_name == "transformer":
            encoding = self.tokenizer.tokenize(
                prompt,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, predicted_class].item()
                response = f"Classification: Class {predicted_class} (confidence: {confidence:.2f})"
        
        else:
            response = f"Model {self.model_name} generation not implemented"
        
        return response


class MARKRetrieverWrapper:
    """Wrapper to use MARK RAG as LangChain retriever"""
    
    def __init__(self, device: Optional[str] = None, top_k: int = 5):
        self.retriever, self.embedding_model, self.device = load_model("rag_encoder", device)
        self.top_k = top_k
        logger.info("Initialized MARK retriever wrapper")
    
    def get_relevant_documents(self, query: str) -> List[LangChainDocument]:
        """Retrieve relevant documents"""
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain is required for this functionality")
        
        results = self.retriever.search(query, top_k=self.top_k)
        
        # Convert to LangChain documents
        documents = []
        for result in results:
            doc = LangChainDocument(
                page_content=result['document'].get('text', ''),
                metadata=result.get('metadata', {})
            )
            documents.append(doc)
        
        return documents


class MARKLangChainGraph:
    """
    LangChain graph integration for MARK system
    
    Features:
    - Document loading from PDF, txt
    - Vector store (FAISS)
    - Conversational memory
    - RAG retrieval
    - MARK models as LLM backends
    - Tool integration
    """
    
    def __init__(
        self,
        llm_model: str = "mamba",
        retriever_top_k: int = 5,
        device: Optional[str] = None
    ):
        """
        Initialize LangChain graph
        
        Args:
            llm_model: MARK model to use as LLM
            retriever_top_k: Number of documents to retrieve
            device: Device to use
        """
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain is required. Install with: pip install langchain")
        
        self.llm_model = llm_model
        self.device = device
        self.retriever_top_k = retriever_top_k
        
        logger.info(f"Initializing LangChain graph with {llm_model}")
        
        # Initialize components
        self.llm = MARKLLMWrapper(llm_model, device)
        self.retriever = MARKRetrieverWrapper(device, retriever_top_k)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        self.documents: List[LangChainDocument] = []
        
        logger.info("LangChain graph initialized")
    
    def load_documents(self, path: str, file_type: str = "txt") -> int:
        """
        Load documents from path
        
        Args:
            path: Path to document(s)
            file_type: Type of files to load
            
        Returns:
            Number of documents loaded
        """
        path_obj = Path(path)
        
        if path_obj.is_file():
            # Load single file
            if file_type == "txt":
                loader = TextLoader(str(path_obj))
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            docs = loader.load()
        
        elif path_obj.is_dir():
            # Load directory
            if file_type == "txt":
                loader = DirectoryLoader(str(path_obj), glob="**/*.txt", loader_cls=TextLoader)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            docs = loader.load()
        
        else:
            raise ValueError(f"Path not found: {path}")
        
        # Split documents
        split_docs = self.text_splitter.split_documents(docs)
        self.documents.extend(split_docs)
        
        logger.info(f"Loaded {len(split_docs)} document chunks from {path}")
        return len(split_docs)
    
    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a single document"""
        doc = LangChainDocument(page_content=text, metadata=metadata or {})
        self.documents.append(doc)
        logger.info("Added document to graph")
    
    def query(self, question: str, use_retrieval: bool = True) -> Dict[str, Any]:
        """
        Query the system
        
        Args:
            question: User question
            use_retrieval: Whether to use RAG retrieval
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query: {question[:100]}...")
        
        if use_retrieval:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.get_relevant_documents(question)
            
            # Prepare context
            context = "\n\n".join([doc.page_content for doc in retrieved_docs[:3]])
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        # Generate answer
        answer = self.llm.generate(prompt)
        
        # Update memory
        self.memory.save_context({"input": question}, {"output": answer})
        
        result = {
            'answer': answer,
            'question': question,
            'retrieved_docs': len(retrieved_docs) if use_retrieval else 0,
            'model': self.llm_model
        }
        
        logger.info("Query processed")
        return result
    
    def chat(self, message: str) -> str:
        """
        Chat with the system (with memory)
        
        Args:
            message: User message
            
        Returns:
            Response string
        """
        result = self.query(message, use_retrieval=True)
        return result['answer']
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search documents
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of matching documents
        """
        k = top_k or self.retriever_top_k
        docs = self.retriever.get_relevant_documents(query)[:k]
        
        results = []
        for doc in docs:
            results.append({
                'text': doc.page_content,
                'metadata': doc.metadata
            })
        
        return results
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """
        Summarize text
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            
        Returns:
            Summary
        """
        prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
        summary = self.llm.generate(prompt, max_length=max_length)
        return summary
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        if hasattr(self.memory, 'chat_memory'):
            messages = self.memory.chat_memory.messages
            history = []
            for msg in messages:
                history.append({
                    'role': 'user' if hasattr(msg, 'content') and 'input' in str(msg) else 'assistant',
                    'content': str(msg.content) if hasattr(msg, 'content') else str(msg)
                })
            return history
        return []
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")


def demo():
    """Demo the LangChain integration"""
    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available. Install with: pip install langchain")
        return
    
    print("="*70)
    print("MARK LANGCHAIN INTEGRATION DEMO")
    print("="*70)
    
    # Initialize graph
    print("\n1. Initializing LangChain graph...")
    graph = MARKLangChainGraph(llm_model="mamba", device="cpu")
    
    # Add some documents
    print("\n2. Adding sample documents...")
    graph.add_document(
        "Contract law governs agreements between parties and enforces obligations.",
        metadata={"category": "contract"}
    )
    graph.add_document(
        "Criminal law addresses offenses against the state and prescribes punishments.",
        metadata={"category": "criminal"}
    )
    
    # Query
    print("\n3. Querying the system...")
    result = graph.query("What is contract law?")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    
    # Chat
    print("\n4. Testing chat (with memory)...")
    response = graph.chat("Tell me about criminal law")
    print(f"Response: {response}")
    
    # Search
    print("\n5. Searching documents...")
    search_results = graph.search("contract", top_k=2)
    print(f"Found {len(search_results)} results")
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)


if __name__ == '__main__':
    demo()
