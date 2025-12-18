"""Generator component for RAG system"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import List, Dict, Optional
from .retriever import RetrievalResult


class RAGGenerator:
    """
    Generator for Retrieval-Augmented Generation.
    
    Combines retrieved documents with generation model to produce
    context-aware responses.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_context_length: int = 2048,
        max_generation_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ):
        """
        Args:
            model_name: Pretrained model name
            device: Device to run on
            max_context_length: Maximum context length
            max_generation_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling
            top_k: Top-k sampling
        """
        self.device = device
        self.max_context_length = max_context_length
        self.max_generation_length = max_generation_length
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Generation config
        self.generation_config = GenerationConfig(
            max_length=max_generation_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
    
    def generate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        num_docs_to_use: Optional[int] = None,
        include_scores: bool = False,
        custom_prompt_template: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate response based on query and retrieved documents.
        
        Args:
            query: User query
            retrieval_result: Retrieved documents and scores
            num_docs_to_use: Number of documents to use (default: all)
            include_scores: Whether to include relevance scores in prompt
            custom_prompt_template: Custom prompt template
            
        Returns:
            Dictionary with generated text and metadata
        """
        # Select documents to use
        if num_docs_to_use is None:
            num_docs_to_use = len(retrieval_result.documents)
        
        documents = retrieval_result.documents[:num_docs_to_use]
        scores = retrieval_result.scores[:num_docs_to_use]
        
        # Build context from documents
        context = self._build_context(documents, scores, include_scores)
        
        # Build prompt
        if custom_prompt_template:
            prompt = custom_prompt_template.format(
                context=context,
                query=query
            )
        else:
            prompt = self._build_default_prompt(context, query)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_context_length,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Extract only the generated part (after the prompt)
        generated_answer = generated_text[len(prompt):].strip()
        
        return {
            'answer': generated_answer,
            'query': query,
            'context': context,
            'prompt': prompt,
            'num_docs_used': len(documents),
            'metadata': retrieval_result.metadata
        }
    
    def _build_context(
        self,
        documents: List,
        scores: List[float],
        include_scores: bool
    ) -> str:
        """Build context string from documents"""
        context_parts = []
        
        for i, (doc, score) in enumerate(zip(documents, scores), 1):
            if include_scores:
                context_parts.append(
                    f"[Document {i}, Relevance: {score:.3f}]\n{doc.content}\n"
                )
            else:
                context_parts.append(f"[Document {i}]\n{doc.content}\n")
        
        return "\n".join(context_parts)
    
    def _build_default_prompt(self, context: str, query: str) -> str:
        """Build default prompt template"""
        return f"""You are a legal assistant. Use the following legal documents to answer the question.

Documents:
{context}

Question: {query}

Answer: """
    
    def batch_generate(
        self,
        queries: List[str],
        retrieval_results: List[RetrievalResult],
        **kwargs
    ) -> List[Dict]:
        """Generate responses for multiple queries"""
        return [
            self.generate(query, result, **kwargs)
            for query, result in zip(queries, retrieval_results)
        ]


class ChainOfThoughtGenerator(RAGGenerator):
    """
    Generator with chain-of-thought reasoning for complex legal questions.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def generate_with_reasoning(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        num_reasoning_steps: int = 3
    ) -> Dict[str, any]:
        """
        Generate answer with explicit reasoning steps.
        
        Args:
            query: User query
            retrieval_result: Retrieved documents
            num_reasoning_steps: Number of reasoning steps
            
        Returns:
            Dictionary with answer and reasoning steps
        """
        documents = retrieval_result.documents
        context = self._build_context(documents, retrieval_result.scores, False)
        
        # Build chain-of-thought prompt
        prompt = f"""You are a legal assistant. Analyze the following legal documents and answer the question using step-by-step reasoning.

Documents:
{context}

Question: {query}

Let's think step by step:

Step 1:"""
        
        # Generate with reasoning
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_context_length,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_generation_length,
                temperature=0.7,
                do_sample=True
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        reasoning = generated_text[len(prompt):].strip()
        
        # Extract final answer (typically after last reasoning step)
        answer_parts = reasoning.split("Answer:")
        if len(answer_parts) > 1:
            final_answer = answer_parts[-1].strip()
        else:
            final_answer = reasoning
        
        return {
            'answer': final_answer,
            'reasoning': reasoning,
            'query': query,
            'context': context,
            'num_docs_used': len(documents)
        }


class SummarizationGenerator(RAGGenerator):
    """
    Specialized generator for document summarization tasks.
    """
    
    def summarize_documents(
        self,
        retrieval_result: RetrievalResult,
        summary_length: str = "medium"
    ) -> Dict[str, str]:
        """
        Summarize retrieved documents.
        
        Args:
            retrieval_result: Retrieved documents to summarize
            summary_length: "short", "medium", or "long"
            
        Returns:
            Dictionary with summary and metadata
        """
        documents = retrieval_result.documents
        
        # Set max length based on summary_length
        length_map = {
            "short": 100,
            "medium": 250,
            "long": 500
        }
        max_length = length_map.get(summary_length, 250)
        
        # Combine all document content
        combined_text = "\n\n".join([doc.content for doc in documents])
        
        # Build summarization prompt
        prompt = f"""Summarize the following legal documents concisely:

{combined_text}

Summary:"""
        
        # Generate summary
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_context_length,
            truncation=True
        ).to(self.device)
        
        # Adjust generation config for summarization
        summary_config = GenerationConfig(
            max_length=max_length,
            min_length=30,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=summary_config
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = generated_text[len(prompt):].strip()
        
        return {
            'summary': summary,
            'num_docs_summarized': len(documents),
            'summary_length': summary_length,
            'original_query': retrieval_result.query
        }


class CitationGenerator(RAGGenerator):
    """
    Generator that includes proper citations for legal responses.
    """
    
    def generate_with_citations(
        self,
        query: str,
        retrieval_result: RetrievalResult
    ) -> Dict[str, any]:
        """
        Generate answer with proper legal citations.
        
        Args:
            query: User query
            retrieval_result: Retrieved documents with metadata
            
        Returns:
            Dictionary with answer and citations
        """
        documents = retrieval_result.documents
        
        # Build context with citation markers
        context_parts = []
        citations = []
        
        for i, doc in enumerate(documents, 1):
            citation_id = f"[{i}]"
            context_parts.append(f"{citation_id} {doc.content}")
            
            # Extract citation info from metadata
            citation_info = {
                'id': citation_id,
                'source': doc.metadata.get('source', 'Unknown'),
                'date': doc.metadata.get('date', 'N/A'),
                'title': doc.metadata.get('title', 'Untitled')
            }
            citations.append(citation_info)
        
        context = "\n\n".join(context_parts)
        
        # Build prompt with citation instructions
        prompt = f"""You are a legal assistant. Use the following legal documents to answer the question. 
Include citation numbers [1], [2], etc. when referencing specific documents.

Documents:
{context}

Question: {query}

Answer (with citations): """
        
        # Generate
        result = self.generate(
            query,
            retrieval_result,
            custom_prompt_template=prompt
        )
        
        result['citations'] = citations
        
        return result
