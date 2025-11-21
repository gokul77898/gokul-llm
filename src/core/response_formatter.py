"""
ChatGPT-Style Response Formatter

Formats all RAG/model responses into structured ChatGPT-style output with:
- Title
- Summary
- Detailed Explanation
- Examples
- Final Answer
"""

from typing import Dict, List, Any, Optional
import re


class ChatGPTFormatter:
    """Format responses in ChatGPT style with structure."""
    
    @staticmethod
    def format_response(
        query: str,
        answer: str,
        context_docs: Optional[List[Dict]] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Format response in ChatGPT style.
        
        Args:
            query: User's question
            answer: Raw answer from model/RAG
            context_docs: Retrieved context documents
            confidence: Confidence score
            metadata: Additional metadata
            
        Returns:
            Formatted response string
        """
        # Extract title from query or answer
        title = ChatGPTFormatter._extract_title(query, answer)
        
        # Generate summary
        summary = ChatGPTFormatter._generate_summary(answer)
        
        # Structure explanation
        explanation = ChatGPTFormatter._structure_explanation(answer)
        
        # Extract or generate example
        example = ChatGPTFormatter._extract_example(answer, context_docs)
        
        # Generate final answer
        final_answer = ChatGPTFormatter._generate_final_answer(answer)
        
        # Build formatted response
        formatted = f"""### 💬 {title}

#### 🔹 Summary
{summary}

#### 🔹 Detailed Explanation
{explanation}
"""
        
        # Add example if available
        if example:
            formatted += f"""
#### 🔹 Example
{example}
"""
        
        # Add references if available
        if context_docs and len(context_docs) > 0:
            references = ChatGPTFormatter._format_references(context_docs)
            formatted += f"""
#### 🔹 References
{references}
"""
        
        # Add final answer
        formatted += f"""
#### 🔹 Final Answer
{final_answer}
"""
        
        # Add confidence if available
        if confidence is not None:
            formatted += f"""
---
*Confidence: {confidence:.1%}*
"""
        
        return formatted.strip()
    
    @staticmethod
    def _extract_title(query: str, answer: str) -> str:
        """Extract or generate title from query."""
        # Clean query
        query_clean = query.strip().rstrip('?!.')
        
        # Capitalize first letter
        if query_clean:
            return query_clean[0].upper() + query_clean[1:]
        
        # Fallback: use first few words of answer
        words = answer.split()[:5]
        return ' '.join(words) + "..."
    
    @staticmethod
    def _generate_summary(answer: str) -> str:
        """Generate 2-3 line summary from answer."""
        # Take first 2 sentences or 200 characters
        sentences = re.split(r'[.!?]+', answer)
        
        summary_parts = []
        char_count = 0
        
        for sentence in sentences[:3]:
            sentence = sentence.strip()
            if sentence and char_count < 200:
                summary_parts.append(sentence)
                char_count += len(sentence)
        
        summary = '. '.join(summary_parts)
        if not summary.endswith('.'):
            summary += '.'
        
        return summary
    
    @staticmethod
    def _structure_explanation(answer: str) -> str:
        """Structure answer into bullet points."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', answer)
        
        # Filter and create bullets
        bullets = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 20:  # Skip very short sentences
                bullets.append(f"- {sentence}")
        
        # Limit to top 5 most relevant points
        bullets = bullets[:5]
        
        return '\n'.join(bullets) if bullets else answer
    
    @staticmethod
    def _extract_example(answer: str, context_docs: Optional[List[Dict]] = None) -> Optional[str]:
        """Extract or generate example from answer or context."""
        # Look for example keywords in answer
        example_patterns = [
            r'(?:for example|for instance|e\.g\.|such as)[,:]?\s*([^.!?]+)',
            r'example[:]?\s*([^.!?]+)',
        ]
        
        for pattern in example_patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                example_text = match.group(1).strip()
                return f"```\n{example_text}\n```"
        
        # If context docs available, extract snippet
        if context_docs and len(context_docs) > 0:
            first_doc = context_docs[0]
            if 'text' in first_doc:
                snippet = first_doc['text'][:200] + "..."
                return f"```\n{snippet}\n```"
        
        return None
    
    @staticmethod
    def _format_references(context_docs: List[Dict]) -> str:
        """Format reference documents."""
        references = []
        
        for i, doc in enumerate(context_docs[:3], 1):
            source = doc.get('source', 'Unknown')
            page = doc.get('page', 'N/A')
            
            ref = f"{i}. **Source:** {source}"
            if page != 'N/A':
                ref += f" (Page {page})"
            
            references.append(ref)
        
        return '\n'.join(references)
    
    @staticmethod
    def _generate_final_answer(answer: str) -> str:
        """Generate concise final answer (1-2 sentences)."""
        # Take first sentence as final answer
        sentences = re.split(r'[.!?]+', answer)
        
        if sentences:
            final = sentences[0].strip()
            if final:
                return final + '.'
        
        # Fallback: first 100 chars
        return answer[:100].strip() + '...'


def format_chatgpt_response(
    query: str,
    answer: str,
    retrieved_docs: Optional[List[Dict]] = None,
    confidence: Optional[float] = None,
    **kwargs
) -> str:
    """
    Convenience function for formatting responses.
    
    Args:
        query: User question
        answer: Model answer
        retrieved_docs: Retrieved context documents
        confidence: Confidence score
        **kwargs: Additional metadata
        
    Returns:
        Formatted ChatGPT-style response
    """
    return ChatGPTFormatter.format_response(
        query=query,
        answer=answer,
        context_docs=retrieved_docs,
        confidence=confidence,
        metadata=kwargs
    )


# Example usage
if __name__ == "__main__":
    # Test the formatter
    sample_query = "What is Section 302 IPC"
    sample_answer = ("Section 302 IPC deals with punishment for murder. "
                    "It states that whoever commits murder shall be punished with death "
                    "or imprisonment for life. The court has discretion in sentencing. "
                    "For example, in Bachan Singh vs State of Punjab, the Supreme Court "
                    "established the 'rarest of rare' doctrine for death penalty.")
    
    sample_docs = [
        {"source": "IPC.pdf", "page": 42, "text": "Section 302: Punishment for murder..."},
        {"source": "Case_Law.pdf", "page": 15, "text": "Bachan Singh judgment..."}
    ]
    
    formatted = format_chatgpt_response(
        query=sample_query,
        answer=sample_answer,
        retrieved_docs=sample_docs,
        confidence=0.92
    )
    
    print(formatted)
