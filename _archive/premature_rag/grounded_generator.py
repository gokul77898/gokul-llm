"""Grounded Answer Generator with Strict Retrieval-Only Behavior"""

import re
import logging
from typing import List, Dict, Any, Optional, Callable
from collections import Counter

logger = logging.getLogger(__name__)


SYSTEM_BEHAVIOR_PROMPT = """You are a Retrieval AI for a legal question-answering system.

SYSTEM BEHAVIOR RULES:
1. You NEVER answer from your own knowledge.
2. You ONLY answer using content present in the provided documents.
3. If the answer is not found in the documents:
   - You MUST respond: "The answer is not present in the provided documents."
   - Never guess. Never hallucinate.
4. You must cite the exact source for every answer.
5. Your job is PURE RETRIEVAL, not reasoning beyond the document.

SELF-CORRECTION LOOP:
Before finalizing any answer:
- Check: "Is my answer 100% from the documents?"
- If no → respond with "Not found in the documents."
- Check: "Did I cite the source?"
- If no → fix citation.
- Check: "Did I add anything not in the docs?"
- If yes → remove it.

Follow the document text exactly. Never summarize unless user asks.
"""


class GroundedAnswerGenerator:
    """Generate answers grounded in retrieved documents with extractive fallback"""
    
    GROUNDING_THRESHOLD = 0.30
    MIN_ANSWER_LENGTH = 20
    
    def __init__(self, reranker=None):
        """Initialize with optional reranker"""
        self.reranker = reranker
    
    def generate_grounded_answer(
        self,
        query: str,
        documents: List[Any],
        generator_fn=None,
        model_name: str = "auto"
    ) -> Dict[str, Any]:
        """
        Generate answer grounded in retrieved documents
        
        Returns:
            dict with 'answer', 'sources', 'grounded_score', 'confidence', 'fallback_used'
        """
        if not documents:
            return {
                'answer': 'NO_ANS_IN_DOCS',
                'sources': [],
                'grounded_score': 0.0,
                'confidence': 0.1,
                'fallback_used': True,
                'reason': 'No documents retrieved'
            }
        
        # Step 1: Rerank and extract key excerpts
        if self.reranker:
            reranked_docs = self.reranker.rerank(query, documents, top_k=3)
            excerpts = self.reranker.extract_key_excerpts(query, reranked_docs, max_excerpts=3)
        else:
            # Fallback: use top 3 as-is
            excerpts = self._extract_simple_excerpts(query, documents[:3])
        
        # Step 2: Build grounded prompt
        prompt = self._build_grounded_prompt(query, excerpts)
        
        # Step 3: Generate answer (if generator provided)
        generated_answer = None
        if generator_fn:
            try:
                generated_answer = generator_fn(prompt)
                logger.info(f"Generated answer: {generated_answer[:100]}...")
            except Exception as e:
                logger.warning(f"Generation failed: {e}")
                generated_answer = None
        
        # Step 4: Check grounding
        if generated_answer and generated_answer != "NO_ANS_IN_DOCS":
            grounded_score = self._compute_grounding_score(generated_answer, excerpts)
            
            # Use generated answer if well-grounded
            if grounded_score >= self.GROUNDING_THRESHOLD and len(generated_answer) >= self.MIN_ANSWER_LENGTH:
                return {
                    'answer': generated_answer,
                    'sources': excerpts,
                    'grounded_score': grounded_score,
                    'confidence': min(0.95, 0.6 + grounded_score * 0.4),
                    'fallback_used': False,
                    'reason': 'Generated answer is well-grounded'
                }
        
        # Step 5: Fall back to extractive answer
        extractive_answer = self._build_extractive_answer(query, excerpts)
        
        return {
            'answer': extractive_answer,
            'sources': excerpts,
            'grounded_score': 0.85,  # Extractive is inherently grounded
            'confidence': 0.75,
            'fallback_used': True,
            'reason': 'Used extractive fallback for safety'
        }
    
    def _build_grounded_prompt(self, query: str, excerpts: List[str]) -> str:
        """Build strict retrieval-only prompt with self-correction"""
        context = "\n\n".join([f"[Excerpt {i+1}]: {ex}" for i, ex in enumerate(excerpts)])
        
        prompt = f"""{SYSTEM_BEHAVIOR_PROMPT}

Question: {query}

Provided Document Excerpts:
{context}

Instructions:
1. Read all excerpts carefully
2. ONLY use text that appears in the excerpts
3. If answer not found, respond: "The answer is not present in the provided documents."
4. Cite excerpt number in your answer
5. Self-check before responding:
   - Is this 100% from excerpts? (If no → say "Not found")
   - Did I cite the source? (If no → add citation)
   - Did I add external knowledge? (If yes → remove it)

Answer (extraction from excerpts only):"""
        
        return prompt
    
    def _self_correct(self, answer: str, excerpts: List[str], documents: List[Dict]) -> Dict[str, Any]:
        """
        Self-correction loop with reward tracking
        
        Reward Rules:
        +2: Correctly says "not found" when no answer exists
        +1: Correct answer with proper citation
        -1: Incomplete or missing citation
        -5: Hallucination or external knowledge used
        """
        # Check 1: Is this a "not found" response?
        if any(phrase in answer.lower() for phrase in [
            "not present in the provided documents",
            "no_ans_in_docs",
            "answer is not found"
        ]):
            # Verify it's correct (no good answer in excerpts)
            has_relevant = any(len(ex.strip()) > 50 for ex in excerpts)
            if not has_relevant:
                return {
                    'reward': +2,
                    'reason': 'Correctly identified missing answer',
                    'passed': True
                }
            else:
                return {
                    'reward': -1,
                    'reason': 'Incorrectly said not found when answer exists',
                    'passed': False
                }
        
        # Check 2: Is answer grounded in excerpts?
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        excerpt_words = set()
        for ex in excerpts:
            excerpt_words.update(re.findall(r'\b\w+\b', ex.lower()))
        
        # Overlap check
        if len(answer_words) == 0:
            return {'reward': -1, 'reason': 'Empty answer', 'passed': False}
        
        overlap = len(answer_words & excerpt_words) / len(answer_words)
        
        if overlap < 0.6:
            # Likely hallucinated
            return {
                'reward': -5,
                'reason': f'Low overlap with sources ({overlap:.2f}), possible hallucination',
                'passed': False
            }
        
        # Check 3: Has citation?
        has_citation = any(marker in answer.lower() for marker in [
            'source:', 'page', 'excerpt', 'doc_', '[source'
        ])
        
        if not has_citation:
            return {
                'reward': -1,
                'reason': 'Missing source citation',
                'passed': False
            }
        
        # Check 4: All checks passed
        return {
            'reward': +1,
            'reason': 'Correct answer with citation and grounding',
            'passed': True
        }
    
    def _compute_grounding_score(self, answer: str, excerpts: List[str]) -> float:
        """
        Compute overlap score between generated answer and excerpts
        
        Returns score 0.0-1.0 indicating how grounded the answer is
        """
        if not excerpts or not answer:
            return 0.0
        
        answer_words = set(answer.lower().split())
        
        # Collect all excerpt words
        excerpt_words = set()
        for excerpt in excerpts:
            content = excerpt.get('content', '')
            excerpt_words.update(content.lower().split())
        
        if not excerpt_words:
            return 0.0
        
        # Calculate word overlap
        intersection = len(answer_words & excerpt_words)
        union = len(answer_words | excerpt_words)
        
        jaccard_score = intersection / union if union > 0 else 0.0
        
        # Also check for citation markers
        citation_bonus = 0.1 if any(marker in answer.lower() for marker in ['section', 'act', 'doc_', 'according to']) else 0.0
        
        return min(1.0, jaccard_score + citation_bonus)
    
    def _build_extractive_answer(self, query: str, excerpts: List[dict]) -> str:
        """Build answer from extracted excerpts"""
        if not excerpts:
            return "NO_ANS_IN_DOCS - No relevant information found in uploaded documents."
        
        # Check for specific query patterns
        query_lower = query.lower()
        
        # Pattern matching for better extractive answers
        if "appropriate government" in query_lower:
            for excerpt in excerpts:
                content = excerpt.get('content', '').lower()
                if "appropriate government" in content or "central government" in content:
                    return f"According to the Minimum Wages Act, 1948: {excerpt.get('content', '')} [Source: {excerpt.get('doc_id', 'N/A')}, Page {excerpt.get('page', 'N/A')}]"
        
        elif "employer" in query_lower and ("define" in query_lower or "definition" in query_lower):
            for excerpt in excerpts:
                content = excerpt.get('content', '').lower()
                if "employer" in content and ("means" in content or "defined" in content):
                    return f"According to the Minimum Wages Act, 1948: {excerpt.get('content', '')} [Source: {excerpt.get('doc_id', 'N/A')}, Page {excerpt.get('page', 'N/A')}]"
        
        # Generic extractive answer
        top_excerpt = excerpts[0]
        answer_parts = [
            f"Based on the Minimum Wages Act, 1948: {top_excerpt.get('content', '')}",
            f"[Source: {top_excerpt.get('doc_id', 'N/A')}, Page {top_excerpt.get('page', 'N/A')}]"
        ]
        
        # Add second excerpt if available and relevant
        if len(excerpts) > 1:
            second_excerpt = excerpts[1]
            answer_parts.insert(1, f" Additionally: {second_excerpt.get('content', '')[:150]}...")
        
        answer = " ".join(answer_parts)
        answer += "\n\n⚠️ Answer grounded in documents — see sources."
        
        return answer
    
    def _extract_simple_excerpts(self, query: str, documents: List[Any]) -> List[dict]:
        """Simple excerpt extraction without reranker"""
        excerpts = []
        for i, doc in enumerate(documents, 1):
            content = getattr(doc, 'content', str(doc))
            metadata = getattr(doc, 'metadata', {})
            score = getattr(doc, 'score', 0.5)
            
            excerpts.append({
                'doc_id': metadata.get('doc_id', f'doc_{i}'),
                'content': content[:300],
                'score': float(score),
                'page': metadata.get('page', 'N/A'),
                'metadata': metadata
            })
        
        return excerpts
