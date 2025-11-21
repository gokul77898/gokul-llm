"""Reward Calculation for Legal Document Tasks"""

import torch
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class RewardType(Enum):
    """Types of rewards"""
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    ROUGE = "rouge"
    BLEU = "bleu"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    COMPLETENESS = "completeness"


@dataclass
class RewardMetrics:
    """Container for reward metrics"""
    total_reward: float
    component_rewards: Dict[str, float]
    metadata: Optional[Dict] = None


class RewardCalculator:
    """
    Advanced reward calculator for legal document tasks.
    
    Computes rewards based on:
    - Accuracy and correctness
    - Relevance and coherence
    - Legal-specific criteria
    """
    
    def __init__(
        self,
        reward_weights: Optional[Dict[str, float]] = None,
        use_shaped_rewards: bool = True
    ):
        """
        Args:
            reward_weights: Weights for different reward components
            use_shaped_rewards: Whether to use reward shaping
        """
        self.reward_weights = reward_weights or {
            'accuracy': 1.0,
            'relevance': 0.5,
            'coherence': 0.3,
            'completeness': 0.4,
            'legal_compliance': 0.6
        }
        self.use_shaped_rewards = use_shaped_rewards
    
    def calculate_summarization_reward(
        self,
        generated_summary: str,
        reference_summary: str,
        source_document: str
    ) -> RewardMetrics:
        """
        Calculate reward for summarization task.
        
        Args:
            generated_summary: Generated summary
            reference_summary: Reference summary
            source_document: Source document
            
        Returns:
            RewardMetrics with total and component rewards
        """
        component_rewards = {}
        
        # ROUGE score (overlap-based)
        rouge_score = self._compute_rouge(generated_summary, reference_summary)
        component_rewards['rouge'] = rouge_score
        
        # Length penalty
        length_ratio = len(generated_summary.split()) / max(len(reference_summary.split()), 1)
        length_penalty = 1.0 if 0.8 <= length_ratio <= 1.2 else 0.5
        component_rewards['length'] = length_penalty
        
        # Relevance to source
        relevance_score = self._compute_relevance(generated_summary, source_document)
        component_rewards['relevance'] = relevance_score
        
        # Coherence
        coherence_score = self._compute_coherence(generated_summary)
        component_rewards['coherence'] = coherence_score
        
        # Legal term preservation
        legal_score = self._compute_legal_term_preservation(
            generated_summary, source_document
        )
        component_rewards['legal_terms'] = legal_score
        
        # Weighted total
        total_reward = (
            component_rewards['rouge'] * self.reward_weights['accuracy'] +
            component_rewards['relevance'] * self.reward_weights['relevance'] +
            component_rewards['coherence'] * self.reward_weights['coherence'] +
            component_rewards['legal_terms'] * self.reward_weights['legal_compliance']
        )
        
        return RewardMetrics(
            total_reward=total_reward,
            component_rewards=component_rewards
        )
    
    def calculate_qa_reward(
        self,
        predicted_answer: str,
        reference_answer: str,
        context: str,
        question: str
    ) -> RewardMetrics:
        """
        Calculate reward for question answering.
        
        Args:
            predicted_answer: Predicted answer
            reference_answer: Reference answer
            context: Context document
            question: Question
            
        Returns:
            RewardMetrics
        """
        component_rewards = {}
        
        # Exact match
        exact_match = float(predicted_answer.strip().lower() == reference_answer.strip().lower())
        component_rewards['exact_match'] = exact_match
        
        # F1 score (token overlap)
        f1 = self._compute_f1(predicted_answer, reference_answer)
        component_rewards['f1'] = f1
        
        # Answer is from context
        in_context = float(predicted_answer.lower() in context.lower())
        component_rewards['in_context'] = in_context
        
        # Relevance to question
        relevance = self._compute_relevance(predicted_answer, question)
        component_rewards['relevance'] = relevance
        
        # Total reward
        total_reward = (
            component_rewards['f1'] * self.reward_weights['accuracy'] +
            component_rewards['relevance'] * self.reward_weights['relevance'] +
            component_rewards['in_context'] * 0.2
        )
        
        return RewardMetrics(
            total_reward=total_reward,
            component_rewards=component_rewards
        )
    
    def calculate_classification_reward(
        self,
        predicted_class: int,
        true_class: int,
        confidence: Optional[float] = None
    ) -> RewardMetrics:
        """
        Calculate reward for classification.
        
        Args:
            predicted_class: Predicted class
            true_class: True class
            confidence: Prediction confidence
            
        Returns:
            RewardMetrics
        """
        component_rewards = {}
        
        # Accuracy
        correct = float(predicted_class == true_class)
        component_rewards['accuracy'] = correct
        
        # Confidence-based reward
        if confidence is not None:
            if correct:
                # Reward high confidence on correct predictions
                component_rewards['confidence'] = confidence
            else:
                # Penalize high confidence on wrong predictions
                component_rewards['confidence'] = -(confidence * 0.5)
        
        # Total reward
        total_reward = (
            component_rewards['accuracy'] * 10.0 +
            component_rewards.get('confidence', 0.0) * 2.0
        )
        
        return RewardMetrics(
            total_reward=total_reward,
            component_rewards=component_rewards
        )
    
    def _compute_rouge(self, generated: str, reference: str) -> float:
        """
        Compute ROUGE-L score (simplified).
        
        Args:
            generated: Generated text
            reference: Reference text
            
        Returns:
            ROUGE score
        """
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()
        
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        # Compute longest common subsequence
        lcs_length = self._lcs_length(gen_tokens, ref_tokens)
        
        # Precision and recall
        precision = lcs_length / len(gen_tokens) if gen_tokens else 0
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return f1
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _compute_f1(self, predicted: str, reference: str) -> float:
        """
        Compute F1 score based on token overlap.
        
        Args:
            predicted: Predicted text
            reference: Reference text
            
        Returns:
            F1 score
        """
        pred_tokens = set(predicted.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        common = pred_tokens & ref_tokens
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return f1
    
    def _compute_relevance(self, text: str, reference: str) -> float:
        """
        Compute relevance score using word overlap.
        
        Args:
            text: Text to evaluate
            reference: Reference text
            
        Returns:
            Relevance score
        """
        text_words = set(text.lower().split())
        ref_words = set(reference.lower().split())
        
        if not text_words or not ref_words:
            return 0.0
        
        overlap = len(text_words & ref_words)
        relevance = overlap / max(len(ref_words), 1)
        
        return min(relevance, 1.0)
    
    def _compute_coherence(self, text: str) -> float:
        """
        Compute coherence score (simplified).
        
        Args:
            text: Text to evaluate
            
        Returns:
            Coherence score
        """
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0
        
        # Simple coherence: check if sentences have overlapping words
        coherence_scores = []
        
        for i in range(len(sentences) - 1):
            sent1_words = set(sentences[i].lower().split())
            sent2_words = set(sentences[i+1].lower().split())
            
            if sent1_words and sent2_words:
                overlap = len(sent1_words & sent2_words)
                score = overlap / max(len(sent1_words), len(sent2_words))
                coherence_scores.append(score)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _compute_legal_term_preservation(
        self,
        generated: str,
        source: str
    ) -> float:
        """
        Check if important legal terms are preserved.
        
        Args:
            generated: Generated text
            source: Source text
            
        Returns:
            Preservation score
        """
        # Common legal terms (simplified list)
        legal_terms = {
            'plaintiff', 'defendant', 'court', 'statute', 'section',
            'article', 'clause', 'judgment', 'appeal', 'contract',
            'agreement', 'liability', 'damages', 'evidence', 'witness'
        }
        
        source_words = set(source.lower().split())
        generated_words = set(generated.lower().split())
        
        # Find legal terms in source
        source_legal = source_words & legal_terms
        
        if not source_legal:
            return 1.0  # No legal terms to preserve
        
        # Check preservation
        generated_legal = generated_words & legal_terms
        preserved = source_legal & generated_legal
        
        preservation_rate = len(preserved) / len(source_legal)
        
        return preservation_rate
    
    def add_reward_shaping(
        self,
        base_reward: float,
        current_state: Dict,
        next_state: Dict
    ) -> float:
        """
        Add reward shaping for smoother learning.
        
        Args:
            base_reward: Base reward
            current_state: Current state info
            next_state: Next state info
            
        Returns:
            Shaped reward
        """
        if not self.use_shaped_rewards:
            return base_reward
        
        # Potential-based reward shaping
        # Phi(s) = some heuristic value of state s
        
        # Simple example: reward progress
        current_progress = current_state.get('progress', 0.0)
        next_progress = next_state.get('progress', 0.0)
        
        progress_bonus = (next_progress - current_progress) * 0.1
        
        shaped_reward = base_reward + progress_bonus
        
        return shaped_reward
