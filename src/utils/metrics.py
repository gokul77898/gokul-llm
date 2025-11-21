"""Evaluation metrics for legal AI tasks"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: Optional[np.ndarray] = None
    additional_metrics: Optional[Dict] = None


def compute_metrics(
    predictions: List[int],
    labels: List[int],
    average: str = 'weighted'
) -> EvaluationMetrics:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        average: Averaging strategy ('micro', 'macro', 'weighted')
        
    Returns:
        EvaluationMetrics object
    """
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average=average, zero_division=0)
    recall = recall_score(labels, predictions, average=average, zero_division=0)
    f1 = f1_score(labels, predictions, average=average, zero_division=0)
    cm = confusion_matrix(labels, predictions)
    
    return EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=cm
    )


def compute_rouge_scores(
    generated: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute ROUGE scores for summarization.
    
    Args:
        generated: Generated summaries
        references: Reference summaries
        
    Returns:
        Dictionary of ROUGE scores
    """
    from collections import Counter
    
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens"""
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
    
    def rouge_n(gen: str, ref: str, n: int) -> float:
        """Compute ROUGE-N score"""
        gen_tokens = gen.lower().split()
        ref_tokens = ref.lower().split()
        
        gen_ngrams = get_ngrams(gen_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        
        overlap = sum((gen_ngrams & ref_ngrams).values())
        
        if sum(ref_ngrams.values()) == 0:
            return 0.0
        
        recall = overlap / sum(ref_ngrams.values())
        
        if sum(gen_ngrams.values()) == 0:
            return 0.0
        
        precision = overlap / sum(gen_ngrams.values())
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    # Compute ROUGE-1, ROUGE-2, ROUGE-L
    rouge_1_scores = [rouge_n(g, r, 1) for g, r in zip(generated, references)]
    rouge_2_scores = [rouge_n(g, r, 2) for g, r in zip(generated, references)]
    
    return {
        'rouge-1': np.mean(rouge_1_scores),
        'rouge-2': np.mean(rouge_2_scores),
    }


def compute_bleu_score(
    generated: List[str],
    references: List[str],
    max_n: int = 4
) -> float:
    """
    Compute BLEU score for generation.
    
    Args:
        generated: Generated texts
        references: Reference texts
        max_n: Maximum n-gram order
        
    Returns:
        BLEU score
    """
    from collections import Counter
    import math
    
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
    
    def modified_precision(gen: str, ref: str, n: int) -> float:
        gen_tokens = gen.lower().split()
        ref_tokens = ref.lower().split()
        
        gen_ngrams = get_ngrams(gen_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        
        clipped_counts = sum((gen_ngrams & ref_ngrams).values())
        total_counts = sum(gen_ngrams.values())
        
        if total_counts == 0:
            return 0.0
        
        return clipped_counts / total_counts
    
    # Compute precisions for each n-gram
    precisions = []
    for n in range(1, max_n + 1):
        prec_scores = [modified_precision(g, r, n) for g, r in zip(generated, references)]
        precisions.append(np.mean(prec_scores))
    
    # Geometric mean
    if min(precisions) == 0:
        return 0.0
    
    log_precisions = [math.log(p) for p in precisions if p > 0]
    geo_mean = math.exp(sum(log_precisions) / len(log_precisions))
    
    # Brevity penalty
    gen_length = sum(len(g.split()) for g in generated)
    ref_length = sum(len(r.split()) for r in references)
    
    if gen_length >= ref_length:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_length / gen_length)
    
    bleu = bp * geo_mean
    return bleu


def compute_perplexity(
    loss: float
) -> float:
    """
    Compute perplexity from loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity score
    """
    import math
    return math.exp(loss)


def print_metrics(metrics: EvaluationMetrics, prefix: str = ""):
    """
    Print evaluation metrics in formatted way.
    
    Args:
        metrics: EvaluationMetrics object
        prefix: Prefix for display
    """
    print(f"\n{prefix}Evaluation Metrics:")
    print(f"  Accuracy:  {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1 Score:  {metrics.f1:.4f}")
    
    if metrics.confusion_matrix is not None:
        print(f"\n  Confusion Matrix:")
        print(metrics.confusion_matrix)
    
    if metrics.additional_metrics:
        print(f"\n  Additional Metrics:")
        for key, value in metrics.additional_metrics.items():
            print(f"    {key}: {value:.4f}")
