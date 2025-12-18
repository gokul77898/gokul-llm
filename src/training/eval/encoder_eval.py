"""
Phase 3.5: Encoder Evaluation

Evaluation functions for encoder (NER) models.
Computes: recall, false-positive rate, precision, F1.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EncoderEvalResult:
    """Result of encoder evaluation."""
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    entity_precision: float
    entity_recall: float
    entity_f1: float
    total_samples: int
    total_entities_predicted: int
    total_entities_gold: int
    per_label_metrics: Dict[str, Dict[str, float]]


class EncoderEvaluator:
    """
    Evaluator for encoder (NER) models.
    
    Computes token-level and entity-level metrics.
    """
    
    def __init__(self, label_list: List[str]):
        """
        Initialize evaluator.
        
        Args:
            label_list: List of valid labels (e.g., ["O", "B-SECTION", "I-SECTION", ...])
        """
        self.label_list = label_list
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.id2label = {i: label for i, label in enumerate(label_list)}
    
    def _extract_entities(
        self,
        labels: List[str],
        tokens: Optional[List[str]] = None
    ) -> List[Tuple[int, int, str]]:
        """
        Extract entity spans from BIO labels.
        
        Returns list of (start_idx, end_idx, label) tuples.
        """
        entities = []
        current_entity = None
        current_start = None
        
        for idx, label in enumerate(labels):
            if label.startswith("B-"):
                # Start new entity
                if current_entity is not None:
                    entities.append((current_start, idx, current_entity))
                current_entity = label[2:]
                current_start = idx
            elif label.startswith("I-"):
                # Continue entity if matching
                entity_type = label[2:]
                if current_entity != entity_type:
                    # Mismatched I- tag, treat as new entity
                    if current_entity is not None:
                        entities.append((current_start, idx, current_entity))
                    current_entity = entity_type
                    current_start = idx
            else:
                # O tag - end current entity
                if current_entity is not None:
                    entities.append((current_start, idx, current_entity))
                current_entity = None
                current_start = None
        
        # Handle entity at end of sequence
        if current_entity is not None:
            entities.append((current_start, len(labels), current_entity))
        
        return entities
    
    def compute_token_metrics(
        self,
        predictions: List[List[str]],
        references: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Compute token-level metrics.
        
        Args:
            predictions: List of predicted label sequences
            references: List of gold label sequences
        
        Returns:
            Dict with precision, recall, f1, false_positive_rate
        """
        from collections import defaultdict
        
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)
        true_negatives = 0
        
        for pred_seq, ref_seq in zip(predictions, references):
            for pred, ref in zip(pred_seq, ref_seq):
                if pred == ref and pred != "O":
                    true_positives[pred] += 1
                elif pred != "O" and ref == "O":
                    false_positives[pred] += 1
                elif pred == "O" and ref != "O":
                    false_negatives[ref] += 1
                elif pred == "O" and ref == "O":
                    true_negatives += 1
                elif pred != ref:
                    false_positives[pred] += 1
                    false_negatives[ref] += 1
        
        # Aggregate
        total_tp = sum(true_positives.values())
        total_fp = sum(false_positives.values())
        total_fn = sum(false_negatives.values())
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = total_fp / (total_fp + true_negatives) if (total_fp + true_negatives) > 0 else 0.0
        
        # Per-label metrics
        per_label = {}
        all_labels = set(true_positives.keys()) | set(false_positives.keys()) | set(false_negatives.keys())
        
        for label in all_labels:
            if label == "O":
                continue
            tp = true_positives[label]
            fp = false_positives[label]
            fn = false_negatives[label]
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            
            per_label[label] = {"precision": p, "recall": r, "f1": f}
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fpr,
            "per_label": per_label
        }
    
    def compute_entity_metrics(
        self,
        predictions: List[List[str]],
        references: List[List[str]]
    ) -> Dict[str, float]:
        """
        Compute entity-level metrics (exact span matching).
        
        Args:
            predictions: List of predicted label sequences
            references: List of gold label sequences
        
        Returns:
            Dict with entity_precision, entity_recall, entity_f1
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_pred = 0
        total_gold = 0
        
        for pred_seq, ref_seq in zip(predictions, references):
            pred_entities = set(self._extract_entities(pred_seq))
            ref_entities = set(self._extract_entities(ref_seq))
            
            tp = len(pred_entities & ref_entities)
            fp = len(pred_entities - ref_entities)
            fn = len(ref_entities - pred_entities)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_pred += len(pred_entities)
            total_gold += len(ref_entities)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "entity_precision": precision,
            "entity_recall": recall,
            "entity_f1": f1,
            "total_predicted": total_pred,
            "total_gold": total_gold
        }
    
    def evaluate(
        self,
        predictions: List[List[str]],
        references: List[List[str]]
    ) -> EncoderEvalResult:
        """
        Run full evaluation.
        
        Args:
            predictions: List of predicted label sequences
            references: List of gold label sequences
        
        Returns:
            EncoderEvalResult with all metrics
        """
        token_metrics = self.compute_token_metrics(predictions, references)
        entity_metrics = self.compute_entity_metrics(predictions, references)
        
        return EncoderEvalResult(
            precision=token_metrics["precision"],
            recall=token_metrics["recall"],
            f1=token_metrics["f1"],
            false_positive_rate=token_metrics["false_positive_rate"],
            entity_precision=entity_metrics["entity_precision"],
            entity_recall=entity_metrics["entity_recall"],
            entity_f1=entity_metrics["entity_f1"],
            total_samples=len(predictions),
            total_entities_predicted=entity_metrics["total_predicted"],
            total_entities_gold=entity_metrics["total_gold"],
            per_label_metrics=token_metrics["per_label"]
        )
    
    def evaluate_from_file(self, predictions_file: str, references_file: str) -> EncoderEvalResult:
        """
        Evaluate from JSONL files.
        
        Expected format per line:
        {"labels": ["O", "B-SECTION", "I-SECTION", ...]}
        """
        predictions = []
        references = []
        
        with open(predictions_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                predictions.append(data["labels"])
        
        with open(references_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                references.append(data["labels"])
        
        return self.evaluate(predictions, references)


def create_encoder_evaluator(label_list: List[str]) -> EncoderEvaluator:
    """Factory function to create an EncoderEvaluator."""
    return EncoderEvaluator(label_list)


if __name__ == "__main__":
    # Demo evaluation with synthetic data
    print("=" * 50)
    print("Encoder Evaluator Demo")
    print("=" * 50)
    
    labels = ["O", "B-SECTION", "I-SECTION", "B-ACT", "I-ACT"]
    evaluator = create_encoder_evaluator(labels)
    
    # Synthetic predictions and references
    predictions = [
        ["O", "B-SECTION", "I-SECTION", "O", "O"],
        ["O", "O", "B-ACT", "I-ACT", "O"],
        ["B-SECTION", "O", "O", "O", "O"]  # False positive
    ]
    
    references = [
        ["O", "B-SECTION", "I-SECTION", "O", "O"],
        ["O", "B-ACT", "I-ACT", "I-ACT", "O"],  # Missed I-ACT
        ["O", "O", "O", "O", "O"]  # False positive in pred
    ]
    
    result = evaluator.evaluate(predictions, references)
    
    print(f"\nToken-level Metrics:")
    print(f"  Precision: {result.precision:.3f}")
    print(f"  Recall: {result.recall:.3f}")
    print(f"  F1: {result.f1:.3f}")
    print(f"  False Positive Rate: {result.false_positive_rate:.3f}")
    
    print(f"\nEntity-level Metrics:")
    print(f"  Precision: {result.entity_precision:.3f}")
    print(f"  Recall: {result.entity_recall:.3f}")
    print(f"  F1: {result.entity_f1:.3f}")
    
    print(f"\nPer-label Metrics:")
    for label, metrics in result.per_label_metrics.items():
        print(f"  {label}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
