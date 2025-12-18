"""
Phase 3.5: Training Metrics Utilities

Defines metrics for encoder and decoder evaluation.
"""

from typing import Dict, Any, List, Tuple
from collections import defaultdict
import numpy as np


# ─────────────────────────────────────────────
# Encoder Metrics (Token Classification)
# ─────────────────────────────────────────────

def compute_encoder_metrics(
    predictions: List[List[str]],
    references: List[List[str]],
    label_list: List[str]
) -> Dict[str, float]:
    """
    Compute encoder (NER) metrics.
    
    Args:
        predictions: List of predicted label sequences
        references: List of ground truth label sequences
        label_list: List of valid labels
    
    Returns:
        Dict with precision, recall, f1, false_positive_rate
    """
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    true_negatives = defaultdict(int)
    
    for pred_seq, ref_seq in zip(predictions, references):
        for pred, ref in zip(pred_seq, ref_seq):
            if pred == ref and pred != "O":
                true_positives[pred] += 1
            elif pred != "O" and ref == "O":
                false_positives[pred] += 1
            elif pred == "O" and ref != "O":
                false_negatives[ref] += 1
            elif pred == "O" and ref == "O":
                true_negatives["O"] += 1
            elif pred != ref:
                false_positives[pred] += 1
                false_negatives[ref] += 1
    
    # Aggregate metrics
    total_tp = sum(true_positives.values())
    total_fp = sum(false_positives.values())
    total_fn = sum(false_negatives.values())
    total_tn = sum(true_negatives.values())
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # False positive rate: FP / (FP + TN)
    fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn
    }


def compute_entity_level_metrics(
    predictions: List[List[Tuple[int, int, str]]],
    references: List[List[Tuple[int, int, str]]]
) -> Dict[str, float]:
    """
    Compute entity-level metrics (exact span matching).
    
    Args:
        predictions: List of [(start, end, label), ...] for each sample
        references: List of [(start, end, label), ...] for each sample
    
    Returns:
        Dict with entity-level precision, recall, f1
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred_entities, ref_entities in zip(predictions, references):
        pred_set = set(pred_entities)
        ref_set = set(ref_entities)
        
        tp = len(pred_set & ref_set)
        fp = len(pred_set - ref_set)
        fn = len(ref_set - pred_set)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1
    }


# ─────────────────────────────────────────────
# Decoder Metrics (Instruction Following)
# ─────────────────────────────────────────────

def compute_fact_adherence(
    outputs: List[str],
    encoder_facts: List[List[str]]
) -> Dict[str, float]:
    """
    Compute fact adherence: how often decoder output mentions encoder facts.
    
    Args:
        outputs: List of decoder outputs
        encoder_facts: List of fact lists (sections, entities) per sample
    
    Returns:
        Dict with fact_adherence score
    """
    adherent_count = 0
    total_with_facts = 0
    
    for output, facts in zip(outputs, encoder_facts):
        if not facts:
            continue
        
        total_with_facts += 1
        output_lower = output.lower()
        
        # Check if any fact is mentioned
        for fact in facts:
            if fact.lower() in output_lower:
                adherent_count += 1
                break
    
    adherence = adherent_count / total_with_facts if total_with_facts > 0 else 0.0
    
    return {
        "fact_adherence": adherence,
        "adherent_samples": adherent_count,
        "total_samples_with_facts": total_with_facts
    }


def compute_refusal_correctness(
    outputs: List[str],
    should_refuse: List[bool]
) -> Dict[str, float]:
    """
    Compute refusal correctness metrics.
    
    Args:
        outputs: List of decoder outputs
        should_refuse: List of booleans indicating if refusal was expected
    
    Returns:
        Dict with refusal metrics
    """
    true_refusals = 0  # Correctly refused
    false_refusals = 0  # Refused when shouldn't have
    missed_refusals = 0  # Didn't refuse when should have
    correct_answers = 0  # Correctly answered
    
    for output, expected_refusal in zip(outputs, should_refuse):
        is_refusal = output.upper().startswith("REFUSE")
        
        if expected_refusal and is_refusal:
            true_refusals += 1
        elif expected_refusal and not is_refusal:
            missed_refusals += 1
        elif not expected_refusal and is_refusal:
            false_refusals += 1
        else:
            correct_answers += 1
    
    total = len(outputs)
    
    # Refusal precision: true_refusals / (true_refusals + false_refusals)
    refusal_precision = true_refusals / (true_refusals + false_refusals) if (true_refusals + false_refusals) > 0 else 1.0
    
    # Refusal recall: true_refusals / (true_refusals + missed_refusals)
    refusal_recall = true_refusals / (true_refusals + missed_refusals) if (true_refusals + missed_refusals) > 0 else 1.0
    
    # Overall accuracy
    accuracy = (true_refusals + correct_answers) / total if total > 0 else 0.0
    
    return {
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "refusal_f1": 2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall) if (refusal_precision + refusal_recall) > 0 else 0.0,
        "overall_accuracy": accuracy,
        "true_refusals": true_refusals,
        "false_refusals": false_refusals,
        "missed_refusals": missed_refusals,
        "correct_answers": correct_answers
    }


def compute_decoder_metrics(
    outputs: List[str],
    encoder_facts: List[List[str]],
    should_refuse: List[bool]
) -> Dict[str, float]:
    """
    Compute all decoder metrics.
    
    Args:
        outputs: List of decoder outputs
        encoder_facts: List of fact lists per sample
        should_refuse: List of expected refusal flags
    
    Returns:
        Combined metrics dict
    """
    fact_metrics = compute_fact_adherence(outputs, encoder_facts)
    refusal_metrics = compute_refusal_correctness(outputs, should_refuse)
    
    return {**fact_metrics, **refusal_metrics}


# ─────────────────────────────────────────────
# HuggingFace Trainer Compatible Metrics
# ─────────────────────────────────────────────

def create_encoder_compute_metrics(label_list: List[str]):
    """
    Create a compute_metrics function for HuggingFace Trainer.
    
    Args:
        label_list: List of valid labels
    
    Returns:
        Function compatible with Trainer.compute_metrics
    """
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Convert to label strings
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        return compute_encoder_metrics(true_predictions, true_labels, label_list)
    
    return compute_metrics
