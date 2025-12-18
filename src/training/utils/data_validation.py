"""
Phase 3.5: Data Validation Utilities

Validates training data contracts for encoder and decoder.
Ensures refusal-aware data is properly formatted.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of data validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]


# ─────────────────────────────────────────────
# Encoder Data Contract
# ─────────────────────────────────────────────
ENCODER_REQUIRED_FIELDS = {"text", "entities", "task"}
ENCODER_ENTITY_FIELDS = {"start", "end", "label"}
ENCODER_VALID_TASKS = {"ner", "classification"}
ENCODER_VALID_LABELS = {
    "SECTION", "ACT", "PARTY", "DATE", "COURT", "JUDGE",
    "OFFENSE", "PENALTY", "CITATION", "MISC"
}


def validate_encoder_sample(sample: Dict[str, Any], line_num: int) -> Tuple[bool, List[str]]:
    """
    Validate a single encoder training sample.
    
    Expected format:
    {
      "text": "...",
      "entities": [{"start": 0, "end": 10, "label": "SECTION"}],
      "task": "ner"
    }
    """
    errors = []
    
    # Check required fields
    missing = ENCODER_REQUIRED_FIELDS - set(sample.keys())
    if missing:
        errors.append(f"Line {line_num}: Missing required fields: {missing}")
        return False, errors
    
    # Validate text
    if not isinstance(sample["text"], str) or len(sample["text"]) < 10:
        errors.append(f"Line {line_num}: 'text' must be a non-empty string (min 10 chars)")
    
    # Validate task
    if sample["task"] not in ENCODER_VALID_TASKS:
        errors.append(f"Line {line_num}: Invalid task '{sample['task']}'. Must be one of {ENCODER_VALID_TASKS}")
    
    # Validate entities
    if not isinstance(sample["entities"], list):
        errors.append(f"Line {line_num}: 'entities' must be a list")
    else:
        for i, entity in enumerate(sample["entities"]):
            if not isinstance(entity, dict):
                errors.append(f"Line {line_num}, entity {i}: Must be a dict")
                continue
            
            missing_ent = ENCODER_ENTITY_FIELDS - set(entity.keys())
            if missing_ent:
                errors.append(f"Line {line_num}, entity {i}: Missing fields: {missing_ent}")
                continue
            
            if not isinstance(entity["start"], int) or not isinstance(entity["end"], int):
                errors.append(f"Line {line_num}, entity {i}: 'start' and 'end' must be integers")
            elif entity["start"] >= entity["end"]:
                errors.append(f"Line {line_num}, entity {i}: 'start' must be < 'end'")
            elif entity["end"] > len(sample["text"]):
                errors.append(f"Line {line_num}, entity {i}: 'end' exceeds text length")
    
    return len(errors) == 0, errors


def validate_encoder_dataset(file_path: str) -> ValidationResult:
    """Validate an entire encoder dataset file."""
    path = Path(file_path)
    errors = []
    warnings = []
    stats = {
        "total_samples": 0,
        "valid_samples": 0,
        "total_entities": 0,
        "label_distribution": {},
        "task_distribution": {}
    }
    
    if not path.exists():
        return ValidationResult(False, [f"File not found: {file_path}"], [], stats)
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                continue
            
            stats["total_samples"] += 1
            valid, sample_errors = validate_encoder_sample(sample, line_num)
            
            if valid:
                stats["valid_samples"] += 1
                stats["total_entities"] += len(sample.get("entities", []))
                
                # Track distributions
                task = sample.get("task", "unknown")
                stats["task_distribution"][task] = stats["task_distribution"].get(task, 0) + 1
                
                for entity in sample.get("entities", []):
                    label = entity.get("label", "unknown")
                    stats["label_distribution"][label] = stats["label_distribution"].get(label, 0) + 1
            else:
                errors.extend(sample_errors)
    
    # Warnings
    if stats["total_samples"] < 100:
        warnings.append(f"Dataset has only {stats['total_samples']} samples. Recommend at least 1000.")
    
    if stats["total_entities"] == 0:
        warnings.append("No entities found in dataset.")
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors[:50],  # Limit errors shown
        warnings=warnings,
        stats=stats
    )


# ─────────────────────────────────────────────
# Decoder Data Contract
# ─────────────────────────────────────────────
DECODER_REQUIRED_FIELDS = {"prompt", "response"}
DECODER_OPTIONAL_FIELDS = {"refusal_allowed", "encoder_facts", "metadata"}


def validate_decoder_sample(sample: Dict[str, Any], line_num: int) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a single decoder training sample.
    
    Expected format:
    {
      "prompt": "ENCODER_FACTS:\n...\nQUESTION:\n...",
      "response": "...",
      "refusal_allowed": true
    }
    """
    errors = []
    warnings = []
    
    # Check required fields
    missing = DECODER_REQUIRED_FIELDS - set(sample.keys())
    if missing:
        errors.append(f"Line {line_num}: Missing required fields: {missing}")
        return False, errors, warnings
    
    # Validate prompt
    if not isinstance(sample["prompt"], str) or len(sample["prompt"]) < 20:
        errors.append(f"Line {line_num}: 'prompt' must be a non-empty string (min 20 chars)")
    
    # Check for encoder facts in prompt
    if "ENCODER_FACTS" not in sample["prompt"] and "encoder_facts" not in sample.get("metadata", {}):
        warnings.append(f"Line {line_num}: Prompt does not contain 'ENCODER_FACTS' marker")
    
    # Validate response
    if not isinstance(sample["response"], str):
        errors.append(f"Line {line_num}: 'response' must be a string")
    elif len(sample["response"]) == 0:
        errors.append(f"Line {line_num}: 'response' cannot be empty")
    
    # Validate refusal_allowed
    if "refusal_allowed" in sample:
        if not isinstance(sample["refusal_allowed"], bool):
            errors.append(f"Line {line_num}: 'refusal_allowed' must be a boolean")
    
    # Check refusal consistency
    is_refusal = sample.get("response", "").upper().startswith("REFUSE")
    refusal_allowed = sample.get("refusal_allowed", True)
    
    if is_refusal and not refusal_allowed:
        warnings.append(f"Line {line_num}: Response is a refusal but 'refusal_allowed' is False")
    
    return len(errors) == 0, errors, warnings


def validate_decoder_dataset(file_path: str) -> ValidationResult:
    """Validate an entire decoder dataset file."""
    path = Path(file_path)
    errors = []
    warnings = []
    stats = {
        "total_samples": 0,
        "valid_samples": 0,
        "refusal_samples": 0,
        "non_refusal_samples": 0,
        "avg_prompt_length": 0,
        "avg_response_length": 0
    }
    
    if not path.exists():
        return ValidationResult(False, [f"File not found: {file_path}"], [], stats)
    
    prompt_lengths = []
    response_lengths = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                continue
            
            stats["total_samples"] += 1
            valid, sample_errors, sample_warnings = validate_decoder_sample(sample, line_num)
            
            if valid:
                stats["valid_samples"] += 1
                prompt_lengths.append(len(sample.get("prompt", "")))
                response_lengths.append(len(sample.get("response", "")))
                
                # Track refusals
                if sample.get("response", "").upper().startswith("REFUSE"):
                    stats["refusal_samples"] += 1
                else:
                    stats["non_refusal_samples"] += 1
            
            errors.extend(sample_errors)
            warnings.extend(sample_warnings)
    
    # Compute averages
    if prompt_lengths:
        stats["avg_prompt_length"] = sum(prompt_lengths) / len(prompt_lengths)
    if response_lengths:
        stats["avg_response_length"] = sum(response_lengths) / len(response_lengths)
    
    # Warnings
    if stats["total_samples"] < 100:
        warnings.append(f"Dataset has only {stats['total_samples']} samples. Recommend at least 1000.")
    
    if stats["refusal_samples"] == 0:
        warnings.append("No refusal samples found. Recommend including refusal examples.")
    
    refusal_ratio = stats["refusal_samples"] / max(stats["total_samples"], 1)
    if refusal_ratio < 0.1:
        warnings.append(f"Refusal ratio is {refusal_ratio:.1%}. Recommend at least 10-20%.")
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors[:50],
        warnings=warnings[:20],
        stats=stats
    )


def validate_all_datasets(base_path: str = "src/training/datasets") -> Dict[str, ValidationResult]:
    """Validate all training datasets."""
    results = {}
    
    encoder_train = Path(base_path) / "encoder" / "train.jsonl"
    encoder_eval = Path(base_path) / "encoder" / "eval.jsonl"
    decoder_train = Path(base_path) / "decoder" / "train.jsonl"
    decoder_eval = Path(base_path) / "decoder" / "eval.jsonl"
    
    if encoder_train.exists():
        results["encoder_train"] = validate_encoder_dataset(str(encoder_train))
    if encoder_eval.exists():
        results["encoder_eval"] = validate_encoder_dataset(str(encoder_eval))
    if decoder_train.exists():
        results["decoder_train"] = validate_decoder_dataset(str(decoder_train))
    if decoder_eval.exists():
        results["decoder_eval"] = validate_decoder_dataset(str(decoder_eval))
    
    return results


if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("Training Data Validation")
    print("=" * 50)
    
    results = validate_all_datasets()
    
    if not results:
        print("\nNo datasets found. Create datasets in:")
        print("  - src/training/datasets/encoder/train.jsonl")
        print("  - src/training/datasets/encoder/eval.jsonl")
        print("  - src/training/datasets/decoder/train.jsonl")
        print("  - src/training/datasets/decoder/eval.jsonl")
        sys.exit(0)
    
    all_valid = True
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Valid: {result.valid}")
        print(f"  Stats: {result.stats}")
        if result.errors:
            print(f"  Errors: {result.errors[:5]}")
        if result.warnings:
            print(f"  Warnings: {result.warnings[:5]}")
        if not result.valid:
            all_valid = False
    
    sys.exit(0 if all_valid else 1)
