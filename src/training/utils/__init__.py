"""
Phase 3.5: Training Utilities

Data validation and metrics utilities.
"""

from .data_validation import (
    validate_encoder_sample,
    validate_encoder_dataset,
    validate_decoder_sample,
    validate_decoder_dataset,
    validate_all_datasets,
    ValidationResult
)

from .metrics import (
    compute_encoder_metrics,
    compute_entity_level_metrics,
    compute_fact_adherence,
    compute_refusal_correctness,
    compute_decoder_metrics,
    create_encoder_compute_metrics
)

__all__ = [
    "validate_encoder_sample",
    "validate_encoder_dataset",
    "validate_decoder_sample",
    "validate_decoder_dataset",
    "validate_all_datasets",
    "ValidationResult",
    "compute_encoder_metrics",
    "compute_entity_level_metrics",
    "compute_fact_adherence",
    "compute_refusal_correctness",
    "compute_decoder_metrics",
    "create_encoder_compute_metrics"
]
