"""
Phase 3.5: Encoder Trainer

Token Classification (NER) training using HuggingFace Trainer.
DO NOT CALL trainer.train() - Infrastructure setup only.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger(__name__)


@dataclass
class EncoderTrainingConfig:
    """Configuration for encoder training."""
    base_model: str = ""  # MUST be set from config - no default model
    output_dir: str = "./outputs/encoder_sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    label_all_tokens: bool = False
    labels: List[str] = field(default_factory=lambda: [
        "O", "B-SECTION", "I-SECTION", "B-ACT", "I-ACT",
        "B-PARTY", "I-PARTY", "B-DATE", "I-DATE"
    ])
    
    @classmethod
    def from_yaml(cls, path: str) -> "EncoderTrainingConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        
        default_labels = [
            "O", "B-SECTION", "I-SECTION", "B-ACT", "I-ACT",
            "B-PARTY", "I-PARTY", "B-DATE", "I-DATE"
        ]
        
        base_model = config.get("model", {}).get("base_model", "")
        if not base_model:
            raise ValueError("base_model must be specified in config - no default allowed")
        
        return cls(
            base_model=base_model,
            output_dir=config.get("training", {}).get("output_dir", "./outputs/encoder_sft"),
            num_train_epochs=config.get("training", {}).get("num_train_epochs", 3),
            per_device_train_batch_size=config.get("training", {}).get("per_device_train_batch_size", 16),
            per_device_eval_batch_size=config.get("training", {}).get("per_device_eval_batch_size", 32),
            learning_rate=config.get("training", {}).get("learning_rate", 2e-5),
            weight_decay=config.get("training", {}).get("weight_decay", 0.01),
            warmup_ratio=config.get("training", {}).get("warmup_ratio", 0.1),
            max_seq_length=config.get("data", {}).get("max_seq_length", 512),
            label_all_tokens=config.get("data", {}).get("label_all_tokens", False),
            labels=config.get("labels", default_labels)
        )


class EncoderDataset:
    """
    Dataset for encoder (NER) training.
    
    Expected JSONL format:
    {
      "text": "...",
      "entities": [{"start": 0, "end": 10, "label": "SECTION"}],
      "task": "ner"
    }
    """
    
    def __init__(self, file_path: str, tokenizer, config: EncoderTrainingConfig):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.config = config
        self.label2id = {label: i for i, label in enumerate(config.labels)}
        self.id2label = {i: label for i, label in enumerate(config.labels)}
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load and validate samples from JSONL file."""
        samples = []
        path = Path(self.file_path)
        
        if not path.exists():
            logger.warning(f"Dataset file not found: {self.file_path}")
            return samples
        
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    if self._validate_sample(sample):
                        samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Invalid JSON - {e}")
        
        logger.info(f"Loaded {len(samples)} samples from {self.file_path}")
        return samples
    
    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate a single sample."""
        required = {"text", "entities", "task"}
        return required.issubset(sample.keys())
    
    def _align_labels_with_tokens(self, text: str, entities: List[Dict]) -> Dict[str, Any]:
        """
        Tokenize text and align entity labels with tokens.
        
        Returns tokenized inputs with aligned labels.
        """
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_offsets_mapping=True,
            padding="max_length"
        )
        
        # Initialize labels as "O" (outside)
        labels = [self.label2id["O"]] * len(tokenized["input_ids"])
        offset_mapping = tokenized["offset_mapping"]
        
        # Mark entity spans
        for entity in entities:
            start, end = entity["start"], entity["end"]
            label = entity["label"]
            
            b_label = f"B-{label}"
            i_label = f"I-{label}"
            
            if b_label not in self.label2id:
                continue
            
            first_token = True
            for idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start is None or token_end is None:
                    continue
                if token_start >= start and token_end <= end:
                    if first_token:
                        labels[idx] = self.label2id[b_label]
                        first_token = False
                    else:
                        labels[idx] = self.label2id[i_label]
        
        # Set special tokens to -100 (ignored in loss)
        for idx, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == 0 and token_end == 0:
                labels[idx] = -100
        
        tokenized["labels"] = labels
        del tokenized["offset_mapping"]
        
        return tokenized
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return self._align_labels_with_tokens(sample["text"], sample["entities"])


class EncoderTrainer:
    """
    Encoder trainer using HuggingFace Trainer.
    
    DO NOT CALL train() - This is infrastructure setup only.
    """
    
    def __init__(self, config: EncoderTrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._initialized = False
    
    def setup(self) -> bool:
        """
        Setup model, tokenizer, and trainer.
        
        Returns True if setup successful, False otherwise.
        Does NOT download weights - just validates config.
        """
        try:
            # Validate config
            if not self.config.base_model:
                raise ValueError("base_model is required")
            
            if not self.config.labels:
                raise ValueError("labels list is required")
            
            logger.info(f"Encoder trainer configured for: {self.config.base_model}")
            logger.info(f"Labels: {self.config.labels}")
            logger.info(f"Output dir: {self.config.output_dir}")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False
    
    def prepare_datasets(
        self,
        train_file: str,
        eval_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare training and evaluation datasets.
        
        Returns dataset info without loading actual data.
        """
        info = {
            "train_file": train_file,
            "eval_file": eval_file,
            "train_exists": Path(train_file).exists(),
            "eval_exists": Path(eval_file).exists() if eval_file else False
        }
        
        logger.info(f"Dataset info: {info}")
        return info
    
    def get_training_args(self) -> Dict[str, Any]:
        """
        Get HuggingFace TrainingArguments as dict.
        
        Does NOT create actual TrainingArguments object.
        """
        return {
            "output_dir": self.config.output_dir,
            "num_train_epochs": self.config.num_train_epochs,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "per_device_eval_batch_size": self.config.per_device_eval_batch_size,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "warmup_ratio": self.config.warmup_ratio,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1"
        }
    
    def validate_ready_to_train(self) -> Dict[str, Any]:
        """
        Validate that all prerequisites for training are met.
        
        Returns validation results.
        """
        results = {
            "config_valid": self._initialized,
            "model_specified": bool(self.config.base_model),
            "labels_defined": len(self.config.labels) > 0,
            "output_dir_specified": bool(self.config.output_dir),
            "ready": False
        }
        
        results["ready"] = all([
            results["config_valid"],
            results["model_specified"],
            results["labels_defined"],
            results["output_dir_specified"]
        ])
        
        return results


def create_encoder_trainer(config_path: str) -> EncoderTrainer:
    """
    Factory function to create an EncoderTrainer from config file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configured EncoderTrainer instance
    """
    config = EncoderTrainingConfig.from_yaml(config_path)
    trainer = EncoderTrainer(config)
    trainer.setup()
    return trainer


if __name__ == "__main__":
    # Validation only - DO NOT TRAIN
    print("=" * 50)
    print("Encoder Trainer Infrastructure Validation")
    print("=" * 50)
    
    config_path = "src/training/configs/encoder_sft.yaml"
    
    if not Path(config_path).exists():
        print(f"Config not found: {config_path}")
        exit(1)
    
    trainer = create_encoder_trainer(config_path)
    
    print("\nTraining Args:")
    for k, v in trainer.get_training_args().items():
        print(f"  {k}: {v}")
    
    print("\nValidation:")
    validation = trainer.validate_ready_to_train()
    for k, v in validation.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 50)
    if validation["ready"]:
        print("ENCODER TRAINER READY (infrastructure only)")
    else:
        print("ENCODER TRAINER NOT READY")
    print("=" * 50)
