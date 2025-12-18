"""
Phase 3.5: Decoder Trainer

Supervised Fine-Tuning (SFT) for instruction-following decoder.
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
class DecoderTrainingConfig:
    """Configuration for decoder training."""
    base_model: str = ""  # MUST be set from config - no default model
    output_dir: str = "./outputs/decoder_sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    # LoRA config
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj"
    ])
    
    # Refusal config
    refusal_enabled: bool = True
    refusal_token: str = "REFUSE:"
    refusal_samples_ratio: float = 0.2
    
    # Prompt template
    prompt_template: str = """You are a legal assistant. Use ONLY the provided facts to answer.
If facts are insufficient, respond with "REFUSE: Missing facts."

ENCODER_FACTS:
{encoder_facts}

QUESTION:
{question}

ANSWER:"""
    
    @classmethod
    def from_yaml(cls, path: str) -> "DecoderTrainingConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})
        data_cfg = config.get("data", {})
        refusal_cfg = config.get("refusal", {})
        peft_cfg = model_cfg.get("peft_config", {})
        
        default_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        default_prompt = """You are a legal assistant. Use ONLY the provided facts to answer.
If facts are insufficient, respond with "REFUSE: Missing facts."

ENCODER_FACTS:
{encoder_facts}

QUESTION:
{question}

ANSWER:"""
        
        base_model = model_cfg.get("base_model", "")
        if not base_model:
            raise ValueError("base_model must be specified in config - no default allowed")
        
        return cls(
            base_model=base_model,
            output_dir=training_cfg.get("output_dir", "./outputs/decoder_sft"),
            num_train_epochs=training_cfg.get("num_train_epochs", 3),
            per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 8),
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
            learning_rate=training_cfg.get("learning_rate", 2e-4),
            weight_decay=training_cfg.get("weight_decay", 0.01),
            warmup_ratio=training_cfg.get("warmup_ratio", 0.03),
            max_seq_length=data_cfg.get("max_seq_length", 2048),
            fp16=training_cfg.get("fp16", True),
            gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
            use_peft=model_cfg.get("use_peft", True),
            lora_r=peft_cfg.get("r", 16),
            lora_alpha=peft_cfg.get("lora_alpha", 32),
            lora_dropout=peft_cfg.get("lora_dropout", 0.05),
            lora_target_modules=peft_cfg.get("target_modules", default_target_modules),
            refusal_enabled=refusal_cfg.get("enabled", True),
            refusal_token=refusal_cfg.get("refusal_token", "REFUSE:"),
            refusal_samples_ratio=refusal_cfg.get("refusal_samples_ratio", 0.2),
            prompt_template=data_cfg.get("prompt_template", default_prompt)
        )


class DecoderDataset:
    """
    Dataset for decoder (SFT) training.
    
    Expected JSONL format:
    {
      "prompt": "ENCODER_FACTS:\n...\nQUESTION:\n...",
      "response": "...",
      "refusal_allowed": true
    }
    """
    
    def __init__(self, file_path: str, tokenizer, config: DecoderTrainingConfig):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.config = config
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
        required = {"prompt", "response"}
        return required.issubset(sample.keys())
    
    def _format_sample(self, sample: Dict[str, Any]) -> str:
        """Format sample as instruction-response pair."""
        prompt = sample["prompt"]
        response = sample["response"]
        
        # Combine prompt and response for causal LM training
        full_text = f"{prompt}\n{response}"
        return full_text
    
    def _tokenize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize a single sample."""
        full_text = self._format_sample(sample)
        
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding="max_length",
            return_tensors=None
        )
        
        # For causal LM, labels = input_ids (shifted internally by model)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return self._tokenize_sample(sample)
    
    def get_refusal_stats(self) -> Dict[str, Any]:
        """Get statistics about refusal samples."""
        total = len(self.samples)
        refusals = sum(
            1 for s in self.samples
            if s.get("response", "").upper().startswith(self.config.refusal_token.upper())
        )
        
        return {
            "total_samples": total,
            "refusal_samples": refusals,
            "refusal_ratio": refusals / total if total > 0 else 0.0,
            "target_ratio": self.config.refusal_samples_ratio
        }


class DecoderTrainer:
    """
    Decoder trainer using SFT with optional LoRA.
    
    DO NOT CALL train() - This is infrastructure setup only.
    """
    
    def __init__(self, config: DecoderTrainingConfig):
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
            
            logger.info(f"Decoder trainer configured for: {self.config.base_model}")
            logger.info(f"LoRA enabled: {self.config.use_peft}")
            logger.info(f"Refusal training enabled: {self.config.refusal_enabled}")
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
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "warmup_ratio": self.config.warmup_ratio,
            "fp16": self.config.fp16,
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss"
        }
    
    def get_peft_config(self) -> Dict[str, Any]:
        """
        Get LoRA/PEFT configuration as dict.
        
        Does NOT create actual PeftConfig object.
        """
        if not self.config.use_peft:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "target_modules": self.config.lora_target_modules,
            "task_type": "CAUSAL_LM"
        }
    
    def validate_ready_to_train(self) -> Dict[str, Any]:
        """
        Validate that all prerequisites for training are met.
        
        Returns validation results.
        """
        results = {
            "config_valid": self._initialized,
            "model_specified": bool(self.config.base_model),
            "output_dir_specified": bool(self.config.output_dir),
            "peft_configured": self.config.use_peft,
            "refusal_configured": self.config.refusal_enabled,
            "ready": False
        }
        
        results["ready"] = all([
            results["config_valid"],
            results["model_specified"],
            results["output_dir_specified"]
        ])
        
        return results
    
    def get_prompt_template(self) -> str:
        """Get the prompt template for training."""
        return self.config.prompt_template


def create_decoder_trainer(config_path: str) -> DecoderTrainer:
    """
    Factory function to create a DecoderTrainer from config file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configured DecoderTrainer instance
    """
    config = DecoderTrainingConfig.from_yaml(config_path)
    trainer = DecoderTrainer(config)
    trainer.setup()
    return trainer


if __name__ == "__main__":
    # Validation only - DO NOT TRAIN
    print("=" * 50)
    print("Decoder Trainer Infrastructure Validation")
    print("=" * 50)
    
    config_path = "src/training/configs/decoder_sft.yaml"
    
    if not Path(config_path).exists():
        print(f"Config not found: {config_path}")
        exit(1)
    
    trainer = create_decoder_trainer(config_path)
    
    print("\nTraining Args:")
    for k, v in trainer.get_training_args().items():
        print(f"  {k}: {v}")
    
    print("\nPEFT Config:")
    for k, v in trainer.get_peft_config().items():
        print(f"  {k}: {v}")
    
    print("\nPrompt Template:")
    print(trainer.get_prompt_template()[:200] + "...")
    
    print("\nValidation:")
    validation = trainer.validate_ready_to_train()
    for k, v in validation.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 50)
    if validation["ready"]:
        print("DECODER TRAINER READY (infrastructure only)")
    else:
        print("DECODER TRAINER NOT READY")
    print("=" * 50)
