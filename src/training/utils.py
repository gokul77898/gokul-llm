"""
Training utilities for data loading and processing.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class TrainingExample:
    """Training example for SFT."""
    instruction: str
    input: str
    output: str
    
    def to_prompt(self) -> str:
        """Convert to training prompt format."""
        return f"{self.instruction}\n\n{self.input}\n\nAnswer: {self.output}"


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    
    def __init__(self, jsonl_path: str, tokenizer: Any, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            jsonl_path: Path to JSONL file
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.examples = self._load_jsonl(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def _load_jsonl(self, path: str) -> List[TrainingExample]:
        """Load examples from JSONL."""
        examples = []
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                examples.append(TrainingExample(
                    instruction=data['instruction'],
                    input=data['input'],
                    output=data['output']
                ))
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized example."""
        example = self.examples[idx]
        prompt = example.to_prompt()
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Config dictionary
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def print_trainable_parameters(model):
    """
    Print number of trainable parameters in model.
    
    Args:
        model: Model instance
    """
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"📊 Model Parameters:")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Total: {all_params:,}")
    print(f"   Trainable %: {100 * trainable_params / all_params:.2f}%")


def get_device(config_device: str = "auto") -> str:
    """
    Get appropriate device for training.
    
    Args:
        config_device: Device from config (auto, cpu, cuda, mps)
        
    Returns:
        Device string
    """
    if config_device != "auto":
        return config_device
    
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
