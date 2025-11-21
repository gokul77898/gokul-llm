"""Configuration loader for training pipelines"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration"""
    vocab_size: int = 5000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    max_length: int = 512
    dropout: float = 0.1
    base_model: Optional[str] = None
    task_type: Optional[str] = None
    num_labels: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None
    normalize_embeddings: Optional[bool] = None
    hidden_dim: Optional[int] = None
    action_space: Optional[int] = None
    policy_model: Optional[str] = None
    freeze_base: Optional[bool] = None
    input_dim: Optional[int] = None
    obs_dim: Optional[int] = None
    action_dim: Optional[int] = None
    num_layers: Optional[int] = None


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    warmup_steps: int = 500
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50
    mixed_precision: bool = True
    num_workers: int = 4
    total_timesteps: Optional[int] = None
    n_steps: Optional[int] = None
    n_epochs: Optional[int] = None
    gamma: Optional[float] = None
    gae_lambda: Optional[float] = None
    clip_range: Optional[float] = None
    vf_coef: Optional[float] = None
    ent_coef: Optional[float] = None
    beta: Optional[float] = None


@dataclass
class DataConfig:
    """Data configuration"""
    train_file: str = "data/train.txt"
    val_file: str = "data/val.txt"
    max_samples: int = 1000
    min_doc_length: int = 100
    max_length: int = 512
    documents_file: Optional[str] = None
    max_documents: Optional[int] = None


@dataclass
class PathConfig:
    """Path configuration"""
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    vocab_file: Optional[str] = None
    index_path: Optional[str] = None
    metadata_path: Optional[str] = None
    embeddings_cache: Optional[str] = None


@dataclass
class SystemConfig:
    """System configuration"""
    device: str = "cuda"
    seed: int = 42
    use_wandb: bool = False
    wandb_project: str = "mark"
    num_workers: int = 4


@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    max_episode_steps: int = 50
    vocab_size: int = 1000
    task_type: str = "summarization"
    max_length: Optional[int] = None


@dataclass
class Config:
    """Main configuration container"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    environment: Optional[EnvironmentConfig] = None


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config objects from dicts
    model_cfg = ModelConfig(**config_dict.get('model', {}))
    training_cfg = TrainingConfig(**config_dict.get('training', {}))
    data_cfg = DataConfig(**config_dict.get('data', {}))
    paths_cfg = PathConfig(**config_dict.get('paths', {}))
    system_cfg = SystemConfig(**config_dict.get('system', {}))
    env_cfg = EnvironmentConfig(**config_dict.get('environment', {})) if 'environment' in config_dict else None
    
    return Config(
        model=model_cfg,
        training=training_cfg,
        data=data_cfg,
        paths=paths_cfg,
        system=system_cfg,
        environment=env_cfg
    )


def save_config(config: Config, path: str):
    """Save configuration to YAML file"""
    config_dict = {
        'model': config.model.__dict__,
        'training': config.training.__dict__,
        'data': config.data.__dict__,
        'paths': config.paths.__dict__,
        'system': config.system.__dict__,
    }
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
