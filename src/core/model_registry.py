"""
Unified Model Registry for MARK MoE System - Local-Only Inference

All model inference is performed locally.
No remote API calls. No HF Inference API. No hosted models.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExpertInfo:
    """Information about a registered expert model"""
    name: str
    model_id: str
    task_types: List[str]
    token_window: int = 512
    tuning: str = "none"
    lora_config: Dict[str, Any] = field(default_factory=dict)
    priority_score: float = 1.0
    role: str = "encoder"  # encoder or decoder


class ModelRegistry:
    """Central registry for all MARK MoE experts"""
    
    def __init__(self, config_path: str = "configs/moe_experts.yaml"):
        self.experts: Dict[str, ExpertInfo] = {}
        self.config_path = config_path
        self._load_registry()
    
    def _load_registry(self):
        """Load experts from YAML config"""
        path = Path(self.config_path)
        if not path.exists():
            logger.warning(f"Config file {path} not found. Registry empty.")
            return

        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        for expert_cfg in config.get('experts', []):
            expert = ExpertInfo(
                name=expert_cfg['name'],
                model_id=expert_cfg['model_id'],
                task_types=expert_cfg['task_types'],
                token_window=expert_cfg.get('token_window', 512),
                tuning=expert_cfg.get('tuning', 'none'),
                lora_config=expert_cfg.get('lora', {}),
                priority_score=expert_cfg.get('priority_score', 1.0),
                role=expert_cfg.get('role', 'encoder')
            )
            self.experts[expert.name] = expert
            logger.info(f"Registered expert: {expert.name} ({expert.model_id}) [role={expert.role}]")

    def get_expert(self, name: str) -> Optional[ExpertInfo]:
        """Get info for a specific expert"""
        return self.experts.get(name)

    def list_experts(self) -> List[ExpertInfo]:
        """List all registered experts"""
        return list(self.experts.values())

    def get_experts_by_task(self, task: str) -> List[ExpertInfo]:
        """Find experts suitable for a specific task"""
        return [e for e in self.experts.values() if task in e.task_types]

    def get_encoders(self) -> List[ExpertInfo]:
        """Get all encoder experts"""
        return [e for e in self.experts.values() if e.role == "encoder"]

    def get_decoders(self) -> List[ExpertInfo]:
        """Get all decoder experts"""
        return [e for e in self.experts.values() if e.role == "decoder"]


# Global registry instance
_registry = None


def get_registry() -> ModelRegistry:
    """Get the global model registry (lazy init)"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
