"""RLHF Trainer Skeleton - NO TRAINING EXECUTED"""
import logging
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class RLHFTrainer:
    """Reinforcement Learning from Human Feedback trainer skeleton"""
    
    def __init__(self, model_name: str = "rl_trained", output_dir: str = "checkpoints/rlhf"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reward_model = None
        logger.info("RLHFTrainer initialized (SKELETON ONLY)")
    
    def train_reward_model(self, feedback_data: List[Dict]) -> Dict[str, Any]:
        """Train reward model from human feedback"""
        logger.warning("⚠️ RLHF TRAINING IS DISABLED - SETUP MODE ONLY")
        raise RuntimeError("RLHF training is disabled. System is in SETUP MODE.")
    
    def ppo_step(self, prompts: List[str], responses: List[str]) -> Dict[str, float]:
        """PPO training step"""
        logger.info("[SKELETON] Would perform PPO step")
        raise NotImplementedError("PPO disabled in SETUP MODE")
    
    def get_training_status(self) -> Dict[str, Any]:
        return {
            "status": "not_started",
            "mode": "SETUP_ONLY",
            "model": self.model_name,
            "message": "RLHF training is disabled in setup mode"
        }
