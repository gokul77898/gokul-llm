"""Checkpoint utilities for training pipelines"""

import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    state: Dict[str, Any],
    path: str,
    is_best: bool = False
):
    """
    Save checkpoint atomically
    
    Args:
        state: Dictionary containing model, optimizer, scheduler, epoch, etc.
        path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to temporary file first
    temp_path = path.parent / f"{path.name}.tmp"
    torch.save(state, temp_path)
    
    # Atomic rename
    shutil.move(str(temp_path), str(path))
    logger.info(f"Checkpoint saved to {path}")
    
    # Save best model
    if is_best:
        best_path = path.parent / "best_model.pt"
        shutil.copy(str(path), str(best_path))
        logger.info(f"Best model saved to {best_path}")


def load_checkpoint(
    path: str,
    device: torch.device = torch.device('cpu')
) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint from file
    
    Args:
        path: Path to checkpoint file
        device: Device to load checkpoint to
        
    Returns:
        Dictionary containing model, optimizer, scheduler, epoch, etc.
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"Checkpoint not found: {path}")
        return None
    
    try:
        checkpoint = torch.load(path, map_location=device)
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return None


def resume_from_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Resume training from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state dict into
        optimizer: Optimizer to load state dict into
        scheduler: Scheduler to load state dict into
        device: Device to load checkpoint to
        
    Returns:
        Dictionary with resume information (epoch, best_metric, etc.)
    """
    checkpoint = load_checkpoint(checkpoint_path, device)
    
    if checkpoint is None:
        logger.info("No checkpoint found, starting from scratch")
        return {'epoch': 0, 'best_metric': float('inf'), 'global_step': 0}
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Model state loaded")
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Optimizer state loaded")
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("Scheduler state loaded")
    
    # Load RNG states for reproducibility
    if 'rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['rng_state'])
    if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    
    resume_info = {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'best_metric': checkpoint.get('best_metric', float('inf')),
    }
    
    logger.info(f"Resuming from epoch {resume_info['epoch']}")
    return resume_info


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in directory"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        return None
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return str(checkpoints[-1])
