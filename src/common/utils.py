"""Common utilities for training pipelines"""

import os
import random
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


def seed_everything(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "cuda") -> torch.device:
    """Get torch device based on availability"""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str.startswith("cuda:"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        else:
            logging.warning(f"CUDA not available, falling back to CPU")
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def init_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """Initialize logger with console and optional file output"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


@contextmanager
def autocast_if_enabled(enabled: bool, device_type: str = "cuda"):
    """Context manager for automatic mixed precision"""
    if enabled and device_type == "cuda":
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


class GradScalerWrapper:
    """Wrapper for GradScaler that works with and without mixed precision"""
    
    def __init__(self, enabled: bool):
        self.enabled = enabled
        if enabled:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def scale(self, loss):
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer):
        if self.enabled:
            self.scaler.step(optimizer)
        else:
            optimizer.step()
    
    def update(self):
        if self.enabled:
            self.scaler.update()
    
    def unscale_(self, optimizer):
        if self.enabled:
            self.scaler.unscale_(optimizer)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds to human readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
