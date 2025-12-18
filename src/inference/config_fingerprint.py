"""
Configuration Fingerprint Module

Phase P6: Configuration Integrity & Kill Switches

Provides:
- Immutable config hash computed at startup
- Feature flags (kill switches) for runtime control
- Fail-fast validation for missing/invalid configs

CONSTRAINTS:
- Config hash computed ONCE at startup
- Any config change MUST change the hash
- Feature flags checked at runtime (not cached)
- Missing configs → hard crash at startup
- Disabled components → forced refusal (not crash)
"""

import hashlib
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PHASE P6: FEATURE FLAGS (KILL SWITCHES)
# PHASE H0: CPU-ONLY DEFAULTS
# ─────────────────────────────────────────────

def get_feature_flags() -> Dict[str, bool]:
    """
    Get current feature flags from environment.
    
    Checked at runtime (not cached) to allow instant kill switches.
    
    Phase H0 Defaults (CPU-only):
    - ENABLE_ENCODER=false (requires GPU)
    - ENABLE_DECODER=false (requires GPU)
    - ENABLE_RAG=true (works on CPU)
    
    Returns:
        Dictionary with feature flag states
    """
    return {
        "encoder": os.getenv("ENABLE_ENCODER", "false").lower() == "true",
        "rag": os.getenv("ENABLE_RAG", "true").lower() == "true",
        "decoder": os.getenv("ENABLE_DECODER", "false").lower() == "true",
    }


def is_encoder_enabled() -> bool:
    """
    Check if encoder is enabled.
    
    Phase H0: Defaults to FALSE (requires GPU).
    """
    return os.getenv("ENABLE_ENCODER", "false").lower() == "true"


def is_rag_enabled() -> bool:
    """
    Check if RAG is enabled.
    
    Phase H0: Defaults to TRUE (works on CPU).
    """
    return os.getenv("ENABLE_RAG", "true").lower() == "true"


def is_decoder_enabled() -> bool:
    """
    Check if decoder is enabled.
    
    Phase H0: Defaults to FALSE (requires GPU).
    """
    return os.getenv("ENABLE_DECODER", "false").lower() == "true"


# ─────────────────────────────────────────────
# PHASE P8: CANARY MODE
# PHASE H1: Default to TRUE for HF deployment
# ─────────────────────────────────────────────

def is_canary_mode() -> bool:
    """
    Check if canary mode is enabled.
    
    Canary mode restrictions:
    - Only 1 concurrent request
    - max_new_tokens = 64
    - Warning in response + trace
    
    Phase H1: Defaults to TRUE for HF deployment safety.
    """
    return os.getenv("CANARY_MODE", "true").lower() == "true"


def get_canary_limits() -> Dict[str, Any]:
    """Get canary mode limits."""
    return {
        "max_concurrent_requests": 1,
        "max_new_tokens": 64,
    }


# ─────────────────────────────────────────────
# PHASE P10: OPS OVERRIDE
# ─────────────────────────────────────────────

def is_ops_override_allowed() -> bool:
    """Check if ops override is allowed."""
    return os.getenv("ALLOW_OPS_OVERRIDE", "false").lower() == "true"


# ─────────────────────────────────────────────
# PHASE P6: CONFIG HASHING
# ─────────────────────────────────────────────

@dataclass
class ConfigFingerprint:
    """Configuration fingerprint with hash and metadata."""
    config_hash: str
    config_files: Dict[str, str]  # filename -> hash
    constants: Dict[str, Any]
    computed_at: str


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    """
    Load YAML file with fail-fast validation.
    
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    import yaml
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        content = yaml.safe_load(f)
    
    if content is None:
        raise ValueError(f"Config file is empty: {path}")
    
    return content


def _canonical_serialize(data: Any) -> str:
    """
    Canonically serialize data for hashing.
    
    - Sorted keys
    - No whitespace
    - Deterministic output
    """
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


def _compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file's canonical content."""
    content = _load_yaml_file(path)
    canonical = _canonical_serialize(content)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]


def _get_safety_constants() -> Dict[str, Any]:
    """
    Get all safety-critical constants for hashing.
    
    These are imported from local_models to ensure consistency.
    """
    from src.inference.local_models import (
        MAX_NEW_TOKENS,
        MAX_INPUT_TOKENS,
        ENCODER_MIN_VRAM_GB,
        DECODER_MIN_VRAM_GB,
        DECODER_PREFERRED_VRAM_GB,
        ENCODER_TIMEOUT_SECONDS,
        DECODER_TIMEOUT_SECONDS,
    )
    
    # Import P3 constants from server
    try:
        from src.inference.server import (
            MAX_CONCURRENT_REQUESTS,
            TOTAL_REQUEST_BUDGET_MS,
        )
    except ImportError:
        # Server not yet imported - use defaults
        MAX_CONCURRENT_REQUESTS = 2
        TOTAL_REQUEST_BUDGET_MS = 30000
    
    return {
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "encoder_min_vram_gb": ENCODER_MIN_VRAM_GB,
        "decoder_min_vram_gb": DECODER_MIN_VRAM_GB,
        "decoder_preferred_vram_gb": DECODER_PREFERRED_VRAM_GB,
        "encoder_timeout_seconds": ENCODER_TIMEOUT_SECONDS,
        "decoder_timeout_seconds": DECODER_TIMEOUT_SECONDS,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "total_request_budget_ms": TOTAL_REQUEST_BUDGET_MS,
    }


def compute_config_fingerprint() -> ConfigFingerprint:
    """
    Compute configuration fingerprint.
    
    FAIL-FAST:
    - Missing config files → raises FileNotFoundError
    - Invalid YAML → raises yaml.YAMLError
    - Hash computation failure → raises exception
    
    Returns:
        ConfigFingerprint with hash and metadata
    """
    from datetime import datetime
    
    # Find project root
    project_root = Path(__file__).parent.parent.parent
    
    # Config files to hash
    config_files = {
        "moe_experts.yaml": project_root / "configs" / "moe_experts.yaml",
    }
    
    # Compute per-file hashes
    file_hashes = {}
    for name, path in config_files.items():
        file_hashes[name] = _compute_file_hash(path)
    
    # Get safety constants
    constants = _get_safety_constants()
    
    # Combine all data for final hash
    combined_data = {
        "files": file_hashes,
        "constants": constants,
    }
    
    canonical = _canonical_serialize(combined_data)
    config_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]
    
    return ConfigFingerprint(
        config_hash=config_hash,
        config_files=file_hashes,
        constants=constants,
        computed_at=datetime.utcnow().isoformat(),
    )


# ─────────────────────────────────────────────
# PHASE P6: GLOBAL CONFIG HASH (COMPUTED ONCE)
# ─────────────────────────────────────────────

# Computed at module import (startup)
_CONFIG_FINGERPRINT: Optional[ConfigFingerprint] = None


def _initialize_config() -> ConfigFingerprint:
    """
    Initialize config fingerprint at startup.
    
    FAIL-FAST: Crashes if config is invalid.
    """
    global _CONFIG_FINGERPRINT
    
    if _CONFIG_FINGERPRINT is None:
        logger.info("[CONFIG] Computing configuration fingerprint...")
        _CONFIG_FINGERPRINT = compute_config_fingerprint()
        logger.info(f"[CONFIG] Hash: {_CONFIG_FINGERPRINT.config_hash}")
    
    return _CONFIG_FINGERPRINT


def get_config_hash() -> str:
    """
    Get the configuration hash.
    
    Returns:
        16-character hex hash string
    """
    if _CONFIG_FINGERPRINT is None:
        _initialize_config()
    return _CONFIG_FINGERPRINT.config_hash


def get_config_fingerprint() -> ConfigFingerprint:
    """
    Get the full configuration fingerprint.
    
    Returns:
        ConfigFingerprint with hash and metadata
    """
    if _CONFIG_FINGERPRINT is None:
        _initialize_config()
    return _CONFIG_FINGERPRINT


def validate_config_at_startup() -> None:
    """
    Validate configuration at startup.
    
    FAIL-FAST:
    - Missing config files → hard crash
    - Invalid YAML → hard crash
    - Hash computation fails → hard crash
    
    Call this at server startup.
    """
    try:
        fingerprint = _initialize_config()
        logger.info(f"[CONFIG] Validated. Hash: {fingerprint.config_hash}")
        logger.info(f"[CONFIG] Files: {list(fingerprint.config_files.keys())}")
        logger.info(f"[CONFIG] Constants: {len(fingerprint.constants)} values")
    except FileNotFoundError as e:
        logger.critical(f"[CONFIG] FATAL: Missing config file: {e}")
        raise SystemExit(f"FATAL: Missing config file: {e}")
    except Exception as e:
        logger.critical(f"[CONFIG] FATAL: Config validation failed: {e}")
        raise SystemExit(f"FATAL: Config validation failed: {e}")
