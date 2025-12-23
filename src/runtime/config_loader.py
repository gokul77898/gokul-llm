"""
Phase-6: Deployment Readiness - Config Loader

Loads configuration from YAML with environment variable overrides.
Validates required keys for deployment readiness.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml


logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    pass


class ConfigLoader:
    """Load and validate configuration with environment overrides.
    
    Priority (highest to lowest):
    1. Environment variables
    2. YAML file
    3. Defaults
    """
    
    # Required top-level keys
    REQUIRED_KEYS = [
        'paths',
        'encoder',
        'decoder',
        'retrieval',
    ]
    
    # Required nested keys
    REQUIRED_NESTED = {
        'paths': ['raw_dir', 'documents_dir', 'chunks_dir', 'chromadb_dir'],
        'encoder': ['model_name', 'embedding_dim'],
        'decoder': ['type', 'enable_inference'],
        'retrieval': ['top_k', 'collection_name'],
    }
    
    # Environment variable mappings
    ENV_MAPPINGS = {
        'RAG_ENCODER_MODEL': ('encoder', 'model_name'),
        'RAG_DECODER_TYPE': ('decoder', 'type'),
        'RAG_DECODER_MODEL': ('decoder', 'model_name'),
        'RAG_DECODER_DEVICE': ('decoder', 'device'),
        'RAG_ENABLE_INFERENCE': ('decoder', 'enable_inference'),
        'RAG_COLLECTION_NAME': ('retrieval', 'collection_name'),
        'RAG_TOP_K': ('retrieval', 'top_k'),
        'CHROMADB_DIR': ('paths', 'chromadb_dir'),
    }
    
    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
        
        Returns:
            Configuration dictionary
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded config from: {config_path}")
            return config
        
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
            raise
    
    @staticmethod
    def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to config.
        
        Args:
            config: Base configuration dictionary
        
        Returns:
            Configuration with environment overrides applied
        """
        overrides_applied = []
        
        for env_var, (section, key) in ConfigLoader.ENV_MAPPINGS.items():
            env_value = os.environ.get(env_var)
            
            if env_value is not None:
                # Ensure section exists
                if section not in config:
                    config[section] = {}
                
                # Convert value types
                if key == 'enable_inference':
                    # Boolean conversion
                    env_value = env_value.lower() in ('true', '1', 'yes')
                elif key == 'top_k':
                    # Integer conversion
                    env_value = int(env_value)
                
                # Apply override
                config[section][key] = env_value
                overrides_applied.append(f"{env_var} -> {section}.{key}")
        
        if overrides_applied:
            logger.info(f"Applied {len(overrides_applied)} environment overrides")
            for override in overrides_applied:
                logger.debug(f"  {override}")
        
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> List[str]:
        """Validate configuration has required keys.
        
        Args:
            config: Configuration dictionary to validate
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required top-level keys
        for key in ConfigLoader.REQUIRED_KEYS:
            if key not in config:
                errors.append(f"Missing required top-level key: {key}")
        
        # Check required nested keys
        for section, required_keys in ConfigLoader.REQUIRED_NESTED.items():
            if section not in config:
                continue  # Already reported above
            
            section_config = config[section]
            if not isinstance(section_config, dict):
                errors.append(f"Section '{section}' must be a dictionary")
                continue
            
            for key in required_keys:
                if key not in section_config:
                    errors.append(f"Missing required key: {section}.{key}")
        
        return errors
    
    @staticmethod
    def load_config(
        config_path: str,
        apply_env: bool = True,
        validate: bool = True
    ) -> Dict[str, Any]:
        """Load, override, and validate configuration.
        
        Args:
            config_path: Path to YAML config file
            apply_env: Apply environment variable overrides (default: True)
            validate: Validate required keys (default: True)
        
        Returns:
            Validated configuration dictionary
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            ConfigValidationError: If validation fails
        """
        # Load YAML
        config = ConfigLoader.load_yaml(config_path)
        
        # Apply environment overrides
        if apply_env:
            config = ConfigLoader.apply_env_overrides(config)
        
        # Validate
        if validate:
            errors = ConfigLoader.validate_config(config)
            if errors:
                error_msg = "Config validation failed:\n  " + "\n  ".join(errors)
                logger.error(error_msg)
                raise ConfigValidationError(error_msg)
            
            logger.info("Config validation passed")
        
        return config


def load_config(
    config_path: str,
    apply_env: bool = True,
    validate: bool = True
) -> Dict[str, Any]:
    """Convenience function to load configuration.
    
    Args:
        config_path: Path to YAML config file
        apply_env: Apply environment variable overrides (default: True)
        validate: Validate required keys (default: True)
    
    Returns:
        Validated configuration dictionary
    """
    return ConfigLoader.load_config(config_path, apply_env, validate)
