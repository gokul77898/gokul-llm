"""
Phase-6: Deployment Readiness - Health Check

System health check for deployment readiness.
Validates config, retriever, decoder, and inference status.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of system health check."""
    healthy: bool
    checks: Dict[str, bool]
    errors: List[str]
    warnings: List[str]
    info: Dict[str, Any]


class HealthChecker:
    """System health checker for deployment readiness."""
    
    @staticmethod
    def check_config_loaded(config: Optional[Dict[str, Any]]) -> tuple[bool, Optional[str]]:
        """Check if config is loaded and valid.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Tuple of (success: bool, error: Optional[str])
        """
        if config is None:
            return False, "Config is None"
        
        if not isinstance(config, dict):
            return False, f"Config is not a dict: {type(config)}"
        
        if not config:
            return False, "Config is empty"
        
        return True, None
    
    @staticmethod
    def check_retriever_ready(config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Check if retriever is ready (ChromaDB path exists).
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Tuple of (success: bool, error: Optional[str])
        """
        try:
            paths = config.get('paths', {})
            chromadb_dir = paths.get('chromadb_dir')
            
            if not chromadb_dir:
                return False, "ChromaDB directory not configured"
            
            # Check if directory exists (not if it has data - that's runtime)
            chromadb_path = Path(chromadb_dir)
            if not chromadb_path.exists():
                return True, None  # OK if doesn't exist yet - will be created
            
            return True, None
        
        except Exception as e:
            return False, f"Retriever check failed: {e}"
    
    @staticmethod
    def check_decoder_registered(config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Check if decoder is registered and available.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Tuple of (success: bool, error: Optional[str])
        """
        try:
            decoder_config = config.get('decoder', {})
            
            if not decoder_config:
                return False, "Decoder not configured"
            
            decoder_type = decoder_config.get('type')
            if not decoder_type:
                return False, "Decoder type not specified"
            
            # Check if decoder type is supported
            supported_types = ['qwen', 'llama', 'stub']
            if decoder_type not in supported_types:
                return False, f"Unsupported decoder type: {decoder_type}"
            
            # Try to import decoder registry
            try:
                from src.decoders import get_decoder
                
                # Try to instantiate decoder (won't load model in stub mode)
                decoder = get_decoder(decoder_config)
                
                return True, None
            
            except ImportError as e:
                return False, f"Decoder module not available: {e}"
            
            except Exception as e:
                return False, f"Decoder instantiation failed: {e}"
        
        except Exception as e:
            return False, f"Decoder check failed: {e}"
    
    @staticmethod
    def check_inference_status(config: Dict[str, Any]) -> tuple[bool, str, List[str]]:
        """Check inference enabled/disabled status.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Tuple of (success: bool, status: str, warnings: List[str])
        """
        warnings = []
        
        try:
            decoder_config = config.get('decoder', {})
            enable_inference = decoder_config.get('enable_inference', False)
            
            if enable_inference:
                status = "ENABLED"
                
                # Check for potential issues
                import os
                if os.environ.get('ALLOW_LLM_INFERENCE') != '1':
                    warnings.append(
                        "Inference enabled in config but ALLOW_LLM_INFERENCE env var not set. "
                        "Runtime guards will block inference."
                    )
                
                device = decoder_config.get('device', 'cpu')
                model_name = decoder_config.get('model_name', '')
                
                # Check for large models on CPU
                large_models = [
                    "Qwen/Qwen2.5-32B-Instruct",
                    "Qwen/Qwen2.5-72B-Instruct",
                    "meta-llama/Llama-3-70B-Instruct",
                    "meta-llama/Llama-2-70B-chat",
                ]
                
                if model_name in large_models and device == 'cpu':
                    warnings.append(
                        f"Large model {model_name} configured with device=cpu. "
                        f"Runtime guards will block inference. Use device=cuda or device=mps."
                    )
            else:
                status = "DISABLED (safe stub mode)"
            
            return True, status, warnings
        
        except Exception as e:
            return False, f"ERROR: {e}", warnings
    
    @staticmethod
    def health_check(config: Optional[Dict[str, Any]]) -> HealthCheckResult:
        """Perform comprehensive system health check.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            HealthCheckResult with detailed status
        """
        checks = {}
        errors = []
        warnings = []
        info = {}
        
        # Check 1: Config loaded
        success, error = HealthChecker.check_config_loaded(config)
        checks['config_loaded'] = success
        if not success:
            errors.append(f"Config: {error}")
            # Can't continue without config
            return HealthCheckResult(
                healthy=False,
                checks=checks,
                errors=errors,
                warnings=warnings,
                info=info
            )
        
        # Check 2: Retriever ready
        success, error = HealthChecker.check_retriever_ready(config)
        checks['retriever_ready'] = success
        if not success:
            errors.append(f"Retriever: {error}")
        else:
            collection_name = config.get('retrieval', {}).get('collection_name', 'unknown')
            info['collection_name'] = collection_name
        
        # Check 3: Decoder registered
        success, error = HealthChecker.check_decoder_registered(config)
        checks['decoder_registered'] = success
        if not success:
            errors.append(f"Decoder: {error}")
        else:
            decoder_type = config.get('decoder', {}).get('type', 'unknown')
            info['decoder_type'] = decoder_type
        
        # Check 4: Inference status
        success, status, check_warnings = HealthChecker.check_inference_status(config)
        checks['inference_status_checked'] = success
        info['inference_status'] = status
        warnings.extend(check_warnings)
        
        # Overall health
        healthy = all(checks.values()) and not errors
        
        return HealthCheckResult(
            healthy=healthy,
            checks=checks,
            errors=errors,
            warnings=warnings,
            info=info
        )


def health_check(config: Optional[Dict[str, Any]]) -> HealthCheckResult:
    """Convenience function for system health check.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        HealthCheckResult with detailed status
    """
    return HealthChecker.health_check(config)


def print_health_check(result: HealthCheckResult, verbose: bool = False) -> None:
    """Print health check result in human-readable format.
    
    Args:
        result: HealthCheckResult to print
        verbose: Show detailed check results (default: False)
    """
    print("=" * 70)
    print("SYSTEM HEALTH CHECK")
    print("=" * 70)
    
    # Overall status
    status_symbol = "✓" if result.healthy else "✗"
    status_text = "HEALTHY" if result.healthy else "UNHEALTHY"
    print(f"\nOverall Status: {status_symbol} {status_text}")
    
    # Individual checks
    if verbose:
        print("\nChecks:")
        for check_name, passed in result.checks.items():
            symbol = "✓" if passed else "✗"
            print(f"  {symbol} {check_name}")
    
    # Info
    if result.info:
        print("\nInfo:")
        for key, value in result.info.items():
            print(f"  {key}: {value}")
    
    # Errors
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  ✗ {error}")
    
    # Warnings
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  ⚠ {warning}")
    
    print()
