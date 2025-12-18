"""Unified Training Orchestrator for MARK System"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.common import load_config, init_logger

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Orchestrates training across all MARK components"""
    
    # Phase 3.7: RL training disabled - archived to _archive/premature_training/
    VALID_TARGETS = ['mamba', 'transformer', 'rag', 'full-pipeline']
    
    def __init__(self, log_dir: str = "logs/orchestrator"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        log_file = self.log_dir / f"orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = init_logger("orchestrator", str(log_file))
        
        self.training_history: List[Dict[str, Any]] = []
    
    def train(
        self,
        target: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        resume: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a specific component
        
        Args:
            target: Training target (mamba, transformer, rag, rl, full-pipeline)
            config_path: Path to config file (optional, uses default if None)
            device: Device to train on
            resume: Resume from checkpoint
            **kwargs: Additional arguments for training script
            
        Returns:
            Dictionary with training results
        """
        if target not in self.VALID_TARGETS:
            raise ValueError(f"Invalid target: {target}. Valid targets: {self.VALID_TARGETS}")
        
        self.logger.info(f"Starting training for target: {target}")
        start_time = datetime.now()
        
        try:
            if target == 'full-pipeline':
                result = self._train_full_pipeline(device, resume)
            else:
                result = self._train_single_component(target, config_path, device, resume, **kwargs)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result.update({
                'target': target,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'success': True
            })
            
            self.training_history.append(result)
            self.logger.info(f"Training completed for {target} in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Training failed for {target}: {e}", exc_info=True)
            result = {
                'target': target,
                'success': False,
                'error': str(e)
            }
            self.training_history.append(result)
            return result
    
    def _train_single_component(
        self,
        target: str,
        config_path: Optional[str],
        device: Optional[str],
        resume: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """Train a single component"""
        # Map target to script and default config
        # Phase 3.7: RL removed - archived to _archive/premature_training/
        script_map = {
            'mamba': ('src.scripts.train_mamba', 'configs/mamba_train.yaml'),
            'transformer': ('src.scripts.train_transfer', 'configs/transfer_train.yaml'),
            'rag': ('src.scripts.train_rag_indexer', 'configs/rag_indexer.yaml'),
        }
        
        script_module, default_config = script_map[target]
        config = config_path or default_config
        
        # Build command
        cmd = [sys.executable, '-m', script_module, '--config', config]
        
        if device:
            cmd.extend(['--device', device])
        
        if resume:
            cmd.append('--resume')
        
        if target == 'rag' and kwargs.get('evaluate', False):
            cmd.append('--evaluate')
        
        self.logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run training script
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Training failed with return code {result.returncode}\nStderr: {result.stderr}")
        
        # Parse checkpoint path from output
        checkpoint_path = self._extract_checkpoint_path(result.stdout, target)
        
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'checkpoint_path': checkpoint_path,
            'config_path': config
        }
    
    def _train_full_pipeline(self, device: Optional[str], resume: bool) -> Dict[str, Any]:
        """Train full pipeline sequentially"""
        self.logger.info("Starting full pipeline training")
        
        results = {}
        # Phase 3.7: RL removed from pipeline - archived to _archive/premature_training/
        pipeline_order = ['mamba', 'transformer', 'rag']
        
        for target in pipeline_order:
            self.logger.info(f"Pipeline step: {target}")
            try:
                result = self._train_single_component(
                    target=target,
                    config_path=None,
                    device=device,
                    resume=resume,
                    evaluate=(target == 'rag')
                )
                results[target] = result
                self.logger.info(f"Completed {target}")
            except Exception as e:
                self.logger.error(f"Failed at {target}: {e}")
                results[target] = {'error': str(e), 'success': False}
                # Continue with other components
        
        return {'pipeline_results': results}
    
    def _extract_checkpoint_path(self, output: str, target: str) -> Optional[str]:
        """Extract checkpoint path from training output"""
        # Look for common patterns
        patterns = [
            'Checkpoint saved to:',
            'checkpoint_dir:',
            'Saved to',
        ]
        
        for line in output.split('\n'):
            for pattern in line:
                if pattern in line:
                    # Extract path
                    parts = line.split(pattern)
                    if len(parts) > 1:
                        return parts[1].strip()
        
        # Return default path
        return f"checkpoints/{target}/"
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history"""
        return self.training_history
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*70)
        print("TRAINING ORCHESTRATOR SUMMARY")
        print("="*70)
        
        for i, record in enumerate(self.training_history, 1):
            print(f"\n{i}. {record['target'].upper()}")
            print(f"   Status: {'SUCCESS' if record.get('success') else 'FAILED'}")
            if 'duration_seconds' in record:
                print(f"   Duration: {record['duration_seconds']:.2f}s")
            if 'checkpoint_path' in record:
                print(f"   Checkpoint: {record['checkpoint_path']}")
            if 'error' in record:
                print(f"   Error: {record['error']}")
        
        print("\n" + "="*70)


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="MARK Training Orchestrator")
    parser.add_argument(
        'command',
        choices=['train', 'list-targets', 'history'],
        help="Command to run"
    )
    parser.add_argument(
        '--target',
        choices=TrainingOrchestrator.VALID_TARGETS,
        help="Training target"
    )
    parser.add_argument(
        '--config',
        type=str,
        help="Path to config file"
    )
    parser.add_argument(
        '--device',
        type=str,
        help="Device to train on (cpu/cuda)"
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help="Resume from checkpoint"
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help="Run evaluation (for RAG)"
    )
    
    args = parser.parse_args()
    
    orchestrator = TrainingOrchestrator()
    
    if args.command == 'list-targets':
        print("Available training targets:")
        for target in TrainingOrchestrator.VALID_TARGETS:
            print(f"  - {target}")
        return
    
    elif args.command == 'history':
        orchestrator.print_summary()
        return
    
    elif args.command == 'train':
        if not args.target:
            parser.error("--target is required for train command")
        
        result = orchestrator.train(
            target=args.target,
            config_path=args.config,
            device=args.device,
            resume=args.resume,
            evaluate=args.evaluate
        )
        
        orchestrator.print_summary()
        
        if not result.get('success'):
            sys.exit(1)


if __name__ == '__main__':
    main()
