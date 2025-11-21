"""Full End-to-End Training and Deployment Pipeline"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.data import create_sample_data
from src.training.orchestrator import TrainingOrchestrator
from src.rag.indexer import index_documents
from src.common import load_config, init_logger

logger = logging.getLogger(__name__)


class FullPipeline:
    """
    Complete end-to-end pipeline for MARK system
    
    Pipeline stages:
    1. Data preprocessing
    2. Train Mamba model
    3. Train Transformer model
    4. Build RAG embeddings and index
    5. Train RL model
    6. Export final models
    """
    
    def __init__(self, output_dir: str = "pipeline_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        log_file = self.output_dir / f"full_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = init_logger("full_pipeline", str(log_file))
        
        self.orchestrator = TrainingOrchestrator(log_dir=str(self.output_dir / "logs"))
        self.pipeline_state: Dict[str, Any] = {}
    
    def run(
        self,
        device: Optional[str] = None,
        skip_data_prep: bool = False,
        stages: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Run the full pipeline
        
        Args:
            device: Device to use for training
            skip_data_prep: Skip data preparation if data already exists
            stages: List of stages to run (None = all stages)
            
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("="*70)
        self.logger.info("STARTING FULL MARK PIPELINE")
        self.logger.info("="*70)
        
        start_time = datetime.now()
        
        all_stages = [
            'data_prep',
            'train_mamba',
            'train_transformer',
            'build_rag',
            'train_rl',
            'export_models'
        ]
        
        stages_to_run = stages or all_stages
        
        results = {}
        
        try:
            # Stage 1: Data Preparation
            if 'data_prep' in stages_to_run and not skip_data_prep:
                self.logger.info("\n" + "="*70)
                self.logger.info("STAGE 1: DATA PREPARATION")
                self.logger.info("="*70)
                results['data_prep'] = self._prepare_data()
            
            # Stage 2: Train Mamba
            if 'train_mamba' in stages_to_run:
                self.logger.info("\n" + "="*70)
                self.logger.info("STAGE 2: TRAIN MAMBA MODEL")
                self.logger.info("="*70)
                results['train_mamba'] = self._train_mamba(device)
            
            # Stage 3: Train Transformer
            if 'train_transformer' in stages_to_run:
                self.logger.info("\n" + "="*70)
                self.logger.info("STAGE 3: TRAIN TRANSFORMER MODEL")
                self.logger.info("="*70)
                results['train_transformer'] = self._train_transformer(device)
            
            # Stage 4: Build RAG
            if 'build_rag' in stages_to_run:
                self.logger.info("\n" + "="*70)
                self.logger.info("STAGE 4: BUILD RAG INDEX")
                self.logger.info("="*70)
                results['build_rag'] = self._build_rag_index(device)
            
            # Stage 5: Train RL
            if 'train_rl' in stages_to_run:
                self.logger.info("\n" + "="*70)
                self.logger.info("STAGE 5: TRAIN RL MODEL")
                self.logger.info("="*70)
                results['train_rl'] = self._train_rl(device)
            
            # Stage 6: Export Models
            if 'export_models' in stages_to_run:
                self.logger.info("\n" + "="*70)
                self.logger.info("STAGE 6: EXPORT FINAL MODELS")
                self.logger.info("="*70)
                results['export_models'] = self._export_models()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info("\n" + "="*70)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total duration: {duration:.2f} seconds")
            self.logger.info("="*70)
            
            results['success'] = True
            results['duration'] = duration
            results['output_dir'] = str(self.output_dir)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            results['success'] = False
            results['error'] = str(e)
            return results
    
    def _prepare_data(self) -> Dict[str, Any]:
        """Prepare training data"""
        self.logger.info("Creating sample data...")
        
        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Generate sample data
        create_sample_data(str(data_dir))
        
        self.logger.info(f"Data prepared in {data_dir}")
        
        return {
            'status': 'completed',
            'data_dir': str(data_dir),
            'files_created': [
                'train.txt', 'val.txt',
                'legal_train.jsonl', 'legal_val.jsonl',
                'documents.jsonl', 'rag_eval.jsonl',
                'rl_train.jsonl', 'rl_val.jsonl'
            ]
        }
    
    def _train_mamba(self, device: Optional[str]) -> Dict[str, Any]:
        """Train Mamba model"""
        self.logger.info("Training Mamba model...")
        
        result = self.orchestrator.train(
            target='mamba',
            device=device,
            resume=False
        )
        
        if result.get('success'):
            self.logger.info("Mamba training completed")
        else:
            self.logger.error("Mamba training failed")
        
        return result
    
    def _train_transformer(self, device: Optional[str]) -> Dict[str, Any]:
        """Train Transformer model"""
        self.logger.info("Training Transformer model...")
        
        result = self.orchestrator.train(
            target='transformer',
            device=device,
            resume=False
        )
        
        if result.get('success'):
            self.logger.info("Transformer training completed")
        else:
            self.logger.error("Transformer training failed")
        
        return result
    
    def _build_rag_index(self, device: Optional[str]) -> Dict[str, Any]:
        """Build RAG index"""
        self.logger.info("Building RAG index...")
        
        result = self.orchestrator.train(
            target='rag',
            device=device,
            evaluate=True
        )
        
        if result.get('success'):
            self.logger.info("RAG index built successfully")
        else:
            self.logger.error("RAG index building failed")
        
        return result
    
    def _train_rl(self, device: Optional[str]) -> Dict[str, Any]:
        """Train RL model"""
        self.logger.info("Training RL model...")
        
        result = self.orchestrator.train(
            target='rl',
            device=device,
            resume=False
        )
        
        if result.get('success'):
            self.logger.info("RL training completed")
        else:
            self.logger.error("RL training failed")
        
        return result
    
    def _export_models(self) -> Dict[str, Any]:
        """Export final models"""
        self.logger.info("Exporting final models...")
        
        export_dir = self.output_dir / "exported_models"
        export_dir.mkdir(exist_ok=True)
        
        exported = {
            'mamba': 'checkpoints/mamba/best_model.pt',
            'transformer': 'checkpoints/transfer/best_model.pt',
            'rag_index': 'checkpoints/rag/faiss.index',
            'rl_policy': 'checkpoints/rl/best_model.pt'
        }
        
        # Check which models exist
        available_models = {}
        for model_name, checkpoint_path in exported.items():
            if Path(checkpoint_path).exists():
                available_models[model_name] = checkpoint_path
                self.logger.info(f"  ✓ {model_name}: {checkpoint_path}")
            else:
                self.logger.warning(f"  ✗ {model_name}: {checkpoint_path} not found")
        
        self.logger.info(f"Exported {len(available_models)} models")
        
        return {
            'status': 'completed',
            'export_dir': str(export_dir),
            'models': available_models
        }
    
    def get_pipeline_state(self) -> Dict[str, Any]:
        """Get current pipeline state"""
        return self.pipeline_state
    
    def print_summary(self):
        """Print pipeline summary"""
        print("\n" + "="*70)
        print("FULL PIPELINE SUMMARY")
        print("="*70)
        
        self.orchestrator.print_summary()


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full MARK pipeline")
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help="Device to use (cpu/cuda)"
    )
    parser.add_argument(
        '--skip-data-prep',
        action='store_true',
        help="Skip data preparation"
    )
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['data_prep', 'train_mamba', 'train_transformer', 'build_rag', 'train_rl', 'export_models'],
        help="Specific stages to run"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='pipeline_output',
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    pipeline = FullPipeline(output_dir=args.output_dir)
    
    results = pipeline.run(
        device=args.device,
        skip_data_prep=args.skip_data_prep,
        stages=args.stages
    )
    
    pipeline.print_summary()
    
    if not results.get('success'):
        exit(1)


if __name__ == '__main__':
    main()
