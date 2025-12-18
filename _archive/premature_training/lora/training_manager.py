"""
Training Pipeline Manager

Orchestrates the complete training pipeline: data prep → LoRA → RLHF → eval.

Usage:
    python -m src.training.training_manager --stage sft|lora|rlhf|eval --confirm
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any

from src.training.utils import load_config


class TrainingManager:
    """Manages training pipeline stages."""
    
    STAGES = ['data_prep', 'lora', 'rlhf', 'eval']
    
    def __init__(self):
        """Initialize training manager."""
        self.status = {
            'data_prep': 'not_started',
            'lora': 'not_started',
            'rlhf': 'disabled',
            'eval': 'not_started'
        }
        
        print("="*70)
        print("  TRAINING PIPELINE MANAGER")
        print("="*70)
    
    def check_prerequisites(self, stage: str) -> bool:
        """Check if prerequisites for a stage are met."""
        if stage == 'data_prep':
            # Check if ChromaDB collection exists
            if not Path("chroma_db").exists():
                print("⚠️  ChromaDB not found")
                print("   Run PDF ingestion first")
                return False
            return True
        
        elif stage == 'lora':
            # Check if training data exists
            if not Path("data/train_sft.jsonl").exists():
                print("⚠️  Training data not found")
                print("   Run data_prep stage first:")
                print("   python -m src.training.training_manager --stage data_prep --confirm")
                return False
            return True
        
        elif stage == 'rlhf':
            # Check if LoRA model exists
            if not Path("checkpoints/lora").exists():
                print("⚠️  LoRA model not found")
                print("   Run lora stage first")
                return False
            return True
        
        elif stage == 'eval':
            # Check if validation data exists
            if not Path("data/val_sft.jsonl").exists():
                print("⚠️  Validation data not found")
                print("   Run data_prep stage first")
                return False
            return True
        
        return True
    
    def run_data_prep(self, confirm: bool = False):
        """Run data preparation stage."""
        print("\n" + "="*70)
        print("  STAGE: DATA PREPARATION")
        print("="*70)
        
        if not confirm:
            print("⚠️  Dry-run mode - showing plan only")
            print("\nThis stage will:")
            print("  1. Connect to ChromaDB collection 'pdf_docs'")
            print("  2. Extract QA pairs from documents")
            print("  3. Generate training/validation JSONL files")
            print("\nTo execute: add --confirm flag")
            return
        
        cmd = [
            sys.executable, "-m", "src.training.data_prep",
            "--collection", "pdf_docs",
            "--out-dir", "data/",
            "--top-k", "3",
            "--max-samples", "1000"
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            self.status['data_prep'] = 'completed'
            print("\n✅ Data preparation complete")
        else:
            print("\n❌ Data preparation failed")
    
    def run_lora(self, confirm: bool = False):
        """Run LoRA fine-tuning stage."""
        print("\n" + "="*70)
        print("  STAGE: LORA FINE-TUNING")
        print("="*70)
        
        if not confirm:
            print("⚠️  Dry-run mode - validation only")
            
            cmd = [
                sys.executable, "-m", "src.training.lora_trainer",
                "--config", "configs/lora_sft.yaml",
                "--dry-run"
            ]
        else:
            print("⚠️  TRAINING MODE - will start fine-tuning")
            
            cmd = [
                sys.executable, "-m", "src.training.lora_trainer",
                "--config", "configs/lora_sft.yaml",
                "--confirm-run"
            ]
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            if confirm:
                self.status['lora'] = 'completed'
                print("\n✅ LoRA training complete")
            else:
                print("\n✅ Dry-run validation complete")
        else:
            print("\n❌ LoRA stage failed")
    
    def run_rlhf(self, confirm: bool = False):
        """Run RLHF stage (skeleton only)."""
        print("\n" + "="*70)
        print("  STAGE: RLHF (SKELETON)")
        print("="*70)
        print("⚠️  RLHF is not fully implemented")
        print("   This is a placeholder for future development")
        
        if confirm:
            cmd = [
                sys.executable, "-m", "src.training.ppo_trainer",
                "--config", "configs/rlhf.yaml",
                "--confirm-run"
            ]
            result = subprocess.run(cmd)
        else:
            print("\n💡 RLHF would require:")
            print("  - Preference dataset")
            print("  - Trained reward model")
            print("  - PPO training loop")
            print("  - Significant compute resources")
    
    def run_eval(self, confirm: bool = False):
        """Run evaluation stage."""
        print("\n" + "="*70)
        print("  STAGE: EVALUATION")
        print("="*70)
        
        cmd = [
            sys.executable, "-m", "src.training.eval",
            "--config", "configs/eval.yaml",
            "--model", "mamba_lora",
            "--dataset", "data/val_sft.jsonl"
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            self.status['eval'] = 'completed'
            print("\n✅ Evaluation complete")
        else:
            print("\n❌ Evaluation failed")
    
    def run_stage(self, stage: str, confirm: bool = False):
        """Run a specific training stage."""
        if stage not in self.STAGES:
            print(f"❌ Unknown stage: {stage}")
            print(f"   Valid stages: {', '.join(self.STAGES)}")
            return False
        
        # Check prerequisites
        if not self.check_prerequisites(stage):
            return False
        
        # Run stage
        if stage == 'data_prep':
            self.run_data_prep(confirm)
        elif stage == 'lora':
            self.run_lora(confirm)
        elif stage == 'rlhf':
            self.run_rlhf(confirm)
        elif stage == 'eval':
            self.run_eval(confirm)
        
        return True


def main():
    """Main entry point for training manager."""
    parser = argparse.ArgumentParser(
        description="Training Pipeline Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run data prep
  python -m src.training.training_manager --stage data_prep
  
  # Actually run data prep
  python -m src.training.training_manager --stage data_prep --confirm
  
  # Dry-run LoRA training
  python -m src.training.training_manager --stage lora
  
  # Actually run LoRA training (after data prep)
  python -m src.training.training_manager --stage lora --confirm
  
  # Run evaluation
  python -m src.training.training_manager --stage eval --confirm
        """
    )
    parser.add_argument(
        '--stage',
        type=str,
        required=True,
        choices=['data_prep', 'lora', 'rlhf', 'eval'],
        help='Training stage to run'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Confirm and execute (without this, runs dry-run)'
    )
    
    args = parser.parse_args()
    
    try:
        manager = TrainingManager()
        success = manager.run_stage(args.stage, args.confirm)
        
        if success:
            print("\n✅ Stage completed successfully")
            return 0
        else:
            print("\n❌ Stage failed or prerequisites not met")
            return 1
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
