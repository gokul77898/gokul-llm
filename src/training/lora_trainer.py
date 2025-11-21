"""
LoRA Fine-Tuning Trainer

Supervised fine-tuning with LoRA adapters for MARK models.

Usage:
    # Dry run (default - no training)
    python -m src.training.lora_trainer --config configs/lora_sft.yaml --dry-run
    
    # Actual training (requires confirmation)
    python -m src.training.lora_trainer --config configs/lora_sft.yaml --confirm-run
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

from src.training.utils import (
    SFTDataset,
    load_config,
    print_trainable_parameters,
    get_device,
    format_time
)


class LoRATrainer:
    """LoRA fine-tuning trainer."""
    
    def __init__(self, config_path: str):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to config YAML
        """
        self.config = load_config(config_path)
        self.device = get_device(self.config['hardware']['device'])
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        print("="*70)
        print("  LoRA FINE-TUNING TRAINER")
        print("="*70)
        print(f"Config: {config_path}")
        print(f"Device: {self.device}")
        print(f"Model: {self.config['model']['name']}")
    
    def load_model(self):
        """Load base model and apply LoRA (auto-detects backend)."""
        print("\n🔧 Loading model...")
        
        model_name = self.config['model']['name']
        base_model = self.config['model'].get('base_model', 'gpt2')
        
        # Check if this is Mamba (will auto-detect backend)
        is_mamba = 'mamba' in model_name.lower()
        
        if is_mamba:
            print(f"   🎯 Auto-detecting Mamba backend...")
            
            # Use the auto-detection system
            from src.core.mamba_loader import load_mamba_model, detect_mamba_backend
            
            backend = detect_mamba_backend()
            print(f"   📍 Detected backend: {backend}")
            
            if backend == "real-mamba":
                # Load REAL Mamba SSM
                print(f"   🔥 Loading REAL Mamba SSM: {base_model}")
                
                try:
                    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
                    
                    # Load tokenizer
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                    except:
                        print(f"   ℹ️  Using GPT-2 tokenizer as fallback")
                        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                    
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    print(f"   ✅ Tokenizer loaded")
                    
                    # Load REAL Mamba model
                    device_str = "cuda" if torch.cuda.is_available() else "cpu"
                    self.model = MambaLMHeadModel.from_pretrained(
                        base_model,
                        device=device_str,
                        dtype=torch.float32
                    )
                    self.mamba_backend = "real-mamba"
                    print(f"   ✅ REAL Mamba SSM loaded")
                    
                except ImportError as e:
                    print(f"   ❌ REAL Mamba SSM not available: {e}")
                    print(f"   Install with: pip install mamba-ssm causal-conv1d>=1.2.0")
                    self.model = None
                    return
                except Exception as e:
                    print(f"   ⚠️  Failed to load REAL Mamba SSM: {e}")
                    self.model = None
                    return
            
            elif backend == "mamba2":
                # Load Mamba2
                print(f"   🍎 Loading Mamba2 (Mac optimized)")
                
                try:
                    from mamba2.models.mamba2 import Mamba2LMHeadModel
                    
                    # Load tokenizer
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                    except:
                        print(f"   ℹ️  Using GPT-2 tokenizer as fallback")
                        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                    
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    print(f"   ✅ Tokenizer loaded")
                    
                    # Load Mamba2 model
                    device_str = "mps" if torch.backends.mps.is_available() else "cpu"
                    self.model = Mamba2LMHeadModel.from_pretrained(
                        base_model,
                        device=device_str,
                        dtype=torch.float32
                    )
                    self.mamba_backend = "mamba2"
                    print(f"   ✅ Mamba2 loaded")
                    
                except ImportError as e:
                    print(f"   ❌ Mamba2 not available: {e}")
                    print(f"   Install with: pip install mamba2")
                    self.model = None
                    return
                except Exception as e:
                    print(f"   ⚠️  Failed to load Mamba2: {e}")
                    self.model = None
                    return
            
            else:
                print(f"   ❌ No Mamba backend available")
                self.model = None
                return
        
        else:
            # Load standard Transformer model
            print(f"   Loading Transformer model: {base_model}")
            self.mamba_backend = None
            
            # Load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            except:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"   ✅ Tokenizer loaded")
            
            # Load base model
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float32,
                    device_map=None
                )
                print(f"   ✅ Base model loaded")
            except Exception as e:
                print(f"   ⚠️  Using mock model (error: {e})")
                self.model = None
                return
        
        # Apply LoRA with backend-specific target modules
        try:
            from peft import LoraConfig, get_peft_model
            
            # Determine target modules based on backend
            if hasattr(self, 'mamba_backend') and self.mamba_backend == "real-mamba":
                # REAL Mamba SSM modules
                target_modules = ["in_proj", "out_proj", "x_proj", "dt_proj"]
                print(f"   📍 Using REAL Mamba SSM LoRA targets: {target_modules}")
            
            elif hasattr(self, 'mamba_backend') and self.mamba_backend == "mamba2":
                # Mamba2 modules
                target_modules = ["mixer.Wq", "mixer.Wk", "mixer.Wv"]
                print(f"   📍 Using Mamba2 LoRA targets: {target_modules}")
            
            else:
                # Transformer modules (use config or default)
                target_modules = self.config['lora'].get('target_modules', ['c_attn', 'c_proj'])
                print(f"   📍 Using Transformer LoRA targets: {target_modules}")
            
            lora_config = LoraConfig(
                r=self.config['lora']['r'],
                lora_alpha=self.config['lora']['lora_alpha'],
                lora_dropout=self.config['lora']['lora_dropout'],
                target_modules=target_modules,
                bias=self.config['lora']['bias'],
                task_type=self.config['lora']['task_type']
            )
            
            self.model = get_peft_model(self.model, lora_config)
            print(f"   ✅ LoRA adapters applied")
            print_trainable_parameters(self.model)
            
        except ImportError:
            print("   ⚠️  PEFT not installed, using full model (not recommended)")
        except Exception as e:
            print(f"   ⚠️  LoRA application failed: {e}")
    
    def load_datasets(self):
        """Load training and validation datasets."""
        print("\n📂 Loading datasets...")
        
        train_file = Path(self.config['data']['train_file'])
        val_file = Path(self.config['data']['val_file'])
        
        if not train_file.exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")
        if not val_file.exists():
            raise FileNotFoundError(f"Validation file not found: {val_file}")
        
        max_length = self.config['data']['max_seq_length']
        
        self.train_dataset = SFTDataset(str(train_file), self.tokenizer, max_length)
        self.val_dataset = SFTDataset(str(val_file), self.tokenizer, max_length)
        
        print(f"   ✅ Training samples: {len(self.train_dataset)}")
        print(f"   ✅ Validation samples: {len(self.val_dataset)}")
    
    def setup_trainer(self):
        """Setup Hugging Face Trainer."""
        print("\n⚙️  Setting up trainer...")
        
        output_dir = Path(self.config['output']['checkpoint_dir']) / self.config['output']['model_name']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_config = self.config['training']
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=training_config['epochs'],
            per_device_train_batch_size=training_config['batch_size'],
            per_device_eval_batch_size=training_config['batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            warmup_steps=training_config['warmup_steps'],
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            eval_steps=training_config['eval_steps'],
            max_grad_norm=training_config['max_grad_norm'],
            weight_decay=training_config['weight_decay'],
            save_total_limit=self.config['output']['save_total_limit'],
            fp16=self.config['hardware']['fp16'],
            bf16=self.config['hardware']['bf16'],
            report_to=self.config['logging']['report_to'],
            logging_dir=self.config['logging']['logging_dir'],
            eval_strategy="steps" if training_config['epochs'] > 0 else "no",
            save_strategy="steps" if training_config['epochs'] > 0 else "no",
            load_best_model_at_end=True if training_config['epochs'] > 0 else False,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )
        
        print(f"   ✅ Trainer configured")
        print(f"   Output: {output_dir}")
    
    def dry_run(self):
        """Run dry-run validation without training."""
        print("\n" + "="*70)
        print("  DRY RUN MODE")
        print("="*70)
        print("✅ Model loaded successfully")
        print("✅ Datasets loaded successfully")
        print("✅ Trainer configured successfully")
        print("\n📊 Training Plan:")
        print(f"   Epochs: {self.config['training']['epochs']}")
        print(f"   Batch size: {self.config['training']['batch_size']}")
        print(f"   Training samples: {len(self.train_dataset)}")
        print(f"   Validation samples: {len(self.val_dataset)}")
        print(f"   Steps per epoch: {len(self.train_dataset) // self.config['training']['batch_size']}")
        
        print("\n⚠️  DRY RUN COMPLETE - NO TRAINING PERFORMED")
        print("\n💡 To start actual training:")
        print("   1. Update config: set training.epochs > 0 and training.dry_run = false")
        print("   2. Run with: --confirm-run flag")
        print("   3. Command: python -m src.training.lora_trainer --config configs/lora_sft.yaml --confirm-run")
    
    def train(self):
        """Run actual training."""
        print("\n" + "="*70)
        print("  STARTING TRAINING")
        print("="*70)
        
        start_time = time.time()
        
        try:
            self.trainer.train()
            
            # Save final model
            output_dir = Path(self.config['output']['checkpoint_dir']) / self.config['output']['model_name']
            self.trainer.save_model(str(output_dir / "final"))
            
            elapsed = time.time() - start_time
            
            print("\n" + "="*70)
            print("  TRAINING COMPLETE")
            print("="*70)
            print(f"✅ Time elapsed: {format_time(elapsed)}")
            print(f"✅ Model saved: {output_dir / 'final'}")
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning Trainer")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run validation only, no training'
    )
    parser.add_argument(
        '--confirm-run',
        action='store_true',
        help='Confirm and start actual training (required for training)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['mamba', 'transformer'],
        default=None,
        help='Specific model to train (mamba or transformer)'
    )
    
    args = parser.parse_args()
    
    try:
        # Override config based on --model flag
        config_path = args.config
        if args.model:
            # Use model-specific config
            if args.model == 'mamba':
                config_path = 'configs/lora_mamba.yaml'
                print(f"\n📋 Using Mamba-specific config: {config_path}")
            elif args.model == 'transformer':
                config_path = 'configs/lora_transformer.yaml'
                print(f"\n📋 Using Transformer-specific config: {config_path}")
        
        # Initialize trainer
        trainer = LoRATrainer(config_path)
        
        # Check if datasets exist
        train_file = Path(trainer.config['data']['train_file'])
        if not train_file.exists():
            print(f"\n❌ Error: Training data not found: {train_file}")
            print("\n💡 Generate data first:")
            print("   python -m src.training.data_prep --collection pdf_docs --out-dir data/")
            return 1
        
        # Load model and datasets
        trainer.load_model()
        
        if trainer.model is None:
            print("\n⚠️  Model loading failed - running in validation mode only")
            print("   This is expected if PEFT or required libraries are not installed")
            print("   Install with: pip install peft transformers accelerate")
            return 0
        
        trainer.load_datasets()
        trainer.setup_trainer()
        
        # Determine mode
        config_dry_run = trainer.config['training'].get('dry_run', True)
        config_epochs = trainer.config['training'].get('epochs', 0)
        
        if args.dry_run or config_dry_run or config_epochs == 0:
            # Dry run mode
            trainer.dry_run()
            return 0
        
        elif args.confirm_run:
            # Actual training
            print("\n⚠️  TRAINING MODE ACTIVATED")
            print("   This will start actual model fine-tuning")
            print("   Ctrl+C to cancel within 5 seconds...")
            time.sleep(5)
            
            trainer.train()
            return 0
        
        else:
            print("\n❌ Error: Training mode requires --confirm-run flag")
            print("\n💡 To start training:")
            print("   python -m src.training.lora_trainer --config configs/lora_sft.yaml --confirm-run")
            return 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
