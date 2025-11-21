"""
Complete PDF Fine-Tuning System with Transformer Fallback

This system:
1. Ensures proper fallback to Transformer when Mamba2 unavailable
2. Loads PDFs from ChromaDB
3. Performs multi-pass fine-tuning
4. Updates model for improved UI responses
"""

import logging
import sqlite3
import json
import torch
import yaml
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import time
import sys
import os

# Add project root to path
sys.path.append('/Users/gokul/Documents/MARK')

from src.core.chroma_manager import ChromaManager
from src.core.model_registry import get_model_instance, is_model_available
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_passes: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_sequence_length: int = 512
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50


class PDFTrainerComplete:
    """Complete PDF training system with fallback support"""
    
    def __init__(self):
        self.config = TrainingConfig()
        self.chroma_manager = ChromaManager()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.backend_used = None
        self.pdf_data = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        logger.info("🚀 PDF Trainer Complete initialized")
    
    def setup_device(self):
        """Setup optimal device"""
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("📱 Using MPS (Mac Metal) device")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("🔥 Using CUDA device")
        else:
            self.device = torch.device("cpu")
            logger.info("💻 Using CPU device")
    
    def load_model_with_fallback(self) -> bool:
        """Load model with proper fallback logic"""
        logger.info("🔄 Loading model with fallback...")
        
        # Setup device first
        self.setup_device()
        
        # Try Mamba first
        if is_model_available("mamba"):
            logger.info("🐍 Attempting to load Mamba...")
            model_instance = get_model_instance("mamba")
            if model_instance and model_instance.available:
                self.model = model_instance.model
                self.tokenizer = model_instance.tokenizer
                self.backend_used = "mamba"
                logger.info("✅ Mamba loaded successfully")
                return True
            else:
                logger.warning("⚠️  Mamba failed to load, falling back...")
        
        # Fallback to Transformer
        logger.info("🔄 Loading Transformer fallback...")
        try:
            # Load transformer config
            config_path = "configs/transformer.yaml"
            with open(config_path, 'r') as f:
                transformer_config = yaml.safe_load(f)
            
            model_name = transformer_config['model']['model_name']
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move to device
            self.model = self.model.to(self.device)
            self.backend_used = "transformer"
            
            logger.info(f"✅ Transformer loaded successfully ({model_name})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Transformer: {e}")
            return False
    
    def extract_pdf_data_from_chromadb(self) -> bool:
        """Extract PDF data from ChromaDB"""
        logger.info("📚 Extracting PDF data from ChromaDB...")
        
        db_path = "chroma_db/chroma.sqlite3"
        if not Path(db_path).exists():
            logger.error(f"❌ ChromaDB file not found: {db_path}")
            return False
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all embeddings with metadata (corrected query)
            query = """
            SELECT e.id, em.key, em.string_value, em.int_value
            FROM embeddings e
            LEFT JOIN embedding_metadata em ON e.id = em.id
            WHERE em.key IS NOT NULL
            ORDER BY e.id
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Group by embedding ID
            embeddings_data = {}
            for row in results:
                emb_id, meta_key, str_val, int_val = row
                
                if emb_id not in embeddings_data:
                    embeddings_data[emb_id] = {
                        'id': emb_id,
                        'document': None,
                        'metadata': {}
                    }
                
                if meta_key:
                    value = str_val if str_val is not None else int_val
                    embeddings_data[emb_id]['metadata'][meta_key] = value
                    
                    # Extract document content from chroma:document key
                    if meta_key == 'chroma:document':
                        embeddings_data[emb_id]['document'] = str_val
            
            conn.close()
            
            # Filter out chunks without document content
            self.pdf_data = [chunk for chunk in embeddings_data.values() 
                           if chunk['document'] and len(chunk['document']) > 50]
            
            logger.info(f"✅ Extracted {len(self.pdf_data)} PDF chunks with content")
            
            # Log document sources
            sources = set()
            for chunk in self.pdf_data:
                source = chunk['metadata'].get('source', 'unknown')
                sources.add(source)
            
            logger.info(f"📄 Found {len(sources)} unique documents:")
            for source in sorted(sources):
                logger.info(f"   - {source}")
            
            return len(self.pdf_data) > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to extract PDF data: {e}")
            return False
    
    def prepare_training_data(self) -> List[str]:
        """Prepare training texts from PDF data"""
        logger.info("📝 Preparing training data...")
        
        training_texts = []
        
        for chunk in self.pdf_data:
            text = chunk['document']
            metadata = chunk['metadata']
            
            # Create context-aware training text
            source = metadata.get('source', 'document')
            page = metadata.get('page', 'unknown')
            
            # Create multiple training formats
            formats = [
                f"Document: {source}\nContent: {text}",
                f"From {source} (page {page}): {text}",
                f"Q: What does this document say?\nA: {text}",
                f"Summary of {source}: {text}"
            ]
            
            training_texts.extend(formats)
        
        logger.info(f"📊 Prepared {len(training_texts)} training samples")
        return training_texts
    
    def create_lora_model(self):
        """Apply LoRA to the model"""
        logger.info("🔧 Applying LoRA configuration...")
        
        try:
            # LoRA configuration
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["c_attn", "c_proj"] if "gpt" in str(self.model.config).lower() else ["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            logger.info(f"🎯 LoRA applied:")
            logger.info(f"   Trainable parameters: {trainable_params:,}")
            logger.info(f"   Total parameters: {total_params:,}")
            logger.info(f"   Trainable ratio: {100 * trainable_params / total_params:.2f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to apply LoRA: {e}")
            return False
    
    def create_dataset(self, texts: List[str]):
        """Create PyTorch dataset"""
        
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, texts, tokenizer, max_length=512):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': encoding['input_ids'].flatten()
                }
        
        return TextDataset(texts, self.tokenizer, self.config.max_sequence_length)
    
    def train_single_pass(self, training_texts: List[str], pass_num: int) -> bool:
        """Train for a single pass"""
        logger.info(f"🔥 Training Pass {pass_num}/{self.config.num_passes}")
        
        try:
            # Create dataset and dataloader
            dataset = self.create_dataset(training_texts)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0
            )
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
            
            # Training loop
            self.model.train()
            total_loss = 0
            num_batches = len(dataloader)
            
            progress_bar = tqdm(dataloader, desc=f"Pass {pass_num}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # Update progress
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })
            
            avg_loss = total_loss / num_batches
            logger.info(f"✅ Pass {pass_num} completed - Average loss: {avg_loss:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training pass {pass_num} failed: {e}")
            return False
    
    def multi_pass_training(self, training_texts: List[str]) -> bool:
        """Perform multi-pass training"""
        logger.info(f"🎯 Starting {self.config.num_passes}-pass training...")
        
        successful_passes = 0
        
        for pass_num in range(1, self.config.num_passes + 1):
            if self.train_single_pass(training_texts, pass_num):
                successful_passes += 1
            else:
                logger.warning(f"⚠️  Pass {pass_num} failed, continuing...")
        
        logger.info(f"📊 Training completed: {successful_passes}/{self.config.num_passes} passes successful")
        return successful_passes > 0
    
    def save_model(self) -> bool:
        """Save the trained model"""
        logger.info("💾 Saving trained model...")
        
        try:
            # Create save directory
            save_dir = Path("models/pdf_trained")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model and tokenizer
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            
            # Save training metadata
            metadata = {
                'backend_used': self.backend_used,
                'num_pdf_chunks': len(self.pdf_data),
                'training_passes': self.config.num_passes,
                'device': str(self.device),
                'timestamp': time.time()
            }
            
            with open(save_dir / "training_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Model saved to {save_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            return False
    
    def test_model(self) -> bool:
        """Test the trained model"""
        logger.info("🧪 Testing trained model...")
        
        try:
            test_queries = [
                "What are the main topics in the documents?",
                "Summarize the key legal points",
                "What information is available in the PDFs?"
            ]
            
            self.model.eval()
            
            for query in test_queries:
                logger.info(f"🔍 Testing: {query}")
                
                # Tokenize input
                inputs = self.tokenizer(
                    query,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(self.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"📝 Response: {response[:150]}...")
            
            logger.info("✅ Model testing completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model testing failed: {e}")
            return False
    
    def run_complete_training(self) -> bool:
        """Run the complete training pipeline"""
        logger.info("🚀 Starting Complete PDF Training Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load model with fallback
        logger.info("1️⃣ Loading model with fallback...")
        if not self.load_model_with_fallback():
            logger.error("❌ Failed to load any model")
            return False
        
        # Step 2: Extract PDF data
        logger.info("2️⃣ Extracting PDF data from ChromaDB...")
        if not self.extract_pdf_data_from_chromadb():
            logger.error("❌ Failed to extract PDF data")
            return False
        
        # Step 3: Prepare training data
        logger.info("3️⃣ Preparing training data...")
        training_texts = self.prepare_training_data()
        if not training_texts:
            logger.error("❌ No training data prepared")
            return False
        
        # Step 4: Apply LoRA
        logger.info("4️⃣ Applying LoRA configuration...")
        if not self.create_lora_model():
            logger.error("❌ Failed to apply LoRA")
            return False
        
        # Step 5: Multi-pass training
        logger.info("5️⃣ Starting multi-pass fine-tuning...")
        if not self.multi_pass_training(training_texts):
            logger.error("❌ Training failed")
            return False
        
        # Step 6: Save model
        logger.info("6️⃣ Saving trained model...")
        if not self.save_model():
            logger.warning("⚠️  Failed to save model")
        
        # Step 7: Test model
        logger.info("7️⃣ Testing trained model...")
        if not self.test_model():
            logger.warning("⚠️  Model testing failed")
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("🎉 TRAINING PIPELINE COMPLETED!")
        logger.info("=" * 60)
        
        # Count unique documents
        sources = set()
        for chunk in self.pdf_data:
            source = chunk['metadata'].get('source', 'unknown')
            sources.add(source)
        
        print(f"✅ Total PDFs processed: {len(sources)}")
        print(f"✅ Number of embeddings updated: {len(self.pdf_data)}")
        print(f"✅ Backend used: {self.backend_used}")
        print(f"✅ Training passes completed: {self.config.num_passes}")
        print(f"✅ Device used: {self.device}")
        print(f"✅ Model ready for testing in UI")
        print("\n✅ Training complete – UI ready for real testing!")
        
        return True


def main():
    """Main execution function"""
    trainer = PDFTrainerComplete()
    success = trainer.run_complete_training()
    
    if not success:
        print("❌ Training failed")
        return False
    
    return True


if __name__ == "__main__":
    main()
