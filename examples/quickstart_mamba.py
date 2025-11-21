"""Quickstart example for Mamba training"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.common import load_config, seed_everything, init_logger
from src.data import create_sample_data, load_mamba_dataset
from src.mamba import DocumentTokenizer, MambaModel
from torch.utils.data import DataLoader


def main():
    print("="*70)
    print("MAMBA QUICKSTART - 1 Epoch Training")
    print("="*70)
    
    # Create sample data
    print("\n1. Creating sample data...")
    create_sample_data()
    
    # Load config
    print("2. Loading configuration...")
    config = load_config("configs/mamba_train.yaml")
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.data.max_samples = 20
    config.training.num_workers = 0
    config.system.device = "cpu"
    
    seed_everything(42)
    logger = init_logger("mamba_quickstart")
    
    # Load data
    print("3. Loading datasets...")
    train_dataset, val_dataset = load_mamba_dataset(config)
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Create tokenizer
    print("4. Creating tokenizer...")
    tokenizer = DocumentTokenizer(vocab_size=500, max_length=128)
    train_texts = [item['text'] for item in train_dataset]
    tokenizer.build_vocab(train_texts)
    print(f"   Vocabulary size: {len(tokenizer.token2id)}")
    
    # Create dataloader
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        encodings = [tokenizer.encode(text, return_tensors=False) for text in texts]
        
        max_len = min(max(len(enc['input_ids']) for enc in encodings), 128)
        input_ids = []
        
        for enc in encodings:
            ids = enc['input_ids'][:max_len]
            padding = max_len - len(ids)
            ids = ids + [tokenizer.pad_token_id] * padding
            input_ids.append(ids)
        
        return {'input_ids': torch.tensor(input_ids)}
    
    train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)
    
    # Create model
    print("5. Creating model...")
    model = MambaModel(
        vocab_size=len(tokenizer.token2id),
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_seq_length=128,
        num_classes=3
    )
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    print("6. Training for 1 epoch...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids']
        
        outputs = model(input_ids, task="generation")
        logits = outputs['logits']
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            ignore_index=tokenizer.pad_token_id
        )
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()
        
        if batch_idx % 5 == 0:
            print(f"   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"\n7. Training completed!")
    print(f"   Average Loss: {avg_loss:.4f}")
    
    # Save checkpoint
    checkpoint_path = Path("checkpoints/mamba") / "quickstart_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 0,
        'loss': avg_loss,
    }, checkpoint_path)
    
    print(f"\n8. Checkpoint saved to: {checkpoint_path}")
    print("\n" + "="*70)
    print("✓ Mamba quickstart completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
