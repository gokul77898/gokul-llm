"""Quickstart example for Transfer Learning"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from src.common import seed_everything, init_logger, load_config
from src.data import create_sample_data, load_transfer_dataset


def main():
    print("="*70)
    print("TRANSFER LEARNING QUICKSTART - 1 Epoch Training")
    print("="*70)
    
    # Create sample data
    print("\n1. Creating sample data...")
    create_sample_data()
    
    # Load config
    print("2. Loading configuration...")
    config = load_config("configs/transfer_train.yaml")
    config.training.num_epochs = 1
    config.training.batch_size = 4
    config.data.max_samples = 20
    config.training.num_workers = 0
    config.system.device = "cpu"
    
    seed_everything(42)
    logger = init_logger("transfer_quickstart")
    
    # Load data
    print("3. Loading datasets...")
    train_dataset, val_dataset = load_transfer_dataset(config)
    print(f"   Train samples: {len(train_dataset)}")
    
    # Create tokenizer and model
    print("4. Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3
    )
    print(f"   Model loaded: bert-base-uncased")
    
    # Create dataloader
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        labels = [item['label'] for item in batch]
        
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
        encodings['labels'] = torch.tensor(labels)
        return encodings
    
    train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn)
    
    # Training
    print("5. Training for 1 epoch...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)
    
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()
        
        if batch_idx % 2 == 0:
            print(f"   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"\n6. Training completed!")
    print(f"   Average Loss: {avg_loss:.4f}")
    
    # Save checkpoint
    checkpoint_path = Path("checkpoints/transfer") / "quickstart_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 0,
        'loss': avg_loss,
    }, checkpoint_path)
    
    print(f"\n7. Checkpoint saved to: {checkpoint_path}")
    print("\n" + "="*70)
    print("✓ Transfer learning quickstart completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
