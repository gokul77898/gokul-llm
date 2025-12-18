"""Safe SFT Training CLI"""
import json, logging, argparse
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--base_checkpoint', default='checkpoints/rlhf/sft/sft_final.pt')
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--dry_run', type=bool, default=False)
    args = parser.parse_args()
    
    logger.info(f"SFT Training: {args.data} -> {args.out}")
    
    # Load data
    with open(args.data) as f:
        examples = [json.loads(line) for line in f if line.strip()]
    
    logger.info(f"Loaded {len(examples)} examples")
    
    if args.dry_run:
        logger.info("DRY RUN - validation would run here")
        return
    
    # Placeholder: actual training would go here
    logger.warning("Full SFT training not yet implemented - use existing RLHF pipeline")
    logger.info(f"Data ready at: {args.data}")
    logger.info(f"Output checkpoint: {args.out}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
