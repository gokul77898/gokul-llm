"""Safe Checkpoint Promotion CLI"""
import argparse, shutil, logging
from pathlib import Path

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from', dest='from_path', required=True)
    parser.add_argument('--to', required=True)
    args = parser.parse_args()
    
    src = Path(args.from_path)
    dst = Path(f"checkpoints/training/{args.to}/model_final.pt")
    
    if not src.exists():
        logger.error(f"Source not found: {src}")
        return
    
    logger.info(f"Promoting: {src} -> {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    logger.info("✅ Checkpoint promoted successfully")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
