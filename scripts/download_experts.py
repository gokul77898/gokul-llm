"""
Script to automatically download MoE experts from Hugging Face.
Reads configs/moe_experts.yaml and downloads models to ./experts/
"""

import argparse
import os
import yaml
import logging
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PATH = "configs/moe_experts.yaml"
EXPERTS_DIR = Path("experts")

def load_config():
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config file not found: {CONFIG_PATH}")
        return None
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def download_expert(expert_config, force=False):
    name = expert_config['name']
    model_id = expert_config['model_id']
    target_dir = EXPERTS_DIR / name
    
    logger.info(f"Checking expert: {name} ({model_id})")
    
    # Special handling for gemma-legal: remove partial downloads
    if name == "gemma-legal" and target_dir.exists():
        logger.info("Removing partial Gemma folder to avoid corrupted downloads...")
        shutil.rmtree(target_dir)
    
    # Check if already exists (simple check for safetensors or pytorch_model.bin)
    if target_dir.exists() and not force:
        has_weights = any(target_dir.glob("*.safetensors")) or any(target_dir.glob("*.bin"))
        if has_weights:
            logger.info(f"✅ Expert {name} already exists in {target_dir}. Skipping.")
            return True

    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    logger.info(f"⬇️ Downloading {name} from {model_id}...")
    logger.info("Downloading only safetensors + tokenizer files (skipping .gguf)")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN"),
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "*.model",
                "*.txt",
                "*.md"
            ],
            ignore_patterns=[
                "*.gguf"
            ]
        )
        logger.info(f"✅ Successfully downloaded {name} to {target_dir}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download {name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download MoE Experts")
    parser.add_argument("--all", action="store_true", help="Download all experts defined in config")
    parser.add_argument("--expert", type=str, help="Download only a specific expert")
    parser.add_argument("--only", type=str, help="Alias for --expert")
    parser.add_argument("--force", action="store_true", help="Force redownload even if exists")
    
    args = parser.parse_args()
    
    # Normalize --only to --expert
    if args.only:
        args.expert = args.only
    
    config = load_config()
    if not config:
        return

    EXPERTS_DIR.mkdir(parents=True, exist_ok=True)
    
    experts = config.get('experts', [])
    downloaded_count = 0
    total_count = 0

    if args.expert:
        # Download single expert
        target = next((e for e in experts if e['name'] == args.expert), None)
        if target:
            total_count = 1
            if download_expert(target, args.force):
                downloaded_count = 1
        else:
            logger.error(f"Expert '{args.expert}' not found in configuration.")
    elif args.all:
        # Download all
        total_count = len(experts)
        for expert in experts:
            if download_expert(expert, args.force):
                downloaded_count += 1
    else:
        logger.info("Please specify --all or --expert <NAME>")
        return

    logger.info("-" * 40)
    logger.info(f"Summary: {downloaded_count}/{total_count} experts ready.")
    logger.info(f"Models located in: {EXPERTS_DIR.absolute()}")

if __name__ == "__main__":
    main()
