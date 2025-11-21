"""Build and evaluate RAG indexer"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common import load_config, seed_everything, init_logger
from src.data.datasets import create_sample_data
from src.rag.indexer import index_documents, load_index
from src.rag.eval import evaluate_rag_pipeline


def main():
    parser = argparse.ArgumentParser(description="Build RAG indexer")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate after indexing")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    seed_everything(config.system.seed)
    
    # Create directories
    Path(config.paths.index_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = init_logger("rag_indexer")
    logger.info(f"Building RAG index with config: {args.config}")
    
    # Create sample data if needed
    if not Path(config.data.documents_file).exists():
        logger.info("Creating sample data...")
        create_sample_data()
    
    # Build index
    logger.info("Building FAISS index...")
    indexer = index_documents(
        config.data.documents_file,
        config.paths.index_path,
        config.paths.metadata_path,
        config
    )
    
    logger.info(f"Index built and saved to {config.paths.index_path}")
    
    # Evaluate if requested
    if args.evaluate:
        eval_file = getattr(config, 'evaluation', None)
        if eval_file and hasattr(eval_file, 'eval_file'):
            eval_file = eval_file.eval_file
        else:
            eval_file = "data/rag_eval.jsonl"
        
        if not Path(eval_file).exists():
            logger.warning(f"Evaluation file not found: {eval_file}")
            logger.info("Creating sample evaluation data...")
            create_sample_data()
        
        if Path(eval_file).exists():
            logger.info("Evaluating index...")
            try:
                metrics = evaluate_rag_pipeline(indexer, eval_file)
                logger.info("Evaluation completed successfully")
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
        else:
            logger.warning("Skipping evaluation (no eval file)")
    
    logger.info("Indexing completed!")


if __name__ == "__main__":
    main()
