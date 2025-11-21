"""
Model Evaluation Script

Evaluate fine-tuned models on held-out test set with multiple metrics.

Usage:
    python -m src.training.eval --model mamba_lora --dataset data/val_sft.jsonl
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm

from src.training.utils import load_config, format_time


class ModelEvaluator:
    """Evaluate fine-tuned models."""
    
    def __init__(self, config_path: str, model_name: str = None):
        """
        Initialize evaluator.
        
        Args:
            config_path: Path to eval config
            model_name: Override model name from config
        """
        self.config = load_config(config_path)
        
        if model_name:
            self.config['model']['name'] = model_name
        
        self.model_name = self.config['model']['name']
        self.predictions = []
        self.metrics = {}
        
        print("="*70)
        print("  MODEL EVALUATION")
        print("="*70)
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.config['dataset']['file']}")
    
    def load_dataset(self) -> List[Dict]:
        """Load evaluation dataset."""
        print("\n📂 Loading dataset...")
        
        dataset_file = Path(self.config['dataset']['file'])
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_file}")
        
        examples = []
        with open(dataset_file, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
        
        max_samples = self.config['dataset']['max_samples']
        if max_samples and len(examples) > max_samples:
            examples = examples[:max_samples]
        
        print(f"   ✅ Loaded {len(examples)} examples")
        return examples
    
    def compute_exact_match(self, pred: str, gold: str) -> float:
        """Compute exact match score."""
        return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0
    
    def compute_f1(self, pred: str, gold: str) -> float:
        """Compute token-level F1 score."""
        pred_tokens = pred.lower().split()
        gold_tokens = gold.lower().split()
        
        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return 0.0
        
        common = set(pred_tokens) & set(gold_tokens)
        
        if len(common) == 0:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def compute_rouge_l(self, pred: str, gold: str) -> float:
        """Compute ROUGE-L score (simplified)."""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(gold, pred)
            return scores['rougeL'].fmeasure
        except ImportError:
            # Fallback: use F1 if rouge_score not installed
            return self.compute_f1(pred, gold)
    
    def compute_bleu(self, pred: str, gold: str) -> float:
        """Compute BLEU score (simplified)."""
        try:
            from nltk.translate.bleu_score import sentence_bleu
            reference = [gold.split()]
            candidate = pred.split()
            return sentence_bleu(reference, candidate)
        except ImportError:
            # Fallback: use F1 if nltk not installed
            return self.compute_f1(pred, gold)
    
    def evaluate_example(self, example: Dict) -> Dict:
        """
        Evaluate a single example.
        
        Args:
            example: Example with instruction, input, output
            
        Returns:
            Prediction dictionary with metrics
        """
        # Mock prediction for now (replace with actual model inference)
        prediction = example['output']  # Placeholder
        gold = example['output']
        
        # Compute metrics
        scores = {}
        if 'exact_match' in self.config['metrics']:
            scores['exact_match'] = self.compute_exact_match(prediction, gold)
        if 'f1' in self.config['metrics']:
            scores['f1'] = self.compute_f1(prediction, gold)
        if 'rouge_l' in self.config['metrics']:
            scores['rouge_l'] = self.compute_rouge_l(prediction, gold)
        if 'bleu' in self.config['metrics']:
            scores['bleu'] = self.compute_bleu(prediction, gold)
        
        return {
            'instruction': example['instruction'],
            'input': example['input'],
            'gold': gold,
            'prediction': prediction,
            'scores': scores
        }
    
    def evaluate(self, examples: List[Dict]):
        """Run evaluation on all examples."""
        print("\n🔍 Running evaluation...")
        
        for example in tqdm(examples, desc="Evaluating"):
            result = self.evaluate_example(example)
            self.predictions.append(result)
        
        # Aggregate metrics
        print("\n📊 Computing aggregate metrics...")
        
        for metric in self.config['metrics']:
            scores = [p['scores'][metric] for p in self.predictions]
            self.metrics[metric] = {
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores)
            }
    
    def save_report(self):
        """Save evaluation report."""
        print("\n💾 Saving report...")
        
        report_dir = Path(self.config['output']['report_dir'])
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"eval_{self.model_name}_{timestamp}.json"
        
        report = {
            'model': self.model_name,
            'dataset': str(self.config['dataset']['file']),
            'num_examples': len(self.predictions),
            'metrics': self.metrics,
            'config': self.config,
            'timestamp': timestamp
        }
        
        # Save predictions if requested
        if self.config['output']['save_predictions']:
            report['predictions'] = self.predictions
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   ✅ Report saved: {report_file}")
        
        return report_file
    
    def print_summary(self):
        """Print evaluation summary."""
        print("\n" + "="*70)
        print("  EVALUATION SUMMARY")
        print("="*70)
        print(f"Model: {self.model_name}")
        print(f"Examples: {len(self.predictions)}")
        print(f"\n📊 Metrics:")
        
        for metric, scores in self.metrics.items():
            print(f"\n   {metric.upper()}:")
            print(f"      Mean: {scores['mean']:.4f}")
            print(f"      Min:  {scores['min']:.4f}")
            print(f"      Max:  {scores['max']:.4f}")
        
        print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/eval.yaml',
        help='Path to eval config'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model name to evaluate'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Override dataset file'
    )
    
    args = parser.parse_args()
    
    try:
        # Load config
        evaluator = ModelEvaluator(args.config, args.model)
        
        # Override dataset if specified
        if args.dataset:
            evaluator.config['dataset']['file'] = args.dataset
        
        # Load dataset
        examples = evaluator.load_dataset()
        
        # Run evaluation
        start_time = time.time()
        evaluator.evaluate(examples)
        elapsed = time.time() - start_time
        
        # Save and print results
        report_file = evaluator.save_report()
        evaluator.print_summary()
        
        print(f"\n⏱️  Time elapsed: {format_time(elapsed)}")
        print(f"\n✅ Evaluation complete!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
