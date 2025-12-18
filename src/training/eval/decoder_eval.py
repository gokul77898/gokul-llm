"""
Phase 3.5: Decoder Evaluation

Evaluation functions for decoder (SFT) models.
Computes: fact-adherence, refusal correctness.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DecoderEvalResult:
    """Result of decoder evaluation."""
    fact_adherence: float
    refusal_precision: float
    refusal_recall: float
    refusal_f1: float
    overall_accuracy: float
    total_samples: int
    samples_with_facts: int
    adherent_samples: int
    true_refusals: int
    false_refusals: int
    missed_refusals: int
    correct_answers: int


class DecoderEvaluator:
    """
    Evaluator for decoder (instruction-following) models.
    
    Computes fact-adherence and refusal correctness metrics.
    """
    
    def __init__(self, refusal_token: str = "REFUSE:"):
        """
        Initialize evaluator.
        
        Args:
            refusal_token: Token that indicates a refusal response
        """
        self.refusal_token = refusal_token
    
    def _is_refusal(self, text: str) -> bool:
        """Check if response is a refusal."""
        return text.strip().upper().startswith(self.refusal_token.upper())
    
    def _extract_facts_from_prompt(self, prompt: str) -> List[str]:
        """
        Extract facts from ENCODER_FACTS block in prompt.
        
        Returns list of fact strings (sections, entities).
        """
        facts = []
        
        # Extract sections (e.g., "Section 420")
        sections = re.findall(r"Section\s+\d+[A-Za-z]?", prompt, re.IGNORECASE)
        facts.extend(sections)
        
        # Extract content between ENCODER_FACTS markers
        match = re.search(
            r"ENCODER_FACTS[:\s]*\n?(.*?)(?:QUESTION|$)",
            prompt,
            re.DOTALL | re.IGNORECASE
        )
        if match:
            facts_block = match.group(1)
            # Extract bullet points
            bullets = re.findall(r"[-•]\s*(.+)", facts_block)
            facts.extend([b.strip() for b in bullets if len(b.strip()) > 3])
        
        return facts
    
    def compute_fact_adherence(
        self,
        outputs: List[str],
        prompts: List[str],
        encoder_facts: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Compute fact adherence: how often decoder output mentions encoder facts.
        
        Args:
            outputs: List of decoder outputs
            prompts: List of prompts (to extract facts if encoder_facts not provided)
            encoder_facts: Optional pre-extracted facts per sample
        
        Returns:
            Dict with fact_adherence metrics
        """
        adherent_count = 0
        total_with_facts = 0
        
        for idx, output in enumerate(outputs):
            # Skip refusals - they don't need to mention facts
            if self._is_refusal(output):
                continue
            
            # Get facts for this sample
            if encoder_facts and idx < len(encoder_facts):
                facts = encoder_facts[idx]
            else:
                facts = self._extract_facts_from_prompt(prompts[idx])
            
            if not facts:
                continue
            
            total_with_facts += 1
            output_lower = output.lower()
            
            # Check if any fact is mentioned
            for fact in facts:
                if fact.lower() in output_lower:
                    adherent_count += 1
                    break
        
        adherence = adherent_count / total_with_facts if total_with_facts > 0 else 1.0
        
        return {
            "fact_adherence": adherence,
            "adherent_samples": adherent_count,
            "total_samples_with_facts": total_with_facts
        }
    
    def compute_refusal_correctness(
        self,
        outputs: List[str],
        should_refuse: List[bool]
    ) -> Dict[str, Any]:
        """
        Compute refusal correctness metrics.
        
        Args:
            outputs: List of decoder outputs
            should_refuse: List of booleans indicating if refusal was expected
        
        Returns:
            Dict with refusal metrics
        """
        true_refusals = 0
        false_refusals = 0
        missed_refusals = 0
        correct_answers = 0
        
        for output, expected_refusal in zip(outputs, should_refuse):
            is_refusal = self._is_refusal(output)
            
            if expected_refusal and is_refusal:
                true_refusals += 1
            elif expected_refusal and not is_refusal:
                missed_refusals += 1
            elif not expected_refusal and is_refusal:
                false_refusals += 1
            else:
                correct_answers += 1
        
        total = len(outputs)
        
        # Refusal precision: true_refusals / (true_refusals + false_refusals)
        refusal_precision = true_refusals / (true_refusals + false_refusals) if (true_refusals + false_refusals) > 0 else 1.0
        
        # Refusal recall: true_refusals / (true_refusals + missed_refusals)
        refusal_recall = true_refusals / (true_refusals + missed_refusals) if (true_refusals + missed_refusals) > 0 else 1.0
        
        # F1
        refusal_f1 = 2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall) if (refusal_precision + refusal_recall) > 0 else 0.0
        
        # Overall accuracy
        accuracy = (true_refusals + correct_answers) / total if total > 0 else 0.0
        
        return {
            "refusal_precision": refusal_precision,
            "refusal_recall": refusal_recall,
            "refusal_f1": refusal_f1,
            "overall_accuracy": accuracy,
            "true_refusals": true_refusals,
            "false_refusals": false_refusals,
            "missed_refusals": missed_refusals,
            "correct_answers": correct_answers
        }
    
    def evaluate(
        self,
        outputs: List[str],
        prompts: List[str],
        should_refuse: List[bool],
        encoder_facts: Optional[List[List[str]]] = None
    ) -> DecoderEvalResult:
        """
        Run full evaluation.
        
        Args:
            outputs: List of decoder outputs
            prompts: List of prompts
            should_refuse: List of expected refusal flags
            encoder_facts: Optional pre-extracted facts per sample
        
        Returns:
            DecoderEvalResult with all metrics
        """
        fact_metrics = self.compute_fact_adherence(outputs, prompts, encoder_facts)
        refusal_metrics = self.compute_refusal_correctness(outputs, should_refuse)
        
        return DecoderEvalResult(
            fact_adherence=fact_metrics["fact_adherence"],
            refusal_precision=refusal_metrics["refusal_precision"],
            refusal_recall=refusal_metrics["refusal_recall"],
            refusal_f1=refusal_metrics["refusal_f1"],
            overall_accuracy=refusal_metrics["overall_accuracy"],
            total_samples=len(outputs),
            samples_with_facts=fact_metrics["total_samples_with_facts"],
            adherent_samples=fact_metrics["adherent_samples"],
            true_refusals=refusal_metrics["true_refusals"],
            false_refusals=refusal_metrics["false_refusals"],
            missed_refusals=refusal_metrics["missed_refusals"],
            correct_answers=refusal_metrics["correct_answers"]
        )
    
    def evaluate_from_file(
        self,
        predictions_file: str,
        references_file: str
    ) -> DecoderEvalResult:
        """
        Evaluate from JSONL files.
        
        Predictions format per line:
        {"output": "..."}
        
        References format per line:
        {"prompt": "...", "should_refuse": true/false, "encoder_facts": [...]}
        """
        outputs = []
        prompts = []
        should_refuse = []
        encoder_facts = []
        
        with open(predictions_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                outputs.append(data["output"])
        
        with open(references_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                prompts.append(data.get("prompt", ""))
                should_refuse.append(data.get("should_refuse", False))
                encoder_facts.append(data.get("encoder_facts", []))
        
        return self.evaluate(outputs, prompts, should_refuse, encoder_facts)


def create_decoder_evaluator(refusal_token: str = "REFUSE:") -> DecoderEvaluator:
    """Factory function to create a DecoderEvaluator."""
    return DecoderEvaluator(refusal_token)


if __name__ == "__main__":
    # Demo evaluation with synthetic data
    print("=" * 50)
    print("Decoder Evaluator Demo")
    print("=" * 50)
    
    evaluator = create_decoder_evaluator()
    
    # Synthetic data
    outputs = [
        "Section 420 IPC deals with cheating and fraud.",  # Correct answer with fact
        "REFUSE: Missing facts.",  # Correct refusal
        "The law is complex.",  # Incorrect - no fact mentioned
        "REFUSE: Cannot answer.",  # False refusal
        "Section 302 covers murder.",  # Correct answer with fact
    ]
    
    prompts = [
        "ENCODER_FACTS:\n- Section 420\nQUESTION: What is Section 420?",
        "ENCODER_FACTS:\n(none)\nQUESTION: What is the law?",
        "ENCODER_FACTS:\n- Section 420\nQUESTION: What is Section 420?",
        "ENCODER_FACTS:\n- Section 302\nQUESTION: What is Section 302?",
        "ENCODER_FACTS:\n- Section 302\nQUESTION: What is Section 302?",
    ]
    
    should_refuse = [False, True, False, False, False]
    
    result = evaluator.evaluate(outputs, prompts, should_refuse)
    
    print(f"\nFact Adherence Metrics:")
    print(f"  Fact Adherence: {result.fact_adherence:.3f}")
    print(f"  Adherent Samples: {result.adherent_samples}/{result.samples_with_facts}")
    
    print(f"\nRefusal Metrics:")
    print(f"  Precision: {result.refusal_precision:.3f}")
    print(f"  Recall: {result.refusal_recall:.3f}")
    print(f"  F1: {result.refusal_f1:.3f}")
    
    print(f"\nOverall:")
    print(f"  Accuracy: {result.overall_accuracy:.3f}")
    print(f"  True Refusals: {result.true_refusals}")
    print(f"  False Refusals: {result.false_refusals}")
    print(f"  Missed Refusals: {result.missed_refusals}")
    print(f"  Correct Answers: {result.correct_answers}")
