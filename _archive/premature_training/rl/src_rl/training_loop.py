"""
RL Training Loop with Proper Rewards
- Penalize hallucinations
- Reward correct document usage
- Auto-retrain every 20 incorrect answers
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class TrainingExample:
    """Training example for RLHF"""
    question: str
    wrong_answer: str
    correct_answer: Optional[str]
    retrieved_docs: List[str]
    timestamp: str
    reward: float
    
class RLTrainingLoop:
    """
    RL Training Loop with proper reward structure
    
    Rewards:
    - Correct answer from docs: +1.0
    - Partially correct: +0.5
    - Hallucination: -1.0
    - Incomplete: -0.3
    """
    
    def __init__(self, buffer_path: str = "checkpoints/rlhf/training_buffer.json"):
        self.buffer_path = Path(buffer_path)
        self.buffer_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.training_buffer: List[TrainingExample] = []
        self.incorrect_count = 0
        self.retrain_threshold = 20
        
        self._load_buffer()
    
    def add_incorrect_answer(
        self,
        question: str,
        wrong_answer: str,
        retrieved_docs: List[str],
        correct_answer: Optional[str] = None
    ):
        """
        Add incorrect answer to training buffer
        
        Args:
            question: User query
            wrong_answer: Model's incorrect answer
            retrieved_docs: Retrieved document contents
            correct_answer: Correct answer if known
        """
        # Calculate reward (penalty for wrong answer)
        reward = self._calculate_reward(wrong_answer, retrieved_docs, correct_answer)
        
        example = TrainingExample(
            question=question,
            wrong_answer=wrong_answer,
            correct_answer=correct_answer,
            retrieved_docs=retrieved_docs,
            timestamp=datetime.now().isoformat(),
            reward=reward
        )
        
        self.training_buffer.append(example)
        self.incorrect_count += 1
        
        logger.info(f"Added training example. Buffer size: {len(self.training_buffer)}")
        
        # Save buffer
        self._save_buffer()
        
        # Check if should retrain
        if self.incorrect_count >= self.retrain_threshold:
            self.trigger_retraining()
    
    def add_correct_answer(
        self,
        question: str,
        answer: str,
        retrieved_docs: List[str]
    ):
        """Add positive example for reinforcement"""
        reward = 1.0  # Full reward for correct answer
        
        example = TrainingExample(
            question=question,
            wrong_answer="",  # Not wrong
            correct_answer=answer,
            retrieved_docs=retrieved_docs,
            timestamp=datetime.now().isoformat(),
            reward=reward
        )
        
        self.training_buffer.append(example)
        self._save_buffer()
        
        logger.info(f"Added positive example. Buffer size: {len(self.training_buffer)}")
    
    def _calculate_reward(
        self,
        answer: str,
        retrieved_docs: List[str],
        correct_answer: Optional[str]
    ) -> float:
        """
        Calculate reward/penalty
        
        Penalties:
        - Hallucination (not in docs): -1.0
        - Incomplete answer: -0.3
        - Nonsensical: -0.8
        
        Rewards:
        - Uses document correctly: +0.5
        - Matches correct answer: +1.0
        """
        if not answer or len(answer.strip()) < 10:
            return -0.3  # Incomplete
        
        # Check for hallucination
        if self._is_hallucination(answer, retrieved_docs):
            logger.warning(f"Hallucination detected: {answer[:100]}")
            return -1.0
        
        # Check if answer uses documents
        if self._uses_documents_correctly(answer, retrieved_docs):
            base_reward = 0.5
            
            # Bonus if matches correct answer
            if correct_answer and self._answers_match(answer, correct_answer):
                return 1.0
            
            return base_reward
        
        # Default penalty for poor answer
        return -0.5
    
    def _is_hallucination(self, answer: str, retrieved_docs: List[str]) -> bool:
        """Check if answer contains hallucinated content"""
        if not retrieved_docs:
            return False
        
        # Extract key phrases from answer
        answer_words = set(answer.lower().split())
        
        # Check if at least 30% of content words are in retrieved docs
        doc_words = set()
        for doc in retrieved_docs:
            doc_words.update(doc.lower().split())
        
        common_words = answer_words & doc_words
        
        # If less than 30% overlap, likely hallucination
        if len(common_words) / max(len(answer_words), 1) < 0.3:
            return True
        
        return False
    
    def _uses_documents_correctly(self, answer: str, retrieved_docs: List[str]) -> bool:
        """Check if answer properly uses retrieved documents"""
        if not retrieved_docs:
            return False
        
        # Check for citations or proper references
        answer_lower = answer.lower()
        
        citation_markers = [
            'according to', 'based on', 'as per', 'the act states',
            'the document', 'section', 'provision'
        ]
        
        has_citation = any(marker in answer_lower for marker in citation_markers)
        
        return has_citation
    
    def _answers_match(self, answer: str, correct_answer: str) -> bool:
        """Check if answer matches correct answer"""
        answer_words = set(answer.lower().split())
        correct_words = set(correct_answer.lower().split())
        
        overlap = len(answer_words & correct_words)
        similarity = overlap / max(len(correct_words), 1)
        
        return similarity > 0.7
    
    def trigger_retraining(self):
        """Trigger model retraining"""
        logger.info(f"Triggering retraining with {len(self.training_buffer)} examples")
        
        # Prepare training data
        training_data = self._prepare_training_data()
        
        # Save training data
        training_file = self.buffer_path.parent / "training_data.json"
        with open(training_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"Training data saved to {training_file}")
        
        # Reset counter
        self.incorrect_count = 0
        
        # TODO: Call actual training script
        # subprocess.run(['python', 'scripts/train_rlhf.py'])
    
    def _prepare_training_data(self) -> List[Dict[str, Any]]:
        """Prepare training data in format for RLHF"""
        training_data = []
        
        for example in self.training_buffer:
            # Create training pair
            training_pair = {
                'prompt': example.question,
                'context': '\n'.join(example.retrieved_docs[:3]),
                'completion': example.correct_answer or "Not available in documents.",
                'reward': example.reward,
                'timestamp': example.timestamp
            }
            training_data.append(training_pair)
        
        return training_data
    
    def _save_buffer(self):
        """Save training buffer to disk"""
        buffer_data = [asdict(ex) for ex in self.training_buffer]
        
        with open(self.buffer_path, 'w') as f:
            json.dump(buffer_data, f, indent=2)
        
        logger.debug(f"Buffer saved: {len(buffer_data)} examples")
    
    def _load_buffer(self):
        """Load training buffer from disk"""
        if self.buffer_path.exists():
            with open(self.buffer_path, 'r') as f:
                buffer_data = json.load(f)
            
            self.training_buffer = [
                TrainingExample(**ex) for ex in buffer_data
            ]
            
            logger.info(f"Loaded {len(self.training_buffer)} examples from buffer")
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get training buffer statistics"""
        if not self.training_buffer:
            return {'total': 0, 'avg_reward': 0.0}
        
        total = len(self.training_buffer)
        avg_reward = sum(ex.reward for ex in self.training_buffer) / total
        
        positive_examples = sum(1 for ex in self.training_buffer if ex.reward > 0)
        negative_examples = sum(1 for ex in self.training_buffer if ex.reward < 0)
        
        return {
            'total': total,
            'positive': positive_examples,
            'negative': negative_examples,
            'avg_reward': avg_reward,
            'incorrect_count': self.incorrect_count,
            'next_retrain': self.retrain_threshold - self.incorrect_count
        }
