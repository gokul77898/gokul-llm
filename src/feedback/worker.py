"""Feedback Processing Worker - Builds SFT Training Buffer"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class FeedbackWorker:
    """Process user feedback and build safe SFT training buffer"""
    
    SFT_BUFFER_THRESHOLD = 20  # Trigger SFT when buffer reaches this size
    HOURS_SINCE_LAST_SFT = 24  # Or trigger after 24 hours
    
    def __init__(self):
        """Initialize feedback worker"""
        self.feedback_dir = Path("feedback")
        self.feedback_dir.mkdir(exist_ok=True)
        
        self.incoming_path = self.feedback_dir / "incoming.jsonl"
        self.sft_buffer_path = self.feedback_dir / "sft_buffer.jsonl"
        self.review_queue_path = self.feedback_dir / "review_queue.jsonl"
        self.processed_path = self.feedback_dir / "processed.jsonl"
        self.last_sft_path = self.feedback_dir / "last_sft_timestamp.txt"
        
        logger.info("FeedbackWorker initialized")
    
    def save_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """
        Save incoming feedback atomically
        
        Args:
            feedback_data: Feedback from /feedback endpoint
            
        Returns:
            True if saved successfully
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in feedback_data:
                feedback_data['timestamp'] = datetime.now().isoformat()
            
            # Atomic write
            with open(self.incoming_path, 'a') as f:
                f.write(json.dumps(feedback_data) + '\n')
            
            logger.info(f"Feedback saved: query={feedback_data.get('query', '')[:50]}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            return False
    
    def process_feedback_batch(self) -> Dict[str, int]:
        """
        Process incoming feedback and route to appropriate queues
        
        Returns:
            Stats dict with counts
        """
        if not self.incoming_path.exists():
            return {'processed': 0, 'sft_added': 0, 'review_added': 0}
        
        stats = {'processed': 0, 'sft_added': 0, 'review_added': 0, 'dropped': 0}
        
        try:
            # Read all incoming feedback
            with open(self.incoming_path, 'r') as f:
                feedback_entries = [json.loads(line) for line in f if line.strip()]
            
            for entry in feedback_entries:
                stats['processed'] += 1
                
                # Validate and route
                if self._should_add_to_sft(entry):
                    self._add_to_sft_buffer(entry)
                    stats['sft_added'] += 1
                elif self._needs_human_review(entry):
                    self._add_to_review_queue(entry)
                    stats['review_added'] += 1
                else:
                    stats['dropped'] += 1
                
                # Archive to processed
                self._archive_feedback(entry)
            
            # Clear incoming file
            self.incoming_path.unlink()
            
            logger.info(f"Processed feedback batch: {stats}")
            
            # Check if SFT should be triggered
            if self._should_trigger_sft():
                logger.warning(f"SFT buffer threshold reached: {self._get_buffer_size()} examples")
            
            return stats
        
        except Exception as e:
            logger.error(f"Feedback processing failed: {e}")
            return stats
    
    def _should_add_to_sft(self, entry: Dict[str, Any]) -> bool:
        """Check if feedback should go to SFT buffer"""
        # Has user-provided corrected answer
        if entry.get('user_corrected_answer') and len(entry['user_corrected_answer'].strip()) > 10:
            return True
        
        # Flagged as incorrect with high confidence original answer
        if entry.get('flagged_incorrect') and entry.get('confidence', 0) > 0.7:
            return True
        
        return False
    
    def _needs_human_review(self, entry: Dict[str, Any]) -> bool:
        """Check if feedback needs human review"""
        # Flagged but no correction provided
        if entry.get('flagged_incorrect') and not entry.get('user_corrected_answer'):
            return True
        
        # Low confidence but flagged
        if entry.get('confidence', 1.0) < 0.4:
            return True
        
        return False
    
    def _add_to_sft_buffer(self, entry: Dict[str, Any]):
        """Add validated entry to SFT training buffer"""
        try:
            # Build SFT training example
            sft_example = self._build_sft_example(entry)
            
            with open(self.sft_buffer_path, 'a') as f:
                f.write(json.dumps(sft_example) + '\n')
            
            logger.info(f"Added to SFT buffer: {sft_example['prompt'][:50]}...")
        
        except Exception as e:
            logger.error(f"Failed to add to SFT buffer: {e}")
    
    def _build_sft_example(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Build SFT training example from feedback"""
        query = entry.get('query', '')
        corrected_answer = entry.get('user_corrected_answer', '')
        sources = entry.get('sources', [])
        
        # Build context from sources
        context_parts = []
        for source in sources[:3]:
            content = source.get('content', '')
            if content:
                context_parts.append(content[:200])
        
        context = "\n".join(context_parts) if context_parts else "No context available"
        
        # Build legal-style prompt
        prompt = (
            f"Question: {query} Provide the answer with section citation.\n"
            f"Context: {context}\n"
            f"Answer:"
        )
        
        # Use corrected answer or build from sources
        response = corrected_answer if corrected_answer else self._extract_answer_from_sources(query, sources)
        
        return {
            'prompt': prompt,
            'response': response,
            'metadata': {
                'source': 'user_feedback',
                'user_id': entry.get('user_id', 'anonymous'),
                'timestamp': entry.get('timestamp', datetime.now().isoformat()),
                'original_answer': entry.get('answer', ''),
                'model': entry.get('auto_model_used', 'unknown')
            }
        }
    
    def _extract_answer_from_sources(self, query: str, sources: List[dict]) -> str:
        """Extract answer from sources when no user correction provided"""
        if not sources:
            return "Answer not available in sources"
        
        # Use top source content
        top_source = sources[0]
        content = top_source.get('content', '')
        doc_id = top_source.get('doc_id', 'unknown')
        page = top_source.get('page', 'N/A')
        
        return f"Based on {doc_id} (Page {page}): {content}"
    
    def _add_to_review_queue(self, entry: Dict[str, Any]):
        """Add entry to human review queue"""
        try:
            with open(self.review_queue_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            
            logger.info(f"Added to review queue: {entry.get('query', '')[:50]}")
        
        except Exception as e:
            logger.error(f"Failed to add to review queue: {e}")
    
    def _archive_feedback(self, entry: Dict[str, Any]):
        """Archive processed feedback"""
        try:
            with open(self.processed_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to archive feedback: {e}")
    
    def _get_buffer_size(self) -> int:
        """Get current SFT buffer size"""
        if not self.sft_buffer_path.exists():
            return 0
        
        with open(self.sft_buffer_path, 'r') as f:
            return sum(1 for line in f if line.strip())
    
    def _should_trigger_sft(self) -> bool:
        """Check if SFT training should be triggered"""
        buffer_size = self._get_buffer_size()
        
        # Check size threshold
        if buffer_size >= self.SFT_BUFFER_THRESHOLD:
            return True
        
        # Check time threshold
        if self.last_sft_path.exists():
            with open(self.last_sft_path, 'r') as f:
                last_sft_time = datetime.fromisoformat(f.read().strip())
            
            hours_since = (datetime.now() - last_sft_time).total_seconds() / 3600
            if hours_since >= self.HOURS_SINCE_LAST_SFT and buffer_size > 0:
                return True
        
        return False
    
    def create_sft_bundle(self) -> str:
        """
        Create SFT training bundle from buffer
        
        Returns:
            Path to bundle file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bundle_path = self.feedback_dir / f"sft_bundle_{timestamp}.jsonl"
        
        try:
            # Copy buffer to bundle
            if self.sft_buffer_path.exists():
                with open(self.sft_buffer_path, 'r') as src:
                    with open(bundle_path, 'w') as dst:
                        dst.write(src.read())
                
                # Update last SFT timestamp
                with open(self.last_sft_path, 'w') as f:
                    f.write(datetime.now().isoformat())
                
                # Clear buffer (move to archive)
                archive_path = self.feedback_dir / f"sft_buffer_archived_{timestamp}.jsonl"
                self.sft_buffer_path.rename(archive_path)
                
                logger.info(f"Created SFT bundle: {bundle_path}")
                return str(bundle_path)
        
        except Exception as e:
            logger.error(f"Failed to create SFT bundle: {e}")
            return ""
    
    def get_review_queue(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get items needing human review"""
        if not self.review_queue_path.exists():
            return []
        
        try:
            with open(self.review_queue_path, 'r') as f:
                items = [json.loads(line) for line in f if line.strip()]
            
            return items[:limit]
        
        except Exception as e:
            logger.error(f"Failed to get review queue: {e}")
            return []


def run_feedback_worker_loop():
    """Background worker loop"""
    worker = FeedbackWorker()
    
    logger.info("Starting feedback worker loop...")
    
    while True:
        try:
            stats = worker.process_feedback_batch()
            if stats['processed'] > 0:
                logger.info(f"Worker processed {stats['processed']} feedback entries")
            
            # Check if SFT should be triggered
            if worker._should_trigger_sft():
                bundle_path = worker.create_sft_bundle()
                if bundle_path:
                    logger.warning(f"⚠️  SFT training ready! Bundle: {bundle_path}")
                    logger.warning("Run: python -m src.training.sft_train --data " + bundle_path)
            
            # Sleep for 5 minutes
            time.sleep(300)
        
        except Exception as e:
            logger.error(f"Worker loop error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_feedback_worker_loop()
