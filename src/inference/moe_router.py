"""
Pure MoE Router for MARK System.
Routes queries to the best available Hugging Face expert model.
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import yaml

from src.core.model_registry import get_registry, ExpertInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MoE-Router")

LOG_FILE = Path("logs/model_routing.json")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

class MoERouter:
    def __init__(self, config_path: str = "configs/moe_experts.yaml"):
        self.registry = get_registry()
        self.config_path = config_path
        self.cache = {}

    def route_for_ui(self, text: str, task: str = "qa") -> Dict[str, Any]:
        """
        Lightweight routing for UI.
        
        Args:
            text: Input text
            task: Task hint
            
        Returns:
            Dict with chosen expert, score, etc.
        """
        cache_key = f"{hash(text)}_{task}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        routes = self.route(text, task_hint=task, top_k=1)
        if not routes:
            return {"chosen": None, "reason": "No expert found"}
            
        selected = routes[0]
        result = {
            "chosen": selected["expert_name"],
            "scores": {r["expert_name"]: r["score"] for r in routes},
            "reason": selected["reason"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Simple cache eviction if too large
        if len(self.cache) > 1000:
            self.cache.clear()
            
        self.cache[cache_key] = result
        return result
        
    def route(self, text: str, task_hint: str = None, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Route the input text to the best experts.
        
        Args:
            text: Input query or document text
            task_hint: Optional task type (e.g., 'qa', 'ner')
            top_k: Number of experts to return
            
        Returns:
            List of dicts with expert info and score
        """
        experts = self.registry.list_experts()
        scores = []
        
        doc_len = len(text.split())
        
        for expert in experts:
            score = expert.priority_score * 10.0
            
            # 1. Task Match
            if task_hint and task_hint in expert.task_types:
                score += 5.0
            
            # 2. Token Window Fit
            # If doc is huge, prefer larger window models (LLaMA)
            if doc_len > 2048 and expert.token_window >= 4096:
                score += 3.0
            elif doc_len < 512 and expert.token_window <= 512:
                # Efficient for short texts
                score += 1.0
                
            # 3. Keyword Heuristics
            lower_text = text.lower()
            if "judgment" in lower_text or "order" in lower_text:
                if "judgment-prediction" in expert.task_types or "qa" in expert.task_types:
                    score += 2.0
            
            if "section" in lower_text or "act" in lower_text:
                 if "ner" in expert.task_types:
                     score += 2.0

            scores.append({
                "expert_name": expert.name,
                "model_id": expert.model_id,
                "score": score,
                "reason": f"Base: {expert.priority_score*10}, Task: {task_hint in expert.task_types if task_hint else 'N/A'}"
            })
            
        # Sort by score descending
        scores.sort(key=lambda x: x["score"], reverse=True)
        selected = scores[:top_k]
        
        self._log_decision(text, selected, task_hint)
        return selected

    def _log_decision(self, text: str, selected: List[Dict], task_hint: str):
        """Log routing decision to JSON file"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "input_hash": hash(text),
            "input_len": len(text),
            "task_hint": task_hint,
            "selected_experts": selected
        }
        
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")

def main():
    parser = argparse.ArgumentParser(description="MARK MoE Router CLI")
    parser.add_argument("--text", type=str, required=True, help="Input text to route")
    parser.add_argument("--task", type=str, help="Optional task hint")
    parser.add_argument("--dry-run", action="store_true", help="Simulate routing only")
    
    args = parser.parse_args()
    
    router = MoERouter()
    print(f"Routing input: '{args.text}' (Task: {args.task})")
    
    results = router.route(args.text, args.task)
    
    print("\nSelected Experts:")
    for res in results:
        print(f"- {res['expert_name']} (Score: {res['score']:.2f})")
        print(f"  Model: {res['model_id']}")
        print(f"  Reason: {res['reason']}")

if __name__ == "__main__":
    main()
