"""
Unified Model Registry for MARK MoE System - Phase 0: HF Inference API Only

All model inference goes through Hugging Face Inference API.
No local model downloads. Requires HF_TOKEN environment variable.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import yaml
import requests

logger = logging.getLogger(__name__)

# HF Inference API base URL
HF_API_URL = "https://api-inference.huggingface.co/models"

def _get_hf_token() -> str:
    """Get HF token from environment variable"""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable required for Hugging Face Inference API")
    return token


@dataclass
class ExpertInfo:
    """Information about a registered expert model"""
    name: str
    model_id: str
    task_types: List[str]
    token_window: int = 512
    tuning: str = "none"
    lora_config: Dict[str, Any] = field(default_factory=dict)
    priority_score: float = 1.0
    role: str = "encoder"  # encoder or decoder


class ModelRegistry:
    """Central registry for all MARK MoE experts"""
    
    def __init__(self, config_path: str = "configs/moe_experts.yaml"):
        self.experts: Dict[str, ExpertInfo] = {}
        self.config_path = config_path
        self._load_registry()
    
    def _load_registry(self):
        """Load experts from YAML config"""
        path = Path(self.config_path)
        if not path.exists():
            logger.warning(f"Config file {path} not found. Registry empty.")
            return

        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        for expert_cfg in config.get('experts', []):
            expert = ExpertInfo(
                name=expert_cfg['name'],
                model_id=expert_cfg['model_id'],
                task_types=expert_cfg['task_types'],
                token_window=expert_cfg.get('token_window', 512),
                tuning=expert_cfg.get('tuning', 'none'),
                lora_config=expert_cfg.get('lora', {}),
                priority_score=expert_cfg.get('priority_score', 1.0),
                role=expert_cfg.get('role', 'encoder')
            )
            self.experts[expert.name] = expert
            logger.info(f"Registered expert: {expert.name} ({expert.model_id}) [role={expert.role}]")

    def get_expert(self, name: str) -> Optional[ExpertInfo]:
        """Get info for a specific expert"""
        return self.experts.get(name)

    def list_experts(self) -> List[ExpertInfo]:
        """List all registered experts"""
        return list(self.experts.values())

    def get_experts_by_task(self, task: str) -> List[ExpertInfo]:
        """Find experts suitable for a specific task"""
        return [e for e in self.experts.values() if task in e.task_types]

    def get_encoders(self) -> List[ExpertInfo]:
        """Get all encoder experts"""
        return [e for e in self.experts.values() if e.role == "encoder"]

    def get_decoders(self) -> List[ExpertInfo]:
        """Get all decoder experts"""
        return [e for e in self.experts.values() if e.role == "decoder"]


# Global registry instance
_registry = None


def get_registry() -> ModelRegistry:
    """Get the global model registry (lazy init)"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


class HFInferenceClient:
    """Client for Hugging Face Inference API - no local model downloads"""
    
    def __init__(self):
        self.token = _get_hf_token()
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    def call_encoder(self, model_id: str, text: str, task: str = "ner") -> Dict[str, Any]:
        """
        Call encoder model via HF Inference API.
        Returns extracted entities/classifications.
        """
        url = f"{HF_API_URL}/{model_id}"
        
        # For NER/token classification
        if task in ["ner", "entity-extraction"]:
            payload = {"inputs": text}
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                return {"success": True, "entities": response.json(), "model_id": model_id}
            else:
                return {"success": False, "error": response.text, "status_code": response.status_code}
        
        # For classification
        elif task in ["classification", "case-classification"]:
            payload = {"inputs": text}
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                return {"success": True, "classification": response.json(), "model_id": model_id}
            else:
                return {"success": False, "error": response.text, "status_code": response.status_code}
        
        # For embeddings/similarity
        elif task in ["similarity", "embeddings"]:
            payload = {"inputs": text}
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                return {"success": True, "embeddings": response.json(), "model_id": model_id}
            else:
                return {"success": False, "error": response.text, "status_code": response.status_code}
        
        return {"success": False, "error": f"Unknown task: {task}"}
    
    def call_decoder(self, model_id: str, prompt: str, encoder_output: Dict[str, Any], max_tokens: int = 256) -> Dict[str, Any]:
        """
        Call decoder model via HF Inference API.
        STRICT CONTRACT: Decoder MUST receive encoder output in prompt.
        """
        # Validate encoder output is present
        if not encoder_output or not encoder_output.get("success"):
            return {
                "success": False,
                "refusal": True,
                "error": "REFUSAL: Decoder requires valid encoder output. Cannot generate without extracted facts."
            }
        
        # Build strict prompt with encoder facts
        encoder_facts = self._format_encoder_output(encoder_output)
        
        strict_prompt = f"""You are a legal AI assistant. You MUST ONLY use the following extracted facts to answer.
If the facts are insufficient, say "I cannot answer based on the provided information."

=== EXTRACTED FACTS (from encoder) ===
{encoder_facts}
=== END FACTS ===

User Query: {prompt}

Answer based ONLY on the facts above:"""
        
        url = f"{HF_API_URL}/{model_id}"
        payload = {
            "inputs": strict_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        response = requests.post(url, headers=self.headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result[0].get("generated_text", "") if isinstance(result, list) else result.get("generated_text", "")
            return {
                "success": True,
                "generated_text": generated_text,
                "model_id": model_id,
                "encoder_facts_used": encoder_facts
            }
        else:
            return {"success": False, "error": response.text, "status_code": response.status_code}
    
    def _format_encoder_output(self, encoder_output: Dict[str, Any]) -> str:
        """Format encoder output as facts for decoder prompt"""
        facts = []
        
        if "entities" in encoder_output:
            entities = encoder_output["entities"]
            if isinstance(entities, list):
                for ent in entities[:20]:  # Limit to 20 entities
                    if isinstance(ent, dict):
                        facts.append(f"- Entity: {ent.get('word', ent.get('entity', 'unknown'))} ({ent.get('entity_group', ent.get('label', 'unknown'))})")
                    else:
                        facts.append(f"- {ent}")
        
        if "classification" in encoder_output:
            classification = encoder_output["classification"]
            if isinstance(classification, list) and classification:
                for cls in classification[:5]:
                    if isinstance(cls, dict):
                        facts.append(f"- Classification: {cls.get('label', 'unknown')} (score: {cls.get('score', 0):.2f})")
        
        if not facts:
            facts.append("- No specific entities or classifications extracted")
        
        return "\n".join(facts)


# Global HF client instance
_hf_client = None


def get_hf_client() -> HFInferenceClient:
    """Get the global HF Inference client (lazy init)"""
    global _hf_client
    if _hf_client is None:
        _hf_client = HFInferenceClient()
    return _hf_client
