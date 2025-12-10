"""
Unified Generator Interface for MARK MoE System.
Calls MoE Router to select expert, then generates response.
"""

import logging
from typing import Dict, Any, Optional
import torch

from src.inference.moe_router import MoERouter
from src.core.model_registry import load_expert_model

logger = logging.getLogger(__name__)

# Global cache for loaded experts to avoid reloading
LOADED_EXPERTS = {}

class Generator:
    def __init__(self):
        self.router = MoERouter()
        
    def generate(self, query: str, model_key: str = "auto", top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Generate response for a query.
        
        Args:
            query: Input text
            model_key: "auto" for MoE routing, or specific expert name
            top_k: Top K sampling params (if applicable)
            
        Returns:
            Dict containing answer, model used, etc.
        """
        
        # 1. Determine Model
        if model_key == "auto" or model_key == "moe":
            task_hint = kwargs.get("task")
            routes = self.router.route(query, task_hint=task_hint, top_k=1)
            if not routes:
                raise RuntimeError("No suitable expert found by router.")
            
            selected = routes[0]
            expert_name = selected["expert_name"]
            logger.info(f"Routed to expert: {expert_name} (Score: {selected['score']:.2f})")
        else:
            expert_name = model_key
            logger.info(f"Using requested expert: {expert_name}")
            
        # 2. Load Model (with caching)
        global LOADED_EXPERTS
        if expert_name not in LOADED_EXPERTS:
            logger.info(f"Loading expert {expert_name} into memory...")
            LOADED_EXPERTS[expert_name] = load_expert_model(expert_name)
            
        model, tokenizer = LOADED_EXPERTS[expert_name]
        
        # 3. Task-Specific Generation
        try:
            device = next(model.parameters()).device
            inputs = tokenizer(query, return_tensors="pt").to(device)
            task_hint = kwargs.get("task", "qa")  # Default to QA if no task specified
            
            # ------------------------------------------
            # CLASSIFICATION (InLegalBERT, InCaseLawBERT)
            # ------------------------------------------
            if task_hint in ["classification", "case-classification"]:
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    confidence, label_idx = torch.max(probs, dim=-1)
                    
                response_text = f"Classification Result: Label {int(label_idx)} (Confidence: {float(confidence):.3f})"
                confidence_score = float(confidence)
                
            # ------------------------------------------
            # NAMED ENTITY RECOGNITION (NER)
            # ------------------------------------------
            elif task_hint == "ner":
                with torch.no_grad():
                    outputs = model(**inputs)
                    if hasattr(outputs, 'logits') and len(outputs.logits.shape) > 2:
                        # Token-level classification (true NER)
                        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                        label_ids = outputs.logits.argmax(dim=-1)[0].tolist()
                        
                        entities = []
                        for token, label in zip(tokens, label_ids):
                            if token not in ['[CLS]', '[SEP]', '[PAD]'] and not token.startswith('##'):
                                entities.append(f"{token}:{label}")
                        
                        response_text = f"Entities: {', '.join(entities[:10])}"  # Limit output
                        confidence_score = 0.85
                    else:
                        # Fallback for sequence classification models
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=-1)
                        confidence, label_idx = torch.max(probs, dim=-1)
                        response_text = f"NER Classification: Label {int(label_idx)} (Confidence: {float(confidence):.3f})"
                        confidence_score = float(confidence)
                        
            # ------------------------------------------
            # GENERATIVE MODELS (Gemma-legal, etc.)
            # ------------------------------------------
            elif task_hint in ["qa", "summarization", "generation", "reasoning"] and hasattr(model, "generate"):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=kwargs.get("max_length", 256),
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove prompt echo
                if response_text.startswith(query):
                    response_text = response_text[len(query):].strip()
                confidence_score = 0.9
                
            # ------------------------------------------
            # FALLBACK
            # ------------------------------------------
            else:
                if hasattr(model, "generate"):
                    # Try generation anyway
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, 
                            max_new_tokens=kwargs.get("max_length", 256),
                            do_sample=True,
                            top_k=top_k
                        )
                    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if response_text.startswith(query):
                        response_text = response_text[len(query):].strip()
                    confidence_score = 0.8
                else:
                    # Classification fallback
                    with torch.no_grad():
                        outputs = model(**inputs)
                    response_text = f"[Model Output] Logits: {outputs.logits.shape if hasattr(outputs, 'logits') else 'No logits'}"
                    confidence_score = 0.7

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            response_text = f"Error generating with {expert_name}: {str(e)}"
            confidence_score = 0.0

        return {
            "answer": response_text,
            "model": expert_name,
            "confidence": confidence_score,
            "metadata": {
                "expert_used": expert_name,
                "routing_score": selected["score"] if model_key in ["auto", "moe"] else "manual",
                "task_type": task_hint
            }
        }

    def generate_with_expert(self, expert_name: str, text: str, max_new_tokens: int = 256) -> str:
        """
        UI-friendly generation function.
        
        Args:
            expert_name: Name of the expert to use
            text: Input text
            max_new_tokens: Max tokens to generate
            
        Returns:
            Generated text string
        """
        global LOADED_EXPERTS
        
        # Load if not cached
        if expert_name not in LOADED_EXPERTS:
            logger.info(f"Loading expert {expert_name} for UI generation...")
            LOADED_EXPERTS[expert_name] = load_expert_model(expert_name)
            
        model, tokenizer = LOADED_EXPERTS[expert_name]
        device = next(model.parameters()).device
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        try:
            if hasattr(model, "generate"):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7
                    )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Clean up prompt echo
                if response.startswith(text):
                    response = response[len(text):].strip()
                return response
            else:
                return f"Model {expert_name} does not support generation."
        except Exception as e:
            logger.error(f"UI Generation failed: {e}")
            return f"Error: {str(e)}"
