"""
Mixture-of-Experts (MoE) Inference Router

Implements intelligent routing between different expert models based on query characteristics.
Integrates with existing Mamba/Transformer auto-detection system.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class ExpertType(Enum):
    """Types of expert models"""
    MAMBA_LONG_CONTEXT = "mamba_long"
    TRANSFORMER_GENERAL = "transformer_general"
    RAG_RETRIEVAL = "rag_retrieval"
    LEGAL_SPECIALIST = "legal_specialist"
    SUMMARIZATION = "summarization"
    QA_SPECIALIST = "qa_specialist"


@dataclass
class Expert:
    """Expert model configuration"""
    name: str
    expert_type: ExpertType
    model_instance: Any
    specialization: List[str]  # Keywords/domains this expert handles
    load_function: Optional[Callable] = None
    is_loaded: bool = False
    performance_score: float = 1.0
    usage_count: int = 0
    avg_latency: float = 0.0


@dataclass
class RoutingDecision:
    """Routing decision result"""
    selected_expert: Expert
    confidence: float
    reasoning: str
    fallback_experts: List[Expert]
    routing_time_ms: float


class QueryClassifier:
    """
    Classifies queries to determine the best expert
    
    Uses simple heuristics and keyword matching for efficiency.
    """
    
    def __init__(self):
        self.classification_rules = {
            ExpertType.MAMBA_LONG_CONTEXT: {
                "keywords": ["long", "document", "analyze", "summary", "detailed", "comprehensive"],
                "min_length": 500,
                "context_indicators": ["multiple pages", "entire document", "full text"]
            },
            ExpertType.RAG_RETRIEVAL: {
                "keywords": ["find", "search", "retrieve", "lookup", "reference", "cite"],
                "query_patterns": ["what does", "where is", "show me", "find information"]
            },
            ExpertType.LEGAL_SPECIALIST: {
                "keywords": ["law", "legal", "court", "judgment", "statute", "case", "ruling", "precedent"],
                "legal_terms": ["plaintiff", "defendant", "jurisdiction", "liability", "contract"]
            },
            ExpertType.SUMMARIZATION: {
                "keywords": ["summarize", "summary", "brief", "overview", "key points", "main ideas"],
                "query_patterns": ["give me a summary", "summarize this", "what are the key"]
            },
            ExpertType.QA_SPECIALIST: {
                "keywords": ["what", "how", "why", "when", "where", "who", "explain"],
                "question_indicators": ["?", "question", "answer"]
            }
        }
        
        logger.info("📊 Query Classifier initialized")
    
    def classify_query(
        self, 
        query: str, 
        context: str = "", 
        metadata: Dict[str, Any] = None
    ) -> Dict[ExpertType, float]:
        """
        Classify query and return confidence scores for each expert type
        
        Args:
            query: User query
            context: Additional context
            metadata: Query metadata
            
        Returns:
            Dictionary mapping expert types to confidence scores
        """
        metadata = metadata or {}
        query_lower = query.lower()
        context_lower = context.lower()
        combined_text = f"{query_lower} {context_lower}"
        
        scores = defaultdict(float)
        
        for expert_type, rules in self.classification_rules.items():
            score = 0.0
            
            # Keyword matching
            keywords = rules.get("keywords", [])
            keyword_matches = sum(1 for kw in keywords if kw in combined_text)
            score += keyword_matches * 0.2
            
            # Pattern matching
            patterns = rules.get("query_patterns", [])
            pattern_matches = sum(1 for pattern in patterns if pattern in combined_text)
            score += pattern_matches * 0.3
            
            # Length-based scoring
            min_length = rules.get("min_length", 0)
            if len(combined_text) >= min_length:
                score += 0.2
            
            # Special indicators
            if expert_type == ExpertType.MAMBA_LONG_CONTEXT:
                # Favor Mamba for long contexts
                if len(context) > 1000:
                    score += 0.4
                if metadata.get("page_count", 0) > 3:
                    score += 0.3
            
            elif expert_type == ExpertType.QA_SPECIALIST:
                # Favor QA for questions
                if "?" in query or any(q in query_lower for q in ["what", "how", "why"]):
                    score += 0.4
            
            scores[expert_type] = min(score, 1.0)  # Cap at 1.0
        
        return dict(scores)


class MoERouter:
    """
    Mixture-of-Experts Router
    
    Routes queries to the most appropriate expert model based on query characteristics.
    """
    
    def __init__(self):
        self.experts: Dict[str, Expert] = {}
        self.classifier = QueryClassifier()
        self.routing_history = []
        
        # Statistics
        self.stats = {
            "total_routes": 0,
            "expert_usage": defaultdict(int),
            "avg_routing_time_ms": 0.0,
            "accuracy_feedback": defaultdict(list)
        }
        
        # Initialize default experts
        self._initialize_default_experts()
        
        logger.info("🎯 MoE Router initialized")
        logger.info(f"   Available experts: {len(self.experts)}")
    
    def _initialize_default_experts(self):
        """Initialize default expert configurations"""
        
        # Mamba Long Context Expert
        self.experts["mamba_long"] = Expert(
            name="mamba_long",
            expert_type=ExpertType.MAMBA_LONG_CONTEXT,
            model_instance=None,
            specialization=["long_context", "document_analysis", "comprehensive_summary"],
            load_function=self._load_mamba_expert
        )
        
        # Transformer General Expert
        self.experts["transformer_general"] = Expert(
            name="transformer_general",
            expert_type=ExpertType.TRANSFORMER_GENERAL,
            model_instance=None,
            specialization=["general_qa", "short_context", "fast_response"],
            load_function=self._load_transformer_expert
        )
        
        # RAG Retrieval Expert
        self.experts["rag_retrieval"] = Expert(
            name="rag_retrieval",
            expert_type=ExpertType.RAG_RETRIEVAL,
            model_instance=None,
            specialization=["document_search", "information_retrieval", "citation"],
            load_function=self._load_rag_expert
        )
        
        # Legal Specialist Expert
        self.experts["legal_specialist"] = Expert(
            name="legal_specialist",
            expert_type=ExpertType.LEGAL_SPECIALIST,
            model_instance=None,
            specialization=["legal_analysis", "case_law", "statute_interpretation"],
            load_function=self._load_legal_expert
        )
        
        logger.info(f"✅ Initialized {len(self.experts)} default experts")
    
    def route_query(
        self,
        query: str,
        context: str = "",
        metadata: Dict[str, Any] = None,
        force_expert: Optional[str] = None
    ) -> RoutingDecision:
        """
        Route query to the most appropriate expert
        
        Args:
            query: User query
            context: Additional context
            metadata: Query metadata
            force_expert: Force routing to specific expert
            
        Returns:
            RoutingDecision with selected expert and reasoning
        """
        start_time = time.time()
        
        # Force routing if specified
        if force_expert and force_expert in self.experts:
            expert = self.experts[force_expert]
            return RoutingDecision(
                selected_expert=expert,
                confidence=1.0,
                reasoning=f"Forced routing to {force_expert}",
                fallback_experts=[],
                routing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Classify query
        classification_scores = self.classifier.classify_query(query, context, metadata)
        
        # Find best expert
        best_expert = None
        best_score = 0.0
        fallback_experts = []
        
        for expert_name, expert in self.experts.items():
            expert_score = classification_scores.get(expert.expert_type, 0.0)
            
            # Apply performance weighting
            weighted_score = expert_score * expert.performance_score
            
            # Apply availability penalty
            if not expert.is_loaded and expert.load_function is None:
                weighted_score *= 0.5  # Penalty for unavailable experts
            
            if weighted_score > best_score:
                if best_expert:
                    fallback_experts.append(best_expert)
                best_expert = expert
                best_score = weighted_score
            elif weighted_score > 0.3:  # Threshold for fallback consideration
                fallback_experts.append(expert)
        
        # Default fallback
        if best_expert is None:
            best_expert = self.experts.get("transformer_general")
            best_score = 0.5
            reasoning = "Default fallback to transformer_general"
        else:
            reasoning = f"Selected based on classification score: {best_score:.2f}"
        
        # Create routing decision
        routing_time = (time.time() - start_time) * 1000
        decision = RoutingDecision(
            selected_expert=best_expert,
            confidence=best_score,
            reasoning=reasoning,
            fallback_experts=fallback_experts[:3],  # Top 3 fallbacks
            routing_time_ms=routing_time
        )
        
        # Update statistics
        self.stats["total_routes"] += 1
        self.stats["expert_usage"][best_expert.name] += 1
        self.stats["avg_routing_time_ms"] = (
            self.stats["avg_routing_time_ms"] * 0.9 + routing_time * 0.1
        )
        
        # Store routing history
        self.routing_history.append({
            "query": query[:100],  # Truncate for storage
            "selected_expert": best_expert.name,
            "confidence": best_score,
            "timestamp": time.time()
        })
        
        # Keep only recent history
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
        
        logger.info(f"🎯 Routed to expert: {best_expert.name} (confidence: {best_score:.2f})")
        
        return decision
    
    def load_expert(self, expert_name: str) -> bool:
        """Load an expert model if not already loaded"""
        if expert_name not in self.experts:
            logger.error(f"Expert '{expert_name}' not found")
            return False
        
        expert = self.experts[expert_name]
        
        if expert.is_loaded:
            return True
        
        if expert.load_function is None:
            logger.warning(f"No load function for expert '{expert_name}'")
            return False
        
        try:
            logger.info(f"🔄 Loading expert: {expert_name}")
            expert.model_instance = expert.load_function()
            expert.is_loaded = expert.model_instance is not None
            
            if expert.is_loaded:
                logger.info(f"✅ Expert '{expert_name}' loaded successfully")
            else:
                logger.warning(f"⚠️  Expert '{expert_name}' failed to load")
            
            return expert.is_loaded
            
        except Exception as e:
            logger.error(f"❌ Failed to load expert '{expert_name}': {e}")
            return False
    
    def _load_mamba_expert(self):
        """Load Mamba expert model"""
        try:
            from src.core.mamba_loader import load_mamba_model
            
            model_instance = load_mamba_model()
            if model_instance.available:
                logger.info("✅ Mamba expert loaded")
                return model_instance
            else:
                logger.warning("⚠️  Mamba expert not available")
                return None
        except Exception as e:
            logger.error(f"Failed to load Mamba expert: {e}")
            return None
    
    def _load_transformer_expert(self):
        """Load Transformer expert model"""
        try:
            from src.core.model_registry import get_model_instance
            
            model_instance = get_model_instance("transformer")
            if model_instance:
                logger.info("✅ Transformer expert loaded")
                return model_instance
            else:
                logger.warning("⚠️  Transformer expert not available")
                return None
        except Exception as e:
            logger.error(f"Failed to load Transformer expert: {e}")
            return None
    
    def _load_rag_expert(self):
        """Load RAG expert model"""
        try:
            from src.rag.pipeline import RAGPipeline
            
            # This would load a pre-configured RAG pipeline
            # For now, return a placeholder
            logger.info("✅ RAG expert loaded (placeholder)")
            return "rag_pipeline_placeholder"
        except Exception as e:
            logger.error(f"Failed to load RAG expert: {e}")
            return None
    
    def _load_legal_expert(self):
        """Load Legal specialist expert model"""
        try:
            # This would load a legal-domain fine-tuned model
            # For now, use the general transformer
            from src.core.model_registry import get_model_instance
            
            model_instance = get_model_instance("transformer")
            if model_instance:
                logger.info("✅ Legal expert loaded (using transformer)")
                return model_instance
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to load Legal expert: {e}")
            return None
    
    def generate_with_expert(
        self,
        expert_name: str,
        query: str,
        context: str = "",
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate response using specific expert
        
        Args:
            expert_name: Name of expert to use
            query: User query
            context: Additional context
            **generation_kwargs: Generation parameters
            
        Returns:
            Generated response with metadata
        """
        if expert_name not in self.experts:
            return {"error": f"Expert '{expert_name}' not found"}
        
        expert = self.experts[expert_name]
        
        # Load expert if needed
        if not expert.is_loaded:
            if not self.load_expert(expert_name):
                return {"error": f"Failed to load expert '{expert_name}'"}
        
        try:
            start_time = time.time()
            
            # Generate based on expert type
            if expert.expert_type == ExpertType.MAMBA_LONG_CONTEXT:
                result = self._generate_with_mamba(expert, query, context, **generation_kwargs)
            elif expert.expert_type == ExpertType.TRANSFORMER_GENERAL:
                result = self._generate_with_transformer(expert, query, context, **generation_kwargs)
            elif expert.expert_type == ExpertType.RAG_RETRIEVAL:
                result = self._generate_with_rag(expert, query, context, **generation_kwargs)
            else:
                # Default to transformer
                result = self._generate_with_transformer(expert, query, context, **generation_kwargs)
            
            # Update expert statistics
            generation_time = time.time() - start_time
            expert.usage_count += 1
            expert.avg_latency = expert.avg_latency * 0.9 + generation_time * 0.1
            
            # Add metadata
            result.update({
                "expert_used": expert_name,
                "expert_type": expert.expert_type.value,
                "generation_time": generation_time,
                "expert_usage_count": expert.usage_count
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed with expert '{expert_name}': {e}")
            return {"error": str(e)}
    
    def _generate_with_mamba(self, expert: Expert, query: str, context: str, **kwargs) -> Dict[str, Any]:
        """Generate using Mamba expert"""
        try:
            model_instance = expert.model_instance
            
            if hasattr(model_instance, 'generate_with_state_space'):
                answer = model_instance.generate_with_state_space(
                    prompt=query,
                    context=context,
                    **kwargs
                )
            else:
                # Fallback generation
                answer = f"Mamba analysis of: {query}"
            
            return {
                "answer": answer,
                "model_type": "mamba",
                "backend": getattr(model_instance, 'backend', 'mamba')
            }
            
        except Exception as e:
            return {"error": f"Mamba generation failed: {e}"}
    
    def _generate_with_transformer(self, expert: Expert, query: str, context: str, **kwargs) -> Dict[str, Any]:
        """Generate using Transformer expert"""
        try:
            # Use existing generator
            from src.core.generator import generate_answer
            
            result = generate_answer(
                model_key="transformer",
                prompt=query,
                context=context,
                **kwargs
            )
            
            return result
            
        except Exception as e:
            return {"error": f"Transformer generation failed: {e}"}
    
    def _generate_with_rag(self, expert: Expert, query: str, context: str, **kwargs) -> Dict[str, Any]:
        """Generate using RAG expert"""
        try:
            # Use existing RAG pipeline
            # This is a placeholder - would use actual RAG pipeline
            return {
                "answer": f"RAG-based response for: {query}",
                "model_type": "rag",
                "retrieved_docs": []
            }
            
        except Exception as e:
            return {"error": f"RAG generation failed: {e}"}
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """Get statistics for all experts"""
        expert_stats = {}
        
        for name, expert in self.experts.items():
            expert_stats[name] = {
                "type": expert.expert_type.value,
                "is_loaded": expert.is_loaded,
                "usage_count": expert.usage_count,
                "avg_latency": expert.avg_latency,
                "performance_score": expert.performance_score,
                "specialization": expert.specialization
            }
        
        return {
            "experts": expert_stats,
            "routing_stats": self.stats,
            "total_experts": len(self.experts),
            "loaded_experts": sum(1 for e in self.experts.values() if e.is_loaded)
        }
    
    def update_expert_performance(self, expert_name: str, feedback_score: float):
        """Update expert performance based on feedback"""
        if expert_name in self.experts:
            expert = self.experts[expert_name]
            
            # Update performance score with exponential moving average
            expert.performance_score = expert.performance_score * 0.9 + feedback_score * 0.1
            
            # Store feedback
            self.stats["accuracy_feedback"][expert_name].append(feedback_score)
            
            logger.info(f"📊 Updated performance for {expert_name}: {expert.performance_score:.2f}")


# Factory functions
def create_moe_router() -> MoERouter:
    """Create MoE router with default configuration"""
    return MoERouter()


# Integration with existing system
def integrate_moe_routing():
    """
    Integration function for existing MARK AI system
    
    This can be called to add MoE routing to existing generation pipeline.
    """
    try:
        logger.info("🔗 Integrating MoE routing with existing system")
        
        router = create_moe_router()
        
        def enhanced_generate(
            query: str,
            context: str = "",
            metadata: Dict[str, Any] = None,
            force_expert: Optional[str] = None,
            **generation_kwargs
        ):
            """Enhanced generation with MoE routing"""
            
            # Route query to appropriate expert
            routing_decision = router.route_query(
                query=query,
                context=context,
                metadata=metadata,
                force_expert=force_expert
            )
            
            logger.info(f"🎯 MoE Routing Decision:")
            logger.info(f"   Expert: {routing_decision.selected_expert.name}")
            logger.info(f"   Confidence: {routing_decision.confidence:.2f}")
            logger.info(f"   Reasoning: {routing_decision.reasoning}")
            
            # Generate with selected expert
            result = router.generate_with_expert(
                expert_name=routing_decision.selected_expert.name,
                query=query,
                context=context,
                **generation_kwargs
            )
            
            # Add routing metadata
            result.update({
                "routing_decision": {
                    "expert": routing_decision.selected_expert.name,
                    "confidence": routing_decision.confidence,
                    "reasoning": routing_decision.reasoning,
                    "routing_time_ms": routing_decision.routing_time_ms
                }
            })
            
            return result
        
        return enhanced_generate, router
        
    except Exception as e:
        logger.error(f"⚠️  Could not integrate MoE routing: {e}")
        return None, None
