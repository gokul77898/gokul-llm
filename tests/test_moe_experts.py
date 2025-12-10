"""
Tests for MARK MoE System
"""

import pytest
from src.core.model_registry import get_registry
from src.inference.moe_router import MoERouter

def test_registry_loading():
    registry = get_registry()
    experts = registry.list_experts()
    assert len(experts) > 0
    assert registry.get_expert("inlegalllama") is not None

def test_router_logic():
    router = MoERouter()
    
    # Test QA routing
    results = router.route("What is the penalty for murder?", task_hint="qa")
    assert results[0]["expert_name"] == "inlegalllama"
    
    # Test NER routing (short text + keywords)
    results = router.route("Section 302 of the IPC", task_hint="ner")
    # Priority logic might favor inlegalbert or opennyai-ner
    top_expert = results[0]["expert_name"]
    assert top_expert in ["inlegalbert", "opennyai-ner"]

def test_expert_config():
    registry = get_registry()
    expert = registry.get_expert("inlegalllama")
    assert expert.tuning == "lora"
    assert expert.lora_config["r"] == 16
