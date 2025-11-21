"""Unit tests for Reinforcement Learning"""

import pytest
import torch
import numpy as np
from src.rl.environment import LegalTaskEnvironment, LegalTaskType
from src.rl.agent import PPOAgent, DQNAgent, CustomPolicyNetwork
from src.rl.rewards import RewardCalculator, RewardMetrics


class TestLegalTaskEnvironment:
    """Test suite for Legal Task Environment"""
    
    def test_environment_initialization(self):
        """Test environment initialization"""
        from src.mamba.model import MambaModel
        from src.mamba.tokenizer import DocumentTokenizer
        
        model = MambaModel(vocab_size=1000, d_model=128, num_layers=2, num_heads=4)
        tokenizer = DocumentTokenizer(vocab_size=1000)
        
        env = LegalTaskEnvironment(
            model=model,
            tokenizer=tokenizer,
            task_type=LegalTaskType.SUMMARIZATION
        )
        
        assert env.task_type == LegalTaskType.SUMMARIZATION
        assert env.action_space is not None
        assert env.observation_space is not None
    
    def test_environment_reset(self):
        """Test environment reset"""
        from src.mamba.model import MambaModel
        from src.mamba.tokenizer import DocumentTokenizer
        
        model = MambaModel(vocab_size=1000, d_model=128, num_layers=2, num_heads=4)
        tokenizer = DocumentTokenizer(vocab_size=1000)
        
        env = LegalTaskEnvironment(model=model, tokenizer=tokenizer)
        
        observation, info = env.reset()
        
        assert observation.shape == env.observation_space.shape
        assert isinstance(info, dict)
        assert env.current_step == 0
    
    def test_environment_step(self):
        """Test environment step"""
        from src.mamba.model import MambaModel
        from src.mamba.tokenizer import DocumentTokenizer
        
        model = MambaModel(vocab_size=1000, d_model=128, num_layers=2, num_heads=4)
        tokenizer = DocumentTokenizer(vocab_size=1000)
        
        env = LegalTaskEnvironment(
            model=model,
            tokenizer=tokenizer,
            task_type=LegalTaskType.SUMMARIZATION
        )
        
        env.reset()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        assert observation.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_classification_environment(self):
        """Test classification task environment"""
        from src.mamba.model import MambaModel
        from src.mamba.tokenizer import DocumentTokenizer
        
        model = MambaModel(vocab_size=1000, d_model=128, num_layers=2, num_heads=4)
        tokenizer = DocumentTokenizer(vocab_size=1000)
        
        env = LegalTaskEnvironment(
            model=model,
            tokenizer=tokenizer,
            task_type=LegalTaskType.DOCUMENT_CLASSIFICATION
        )
        
        observation, info = env.reset(options={'document': 'Test document', 'target': 2})
        action = 2  # Predict class 2
        observation, reward, terminated, truncated, info = env.step(action)
        
        assert terminated  # Should be done after one step for classification


class TestRewardCalculator:
    """Test suite for Reward Calculator"""
    
    def test_calculator_initialization(self):
        """Test calculator initialization"""
        calculator = RewardCalculator()
        
        assert calculator.reward_weights is not None
        assert 'accuracy' in calculator.reward_weights
    
    def test_summarization_reward(self):
        """Test summarization reward calculation"""
        calculator = RewardCalculator()
        
        generated = "This is a summary of the legal document."
        reference = "This is a summary of legal content."
        source = "This is the full legal document with lots of details."
        
        reward_metrics = calculator.calculate_summarization_reward(
            generated_summary=generated,
            reference_summary=reference,
            source_document=source
        )
        
        assert isinstance(reward_metrics, RewardMetrics)
        assert isinstance(reward_metrics.total_reward, float)
        assert 'rouge' in reward_metrics.component_rewards
    
    def test_qa_reward(self):
        """Test QA reward calculation"""
        calculator = RewardCalculator()
        
        predicted = "John Smith"
        reference = "John Smith"
        context = "The case involves John Smith as the plaintiff."
        question = "Who is the plaintiff?"
        
        reward_metrics = calculator.calculate_qa_reward(
            predicted_answer=predicted,
            reference_answer=reference,
            context=context,
            question=question
        )
        
        assert isinstance(reward_metrics, RewardMetrics)
        assert reward_metrics.total_reward > 0  # Exact match should give positive reward
    
    def test_classification_reward(self):
        """Test classification reward calculation"""
        calculator = RewardCalculator()
        
        # Correct prediction
        reward_correct = calculator.calculate_classification_reward(
            predicted_class=1,
            true_class=1,
            confidence=0.9
        )
        
        assert reward_correct.total_reward > 0
        
        # Incorrect prediction
        reward_incorrect = calculator.calculate_classification_reward(
            predicted_class=1,
            true_class=2,
            confidence=0.9
        )
        
        assert reward_incorrect.total_reward < reward_correct.total_reward
    
    def test_rouge_computation(self):
        """Test ROUGE score computation"""
        calculator = RewardCalculator()
        
        generated = "the quick brown fox"
        reference = "the quick brown fox"
        
        rouge = calculator._compute_rouge(generated, reference)
        
        assert rouge == 1.0  # Perfect match
    
    def test_f1_computation(self):
        """Test F1 score computation"""
        calculator = RewardCalculator()
        
        predicted = "the quick brown fox"
        reference = "the quick brown fox"
        
        f1 = calculator._compute_f1(predicted, reference)
        
        assert f1 == 1.0  # Perfect match


class TestCustomPolicyNetwork:
    """Test suite for Custom Policy Network"""
    
    def test_network_initialization(self):
        """Test network initialization"""
        network = CustomPolicyNetwork(
            input_dim=512,
            action_dim=100,
            hidden_dims=[256, 128]
        )
        
        assert network.input_dim == 512
        assert network.action_dim == 100
    
    def test_forward_pass(self):
        """Test forward pass"""
        network = CustomPolicyNetwork(
            input_dim=512,
            action_dim=100
        )
        
        batch_size = 4
        x = torch.randn(batch_size, 512)
        
        logits, value = network(x)
        
        assert logits.shape == (batch_size, 100)
        assert value.shape == (batch_size, 1)
    
    def test_get_action(self):
        """Test action selection"""
        network = CustomPolicyNetwork(
            input_dim=512,
            action_dim=100
        )
        
        batch_size = 4
        x = torch.randn(batch_size, 512)
        
        action, log_prob, value = network.get_action(x, deterministic=False)
        
        assert action.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        assert value.shape == (batch_size, 1)
    
    def test_deterministic_action(self):
        """Test deterministic action selection"""
        import random
        
        network = CustomPolicyNetwork(
            input_dim=512,
            action_dim=100
        )
        network.eval()  # Set to eval mode
        
        # Use same input for both calls
        torch.manual_seed(42)
        random.seed(42)
        x = torch.randn(1, 512)
        
        # In deterministic mode, should select argmax of logits
        with torch.no_grad():
            action1, _, _ = network.get_action(x, deterministic=True)
            action2, _, _ = network.get_action(x, deterministic=True)
        
        # Deterministic should give same action for same input
        assert action1.item() == action2.item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU for RL agents")
class TestRLAgents:
    """Test suite for RL Agents (requires GPU)"""
    
    def test_ppo_agent_creation(self):
        """Test PPO agent creation"""
        from src.mamba.model import MambaModel
        from src.mamba.tokenizer import DocumentTokenizer
        
        model = MambaModel(vocab_size=1000, d_model=128, num_layers=2, num_heads=4)
        tokenizer = DocumentTokenizer(vocab_size=1000)
        
        env = LegalTaskEnvironment(model=model, tokenizer=tokenizer)
        agent = PPOAgent(env, device="cpu")
        
        assert agent.agent is not None
    
    def test_dqn_agent_creation(self):
        """Test DQN agent creation"""
        from src.mamba.model import MambaModel
        from src.mamba.tokenizer import DocumentTokenizer
        
        model = MambaModel(vocab_size=1000, d_model=128, num_layers=2, num_heads=4)
        tokenizer = DocumentTokenizer(vocab_size=1000)
        
        env = LegalTaskEnvironment(model=model, tokenizer=tokenizer)
        agent = DQNAgent(env, device="cpu")
        
        assert agent.agent is not None


def test_reward_shaping():
    """Test reward shaping"""
    calculator = RewardCalculator(use_shaped_rewards=True)
    
    base_reward = 1.0
    current_state = {'progress': 0.5}
    next_state = {'progress': 0.6}
    
    shaped_reward = calculator.add_reward_shaping(
        base_reward,
        current_state,
        next_state
    )
    
    # Should add progress bonus
    assert shaped_reward > base_reward


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
