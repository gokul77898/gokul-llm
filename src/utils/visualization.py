"""Visualization utilities"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import torch


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """
    Plot training history.
    
    Args:
        history: Dictionary with metrics history
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot training loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
    
    # Plot accuracy
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Accuracy')
        if 'val_f1' in history:
            axes[1].plot(history['val_f1'], label='F1 Score')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Validation Metrics')
        axes[1].legend()
        axes[1].grid(True)
    
    # Plot learning rate
    if 'learning_rate' in history:
        axes[2].plot(history['learning_rate'])
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_attention_weights(
    attention_weights: torch.Tensor,
    tokens: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    """
    Plot attention weights heatmap.
    
    Args:
        attention_weights: Attention weights [seq_len, seq_len]
        tokens: List of tokens
        save_path: Optional path to save figure
        figsize: Figure size
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        cbar=True
    )
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.title('Attention Weights')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6)
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Optional path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_document_embeddings(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = 'tsne',
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    """
    Plot document embeddings in 2D using dimensionality reduction.
    
    Args:
        embeddings: Document embeddings [num_docs, embedding_dim]
        labels: Optional labels for coloring
        method: Reduction method ('tsne', 'pca', 'umap')
        save_path: Optional path to save figure
        figsize: Figure size
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # Reduce to 2D
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=figsize)
    
    if labels is not None:
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6
        )
        plt.colorbar(scatter, label='Label')
    else:
        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.6
        )
    
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'Document Embeddings ({method.upper()})')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_rl_rewards(
    rewards: List[float],
    window_size: int = 100,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5)
):
    """
    Plot RL training rewards with moving average.
    
    Args:
        rewards: List of episode rewards
        window_size: Window size for moving average
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Raw rewards
    axes[0].plot(rewards, alpha=0.3, label='Episode Reward')
    
    # Moving average
    if len(rewards) >= window_size:
        moving_avg = np.convolve(
            rewards,
            np.ones(window_size) / window_size,
            mode='valid'
        )
        axes[0].plot(
            range(window_size - 1, len(rewards)),
            moving_avg,
            label=f'Moving Avg ({window_size})',
            linewidth=2
        )
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Rewards')
    axes[0].legend()
    axes[0].grid(True)
    
    # Reward distribution
    axes[1].hist(rewards, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(rewards), color='red', linestyle='--', label='Mean')
    axes[1].set_xlabel('Reward')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Reward Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
