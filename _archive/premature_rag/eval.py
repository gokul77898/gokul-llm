"""Evaluation utilities for RAG system"""

import json
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    """
    Calculate Recall@K
    
    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Recall@K score
    """
    if not relevant_docs:
        return 0.0
    
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    
    hits = len(retrieved_set & relevant_set)
    return hits / len(relevant_set)


def mean_reciprocal_rank(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR)
    
    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs
        
    Returns:
        MRR score
    """
    relevant_set = set(relevant_docs)
    
    for i, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in relevant_set:
            return 1.0 / i
    
    return 0.0


def ndcg_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain @ K
    
    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        NDCG@K score
    """
    relevant_set = set(relevant_docs)
    
    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs[:k], 1):
        if doc_id in relevant_set:
            dcg += 1.0 / np.log2(i + 1)
    
    # Calculate IDCG (ideal DCG)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_docs), k)))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def evaluate_retrieval(
    retriever,
    eval_data: List[Dict[str, Any]],
    top_k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate retriever on evaluation dataset
    
    Args:
        retriever: Retriever object with search method
        eval_data: List of dicts with 'query' and 'relevant_docs'
        top_k_values: List of k values for recall@k
        
    Returns:
        Dictionary of evaluation metrics
    """
    all_recalls = defaultdict(list)
    all_mrrs = []
    all_ndcgs = defaultdict(list)
    
    for item in eval_data:
        query = item['query']
        relevant_docs = item['relevant_docs']
        
        # Retrieve documents
        results = retriever.search(query, k=max(top_k_values))
        retrieved_ids = [r.get('document', {}).get('id', f"doc_{r.get('index', -1)}") for r in results]
        
        # Calculate metrics
        for k in top_k_values:
            recall = recall_at_k(retrieved_ids, relevant_docs, k)
            all_recalls[k].append(recall)
            
            ndcg = ndcg_at_k(retrieved_ids, relevant_docs, k)
            all_ndcgs[k].append(ndcg)
        
        mrr = mean_reciprocal_rank(retrieved_ids, relevant_docs)
        all_mrrs.append(mrr)
    
    # Aggregate metrics
    metrics = {}
    for k in top_k_values:
        metrics[f'recall@{k}'] = np.mean(all_recalls[k])
        metrics[f'ndcg@{k}'] = np.mean(all_ndcgs[k])
    
    metrics['mrr'] = np.mean(all_mrrs)
    
    return metrics


def evaluate_rag_pipeline(
    indexer,
    eval_file: str,
    top_k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate RAG pipeline from evaluation file
    
    Args:
        indexer: FAISSIndexer object
        eval_file: Path to evaluation JSONL file
        top_k_values: List of k values for metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load evaluation data
    eval_data = []
    with open(eval_file, 'r') as f:
        for line in f:
            eval_data.append(json.loads(line))
    
    logger.info(f"Loaded {len(eval_data)} evaluation queries")
    
    # Evaluate
    metrics = evaluate_retrieval(indexer, eval_data, top_k_values)
    
    # Log results
    logger.info("Evaluation Results:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    return metrics
