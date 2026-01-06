#!/usr/bin/env python3
"""
Phase 10: Graph-RAG vs Plain RAG Evaluation

Compares retrieval performance between:
1) Plain RAG retrieval
2) Graph-constrained RAG retrieval (Graph-RAG)

Runs on real ChromaDB data only.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def log(msg: str) -> None:
    """Simple logging with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def load_queries(queries_file: str) -> List[Dict]:
    """Load evaluation queries from JSONL file."""
    queries = []
    with open(queries_file, 'r') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line.strip()))
    return queries

def compute_metrics(expected_sections: List[str], retrieved_sections: List[str]) -> Tuple[float, float]:
    """Compute recall and precision@K."""
    if not expected_sections:
        return 0.0, 0.0
    
    expected_set = set(expected_sections)
    retrieved_set = set(retrieved_sections)
    
    intersection = expected_set & retrieved_set
    
    recall = len(intersection) / len(expected_set)
    precision = len(intersection) / len(retrieved_sections) if retrieved_sections else 0.0
    
    return recall, precision

def evaluate_query(query: str, expected_sections: List[str], retriever, graph_filter) -> Dict[str, Any]:
    """Evaluate a single query with both RAG and Graph-RAG."""
    
    # 1. Run REAL retrieval
    retrieved_chunks = retriever.retrieve(query, top_k=10)
    
    # 2. Extract metadata
    rag_sections = []
    rag_semantic_ids = []
    rag_chunk_ids = []
    
    for chunk in retrieved_chunks:
        act = chunk.get('act', '')
        section = chunk.get('section', '')
        semantic_id = chunk.get('semantic_id', '')
        chunk_id = chunk.get('chunk_id', '')
        
        if act and section:
            section_key = f"{act}::{section}"
            rag_sections.append(section_key)
        
        rag_semantic_ids.append(semantic_id)
        rag_chunk_ids.append(chunk_id)
    
    # 3. Compute RAG metrics
    rag_recall, rag_precision = compute_metrics(expected_sections, rag_sections)
    
    # 4. Run Graph-RAG filtering
    try:
        filtered_chunks = graph_filter.filter_chunks(
            query=query,
            retrieved_chunks=retrieved_chunks
        )
        
        # 5. Extract filtered metadata
        graph_sections = []
        graph_semantic_ids = []
        graph_chunk_ids = []
        exclusion_reasons = []
        
        for chunk in filtered_chunks.get('allowed_chunks', []):
            act = chunk.get('act', '')
            section = chunk.get('section', '')
            semantic_id = chunk.get('semantic_id', '')
            chunk_id = chunk.get('chunk_id', '')
            
            if act and section:
                section_key = f"{act}::{section}"
                graph_sections.append(section_key)
            
            graph_semantic_ids.append(semantic_id)
            graph_chunk_ids.append(chunk_id)
        
        for chunk in filtered_chunks.get('excluded_chunks', []):
            exclusion_reasons.append(chunk.get('reason', 'unknown'))
        
        # 6. Compute Graph-RAG metrics
        graph_recall, graph_precision = compute_metrics(expected_sections, graph_sections)
        
        # 7. Compute noise reduction
        noise_reduction = (len(rag_sections) - len(graph_sections)) / len(rag_sections) if rag_sections else 0.0
        
        return {
            'query': query,
            'expected_sections': expected_sections,
            'rag': {
                'count': len(rag_sections),
                'sections': rag_sections,
                'semantic_ids': rag_semantic_ids,
                'chunk_ids': rag_chunk_ids,
                'recall': rag_recall,
                'precision': rag_precision
            },
            'graph_rag': {
                'count': len(graph_sections),
                'sections': graph_sections,
                'semantic_ids': graph_semantic_ids,
                'chunk_ids': graph_chunk_ids,
                'recall': graph_recall,
                'precision': graph_precision,
                'exclusion_reasons': exclusion_reasons,
                'noise_reduction': noise_reduction
            },
            'success': True
        }
        
    except Exception as e:
        log(f"ERROR in graph filtering: {e}")
        return {
            'query': query,
            'expected_sections': expected_sections,
            'rag': {
                'count': len(rag_sections),
                'sections': rag_sections,
                'semantic_ids': rag_semantic_ids,
                'chunk_ids': rag_chunk_ids,
                'recall': rag_recall,
                'precision': rag_precision
            },
            'graph_rag': None,
            'error': str(e),
            'success': False
        }

def main():
    parser = argparse.ArgumentParser(description="Phase 10: Graph-RAG vs Plain RAG Evaluation")
    parser.add_argument(
        "--queries",
        default="eval/phase10_queries.jsonl",
        help="Path to evaluation queries file"
    )
    parser.add_argument(
        "--output",
        default="results/phase10_results.json",
        help="Path to output results file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("PHASE 10: GRAPH-RAG VS PLAIN RAG EVALUATION")
    print("=" * 80)
    log(f"Queries file: {args.queries}")
    log(f"Output file: {args.output}")
    
    # Check dependencies
    try:
        from src.rag.retrieval.retriever import LegalRetriever
        from src.graph.graph_rag_filter import GraphRAGFilter
        import pickle
    except ImportError as e:
        log(f"ERROR: Missing dependencies: {e}")
        sys.exit(1)
    
    # Load queries
    try:
        queries = load_queries(args.queries)
        log(f"Loaded {len(queries)} evaluation queries")
    except Exception as e:
        log(f"ERROR: Failed to load queries: {e}")
        sys.exit(1)
    
    # Initialize retriever
    try:
        retriever = LegalRetriever()
        stats = retriever.initialize()
        log(f"Retriever initialized: {stats}")
        
        if stats.get('bm25_chunks', 0) == 0:
            log("ERROR: No chunks in retriever")
            sys.exit(1)
            
    except Exception as e:
        log(f"ERROR: Failed to initialize retriever: {e}")
        sys.exit(1)
    
    # Initialize graph filter
    try:
        graph_path = PROJECT_ROOT / "data" / "graph" / "legal_graph_v2.pkl"
        if not graph_path.exists():
            log(f"ERROR: Graph file not found: {graph_path}")
            sys.exit(1)
        
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        
        graph_filter = GraphRAGFilter(graph)
        log("Graph filter initialized")
        
    except Exception as e:
        log(f"ERROR: Failed to initialize graph filter: {e}")
        sys.exit(1)
    
    # Run evaluation
    results = []
    successful_evals = 0
    
    print()
    log("Running evaluation...")
    
    for i, query_data in enumerate(queries, 1):
        query = query_data['query']
        expected_sections = query_data['expected_sections']
        
        if args.verbose:
            log(f"Query {i}/{len(queries)}: {query}")
        
        result = evaluate_query(query, expected_sections, retriever, graph_filter)
        results.append(result)
        
        if result['success']:
            successful_evals += 1
            if args.verbose:
                rag_prec = result['rag']['precision']
                graph_prec = result['graph_rag']['precision'] if result['graph_rag'] else 0.0
                log(f"  RAG Precision: {rag_prec:.3f}, Graph-RAG Precision: {graph_prec:.3f}")
        else:
            log(f"  FAILED: {result.get('error', 'Unknown error')}")
    
    # Compute aggregate metrics
    rag_recalls = [r['rag']['recall'] for r in results if r['success']]
    rag_precisions = [r['rag']['precision'] for r in results if r['success']]
    graph_recalls = [r['graph_rag']['recall'] for r in results if r['success'] and r['graph_rag']]
    graph_precisions = [r['graph_rag']['precision'] for r in results if r['success'] and r['graph_rag']]
    noise_reductions = [r['graph_rag']['noise_reduction'] for r in results if r['success'] and r['graph_rag']]
    
    # Collect all exclusion reasons
    all_exclusion_reasons = []
    for r in results:
        if r['success'] and r['graph_rag']:
            all_exclusion_reasons.extend(r['graph_rag']['exclusion_reasons'])
    
    # Print results
    print()
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"Total queries: {len(queries)}")
    print(f"Successful evaluations: {successful_evals}")
    print()
    
    # Per-query table
    print("PER-QUERY COMPARISON:")
    print("-" * 80)
    print(f"{'Query':<40} {'RAG_Prec':<10} {'Graph_Prec':<12} {'Noise_Red':<10}")
    print("-" * 80)
    
    for r in results:
        if r['success'] and r['graph_rag']:
            query_short = r['query'][:37] + "..." if len(r['query']) > 40 else r['query']
            rag_prec = r['rag']['precision']
            graph_prec = r['graph_rag']['precision']
            noise_red = r['graph_rag']['noise_reduction']
            print(f"{query_short:<40} {rag_prec:<10.3f} {graph_prec:<12.3f} {noise_red:<10.3f}")
    
    print()
    
    # Aggregate metrics
    print("AGGREGATE METRICS:")
    print("-" * 40)
    print(f"Mean Recall@10 (RAG): {sum(rag_recalls)/len(rag_recalls):.3f}")
    print(f"Mean Precision@10 (RAG): {sum(rag_precisions)/len(rag_precisions):.3f}")
    if graph_recalls:
        print(f"Mean Recall@10 (Graph-RAG): {sum(graph_recalls)/len(graph_recalls):.3f}")
        print(f"Mean Precision@10 (Graph-RAG): {sum(graph_precisions)/len(graph_precisions):.3f}")
        print(f"Avg noise reduction: {sum(noise_reductions)/len(noise_reductions):.3f}")
    print()
    
    # Exclusion reasons breakdown
    if all_exclusion_reasons:
        from collections import Counter
        reason_counts = Counter(all_exclusion_reasons)
        print("EXCLUSION REASONS BREAKDOWN:")
        print("-" * 30)
        for reason, count in reason_counts.most_common():
            print(f"{reason}: {count}")
        print()
    
    # Save results
    results_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(queries),
            'successful_evaluations': successful_evals,
            'rag_mean_recall': sum(rag_recalls)/len(rag_recalls) if rag_recalls else 0,
            'rag_mean_precision': sum(rag_precisions)/len(rag_precisions) if rag_precisions else 0,
            'graph_rag_mean_recall': sum(graph_recalls)/len(graph_recalls) if graph_recalls else 0,
            'graph_rag_mean_precision': sum(graph_precisions)/len(graph_precisions) if graph_precisions else 0,
            'avg_noise_reduction': sum(noise_reductions)/len(noise_reductions) if noise_reductions else 0,
            'exclusion_reasons': dict(Counter(all_exclusion_reasons)) if all_exclusion_reasons else {}
        },
        'results': results
    }
    
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    log(f"Results saved to: {output_path}")
    
    # Final assertion
    print("=" * 80)
    if (graph_precisions and rag_precisions and 
        sum(graph_precisions)/len(graph_precisions) > sum(rag_precisions)/len(rag_precisions) and
        abs((sum(graph_recalls)/len(graph_recalls)) - (sum(rag_recalls)/len(rag_recalls))) < 0.1):
        print("PHASE 10 COMPLETE — GRAPH-RAG IMPROVES PRECISION WITHOUT RECALL LOSS")
    else:
        print("PHASE 10 FAILED — GRAPH-RAG DOES NOT MEET EXPECTATIONS")
        if not graph_precisions:
            print("  - No Graph-RAG results")
        elif not rag_precisions:
            print("  - No RAG results")
        else:
            graph_avg = sum(graph_precisions)/len(graph_precisions)
            rag_avg = sum(rag_precisions)/len(rag_precisions)
            print(f"  - Graph-RAG precision: {graph_avg:.3f}")
            print(f"  - RAG precision: {rag_avg:.3f}")
            if graph_recalls and rag_recalls:
                graph_recall_avg = sum(graph_recalls)/len(graph_recalls)
                rag_recall_avg = sum(rag_recalls)/len(rag_recalls)
                print(f"  - Recall difference: {abs(graph_recall_avg - rag_recall_avg):.3f}")

if __name__ == "__main__":
    main()
