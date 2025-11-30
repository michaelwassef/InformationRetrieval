"""
Evaluation metrics for Information Retrieval system.
Implements Precision, Recall, and Mean Average Precision (MAP).
"""

from typing import List, Set, Dict, Tuple
import numpy as np


def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Calculate Precision@k.
    
    Args:
        retrieved: List of retrieved document IDs (ranked)
        relevant: Set of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Precision@k score
    """
    if k == 0:
        return 0.0
    
    retrieved_k = retrieved[:k]
    if len(retrieved_k) == 0:
        return 0.0
    
    relevant_retrieved = len([doc_id for doc_id in retrieved_k if doc_id in relevant])
    return relevant_retrieved / len(retrieved_k)


def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Calculate Recall@k.
    
    Args:
        retrieved: List of retrieved document IDs (ranked)
        relevant: Set of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Recall@k score
    """
    if len(relevant) == 0:
        return 0.0
    
    retrieved_k = retrieved[:k]
    relevant_retrieved = len([doc_id for doc_id in retrieved_k if doc_id in relevant])
    return relevant_retrieved / len(relevant)


def average_precision(retrieved: List[int], relevant: Set[int]) -> float:
    """
    Calculate Average Precision (AP).
    
    Args:
        retrieved: List of retrieved document IDs (ranked)
        relevant: Set of relevant document IDs
        
    Returns:
        Average Precision score
    """
    if len(relevant) == 0:
        return 0.0
    
    relevant_retrieved = 0
    precision_sum = 0.0
    
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            relevant_retrieved += 1
            precision = relevant_retrieved / (i + 1)
            precision_sum += precision
    
    if relevant_retrieved == 0:
        return 0.0
    
    return precision_sum / len(relevant)


def mean_average_precision(retrieved_list: List[List[int]], 
                          relevant_list: List[Set[int]]) -> float:
    """
    Calculate Mean Average Precision (MAP).
    
    Args:
        retrieved_list: List of retrieved document ID lists (one per query)
        relevant_list: List of relevant document ID sets (one per query)
        
    Returns:
        Mean Average Precision score
    """
    if len(retrieved_list) != len(relevant_list):
        raise ValueError("Number of queries must match number of relevance sets")
    
    if len(retrieved_list) == 0:
        return 0.0
    
    ap_scores = []
    for retrieved, relevant in zip(retrieved_list, relevant_list):
        ap = average_precision(retrieved, relevant)
        ap_scores.append(ap)
    
    return np.mean(ap_scores)


def f1_score_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Calculate F1 score@k.
    
    Args:
        retrieved: List of retrieved document IDs (ranked)
        relevant: Set of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        F1@k score
    """
    prec = precision_at_k(retrieved, relevant, k)
    rec = recall_at_k(retrieved, relevant, k)
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)


class Evaluator:
    """Evaluation framework for retrieval systems."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
    
    def evaluate_query(self, retrieved: List[int], relevant: Set[int], 
                      k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate a single query.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k_values: List of k values for Precision@k and Recall@k
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Average Precision
        metrics['AP'] = average_precision(retrieved, relevant)
        
        # Precision@k and Recall@k for different k values
        for k in k_values:
            metrics[f'P@{k}'] = precision_at_k(retrieved, relevant, k)
            metrics[f'R@{k}'] = recall_at_k(retrieved, relevant, k)
            metrics[f'F1@{k}'] = f1_score_at_k(retrieved, relevant, k)
        
        return metrics
    
    def evaluate_multiple_queries(self, retrieved_list: List[List[int]],
                                  relevant_list: List[Set[int]],
                                  k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate multiple queries.
        
        Args:
            retrieved_list: List of retrieved document ID lists (one per query)
            relevant_list: List of relevant document ID sets (one per query)
            k_values: List of k values for Precision@k and Recall@k
            
        Returns:
            Dictionary of average evaluation metrics
        """
        if len(retrieved_list) != len(relevant_list):
            raise ValueError("Number of queries must match number of relevance sets")
        
        all_metrics = []
        for retrieved, relevant in zip(retrieved_list, relevant_list):
            metrics = self.evaluate_query(retrieved, relevant, k_values)
            all_metrics.append(metrics)
        
        # Calculate averages
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # Add MAP
        avg_metrics['MAP'] = mean_average_precision(retrieved_list, relevant_list)
        
        return avg_metrics
    
    def print_evaluation_results(self, metrics: Dict[str, float]) -> None:
        """
        Print evaluation results in a formatted way.
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        if 'MAP' in metrics:
            print(f"Mean Average Precision (MAP): {metrics['MAP']:.4f}")
        
        if 'AP' in metrics:
            print(f"Average Precision (AP): {metrics['AP']:.4f}")
        
        # Print Precision@k, Recall@k, F1@k
        k_values = sorted([int(k.split('@')[1]) for k in metrics.keys() 
                          if '@' in k and k.startswith('P@')])
        
        if k_values:
            print("\nPrecision@k, Recall@k, F1@k:")
            for k in k_values:
                p = metrics.get(f'P@{k}', 0.0)
                r = metrics.get(f'R@{k}', 0.0)
                f1 = metrics.get(f'F1@{k}', 0.0)
                print(f"  k={k}: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
        
        print("="*80)


def create_relevance_set(query: str, documents: List[Dict], 
                        manual_labels: Dict[int, bool] = None) -> Set[int]:
    """
    Create a relevance set for evaluation.
    This is a helper function - in practice, relevance should be manually labeled.
    
    Args:
        query: Query string
        documents: List of document dictionaries with 'id' and 'text' fields
        manual_labels: Optional dictionary mapping doc_id to relevance (True/False)
        
    Returns:
        Set of relevant document IDs
    """
    if manual_labels:
        return {doc_id for doc_id, is_relevant in manual_labels.items() if is_relevant}
    
    # Simple heuristic: documents containing query terms are considered relevant
    # This is just for demonstration - real evaluation needs manual labeling
    query_terms = set(query.lower().split())
    relevant = set()
    
    for doc in documents:
        doc_text = doc.get('text', '').lower()
        doc_terms = set(doc_text.split())
        # If document contains at least one query term, consider it relevant
        if query_terms.intersection(doc_terms):
            relevant.add(doc['id'])
    
    return relevant

