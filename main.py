"""
Main entry point for the Information Retrieval system.
"""

import argparse
from data_loader import DataLoader
from preprocessing import TextPreprocessor
from vectorization import DocumentVectorizer
from retrieval_models import RetrievalSystem
from evaluation import Evaluator, create_relevance_set
from visualization import (plot_word_frequency, create_wordcloud,
                          plot_similarity_scores, plot_model_comparison,
                          perform_lda_topic_modeling, visualize_lda_topics,
                          print_lda_topics, plot_evaluation_metrics)
from utils import print_separator
import sys


def initialize_system(max_documents: int = 1000, use_tfidf: bool = True):
    """
    Initialize the IR system with data loading and preprocessing.
    
    Args:
        max_documents: Maximum number of documents to load
        use_tfidf: Whether to use TF-IDF (True) or BoW (False)
        
    Returns:
        Tuple of (data_loader, retrieval_system, document_titles_dict)
    """
    print_separator()
    print("Initializing Information Retrieval System...")
    print_separator()
    
    # Load data
    print("\n[1/5] Loading dataset...")
    data_loader = DataLoader(max_documents=max_documents)
    try:
        data_loader.load_dataset()
        data_loader.print_dataset_info()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have internet connection and datasets library installed.")
        sys.exit(1)
    
    documents = data_loader.get_documents()
    document_texts = [doc['text'] for doc in documents]
    document_ids = [doc['id'] for doc in documents]
    document_titles = {doc['id']: doc['title'] for doc in documents}
    
    # Initialize preprocessor
    print("\n[2/5] Initializing text preprocessor...")
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_stopwords=True,
        remove_punctuation=True,
        stem=False,
        lemmatize=True
    )
    
    # Initialize vectorizer
    print("\n[3/5] Initializing vectorizer...")
    vectorizer = DocumentVectorizer(
        preprocessor=preprocessor,
        method='tfidf' if use_tfidf else 'bow',
        max_features=5000,
        min_df=2,
        max_df=0.95
    )
    
    # Vectorize documents
    print("Vectorizing documents...")
    vectorizer.fit_transform(document_texts)
    print(f"Vocabulary size: {vectorizer.get_vocabulary_size()}")
    
    # Initialize retrieval system
    print("\n[4/5] Initializing retrieval models...")
    retrieval_system = RetrievalSystem(
        documents=document_texts,
        document_ids=document_ids,
        preprocessor=preprocessor,
        vectorizer=vectorizer
    )
    
    print("\n[5/5] System initialization complete!")
    print_separator()
    
    return data_loader, retrieval_system, document_titles


def interactive_search(retrieval_system, document_titles, data_loader):
    """
    Interactive search interface.
    
    Args:
        retrieval_system: Initialized RetrievalSystem instance
        document_titles: Dictionary mapping document_id to title
        data_loader: DataLoader instance
    """
    print("\n" + "="*80)
    print("INTERACTIVE SEARCH")
    print("="*80)
    print("Enter queries to search. Type 'quit' to exit.")
    print("For Boolean queries, use AND, OR, NOT operators (e.g., 'python AND machine')")
    print("="*80)
    
    while True:
        query = input("\nEnter query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        print_separator("-", 80)
        print(f"Query: {query}")
        print_separator("-", 80)
        
        # Search with different models
        try:
            # VSM
            print("\n[Vector Space Model]")
            vsm_results = retrieval_system.search_vsm(query, top_k=5)
            if vsm_results:
                for rank, (doc_id, score) in enumerate(vsm_results, 1):
                    title = document_titles.get(doc_id, f"Document {doc_id}")
                    doc = data_loader.get_document_by_id(doc_id)
                    preview = doc['text'][:150] + "..." if doc and len(doc['text']) > 150 else (doc['text'] if doc else "")
                    print(f"{rank}. [{score:.4f}] {title}")
                    print(f"   Preview: {preview}")
            else:
                print("No results found.")
            
            # BM25
            print("\n[BM25]")
            bm25_results = retrieval_system.search_bm25(query, top_k=5)
            if bm25_results:
                for rank, (doc_id, score) in enumerate(bm25_results, 1):
                    title = document_titles.get(doc_id, f"Document {doc_id}")
                    doc = data_loader.get_document_by_id(doc_id)
                    preview = doc['text'][:150] + "..." if doc and len(doc['text']) > 150 else (doc['text'] if doc else "")
                    print(f"{rank}. [{score:.4f}] {title}")
                    print(f"   Preview: {preview}")
            else:
                print("No results found.")
            
            # Boolean (only if query contains operators)
            if any(op in query.upper() for op in ['AND', 'OR', 'NOT']):
                print("\n[Boolean Retrieval]")
                boolean_results = retrieval_system.search_boolean(query, top_k=5)
                if boolean_results:
                    for rank, doc_id in enumerate(boolean_results, 1):
                        title = document_titles.get(doc_id, f"Document {doc_id}")
                        doc = data_loader.get_document_by_id(doc_id)
                        preview = doc['text'][:150] + "..." if doc and len(doc['text']) > 150 else (doc['text'] if doc else "")
                        print(f"{rank}. {title}")
                        print(f"   Preview: {preview}")
                else:
                    print("No results found.")
        
        except Exception as e:
            print(f"Error during search: {e}")


def run_evaluation(retrieval_system, data_loader, document_titles):
    """
    Run evaluation on sample queries.
    
    Args:
        retrieval_system: Initialized RetrievalSystem instance
        data_loader: DataLoader instance
        document_titles: Dictionary mapping document_id to title
    """
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    # Sample queries for evaluation
    sample_queries = [
        "artificial intelligence",
        "machine learning",
        "computer science",
        "history",
        "mathematics"
    ]
    
    evaluator = Evaluator()
    all_metrics = {}
    
    for query in sample_queries:
        print(f"\nEvaluating query: '{query}'")
        
        # Get results from different models
        vsm_results = [doc_id for doc_id, _ in retrieval_system.search_vsm(query, top_k=20)]
        bm25_results = [doc_id for doc_id, _ in retrieval_system.search_bm25(query, top_k=20)]
        
        # Create relevance set (using heuristic - in practice, use manual labels)
        documents = data_loader.get_documents()
        relevant = create_relevance_set(query, documents)
        
        if len(relevant) == 0:
            print(f"  No relevant documents found for query '{query}'. Skipping...")
            continue
        
        # Evaluate VSM
        vsm_metrics = evaluator.evaluate_query(vsm_results, relevant)
        print(f"  VSM - MAP: {vsm_metrics.get('AP', 0):.4f}, P@10: {vsm_metrics.get('P@10', 0):.4f}")
        
        # Evaluate BM25
        bm25_metrics = evaluator.evaluate_query(bm25_results, relevant)
        print(f"  BM25 - MAP: {bm25_metrics.get('AP', 0):.4f}, P@10: {bm25_metrics.get('P@10', 0):.4f}")
        
        # Store metrics
        if 'VSM' not in all_metrics:
            all_metrics['VSM'] = []
            all_metrics['BM25'] = []
        
        all_metrics['VSM'].append(vsm_metrics)
        all_metrics['BM25'].append(bm25_metrics)
    
    # Calculate average metrics
    if all_metrics:
        avg_metrics = {}
        for model_name, metrics_list in all_metrics.items():
            avg_metrics[model_name] = {
                key: sum(m.get(key, 0) for m in metrics_list) / len(metrics_list)
                for key in metrics_list[0].keys()
            }
        
        evaluator.print_evaluation_results({'VSM': avg_metrics['VSM'], 
                                           'BM25': avg_metrics['BM25']})


def generate_visualizations(retrieval_system, data_loader, document_titles):
    """
    Generate visualizations for the dataset and retrieval results.
    
    Args:
        retrieval_system: Initialized RetrievalSystem instance
        data_loader: DataLoader instance
        document_titles: Dictionary mapping document_id to title
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Get preprocessed documents
    documents = data_loader.get_documents()
    document_texts = [doc['text'] for doc in documents]
    
    # Combine all documents for word cloud
    print("\n[1/4] Creating word cloud...")
    combined_text = ' '.join(document_texts[:50])  # Use first 50 documents
    preprocessed_text = retrieval_system.preprocessor.preprocess(combined_text)
    create_wordcloud(preprocessed_text, "Top Keywords in Documents")
    
    # Word frequency distribution
    print("\n[2/4] Plotting word frequency distribution...")
    plot_word_frequency(preprocessed_text, top_n=20, 
                       title="Top 20 Words in Document Collection")
    
    # Example query visualization
    print("\n[3/4] Visualizing query results...")
    example_query = "machine learning"
    vsm_results = retrieval_system.search_vsm(example_query, top_k=10)
    bm25_results = retrieval_system.search_bm25(example_query, top_k=10)
    
    plot_similarity_scores(vsm_results, document_titles, 
                          f"VSM Results for: '{example_query}'", top_k=10)
    
    plot_model_comparison({
        'VSM': vsm_results,
        'BM25': bm25_results
    }, document_titles, top_k=5)
    
    # LDA Topic Modeling
    print("\n[4/4] Performing LDA topic modeling...")
    preprocessed_docs = retrieval_system.preprocessed_documents[:100]  # Use first 100 docs
    lda_model = perform_lda_topic_modeling(preprocessed_docs, num_topics=5)
    print_lda_topics(lda_model, num_words=10)
    visualize_lda_topics(lda_model, num_words=10)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Information Retrieval System')
    parser.add_argument('--max-docs', type=int, default=1000,
                       help='Maximum number of documents to load (default: 1000)')
    parser.add_argument('--mode', choices=['interactive', 'eval', 'visualize', 'all'],
                       default='all', help='Run mode (default: all)')
    parser.add_argument('--use-bow', action='store_true',
                       help='Use Bag-of-Words instead of TF-IDF')
    
    args = parser.parse_args()
    
    # Initialize system
    data_loader, retrieval_system, document_titles = initialize_system(
        max_documents=args.max_docs,
        use_tfidf=not args.use_bow
    )
    
    # Run based on mode
    if args.mode in ['interactive', 'all']:
        interactive_search(retrieval_system, document_titles, data_loader)
    
    if args.mode in ['eval', 'all']:
        run_evaluation(retrieval_system, data_loader, document_titles)
    
    if args.mode in ['visualize', 'all']:
        try:
            generate_visualizations(retrieval_system, data_loader, document_titles)
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            print("Some visualizations may require display capabilities.")


if __name__ == "__main__":
    main()

