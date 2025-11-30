"""
Visualization functions for Information Retrieval system.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import Counter
from gensim import corpora, models
from gensim.models import LdaModel
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_word_frequency(text: str, top_n: int = 20, title: str = "Word Frequency Distribution") -> None:
    """
    Plot frequency distribution of words.
    
    Args:
        text: Preprocessed text string
        top_n: Number of top words to display
        title: Plot title
    """
    # Tokenize and count
    words = text.split()
    word_freq = Counter(words)
    
    # Get top N words
    top_words = word_freq.most_common(top_n)
    words_list, freqs_list = zip(*top_words) if top_words else ([], [])
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(words_list)), freqs_list)
    plt.yticks(range(len(words_list)), words_list)
    plt.xlabel('Frequency')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def create_wordcloud(text: str, title: str = "Word Cloud", 
                    width: int = 800, height: int = 400) -> None:
    """
    Create and display a word cloud.
    
    Args:
        text: Preprocessed text string
        title: Plot title
        width: Word cloud width
        height: Word cloud height
    """
    # Create word cloud
    wordcloud = WordCloud(width=width, height=height, 
                         background_color='white',
                         max_words=100,
                         colormap='viridis').generate(text)
    
    # Display
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()


def plot_similarity_scores(results: List[Tuple[int, float]], 
                           document_titles: Dict[int, str] = None,
                           title: str = "Document-Query Similarity Scores",
                           top_k: int = 10) -> None:
    """
    Plot document-query similarity scores as a bar chart.
    
    Args:
        results: List of (document_id, score) tuples
        document_titles: Dictionary mapping document_id to title
        title: Plot title
        top_k: Number of top results to display
    """
    # Get top k results
    top_results = results[:top_k]
    
    if not top_results:
        print("No results to display.")
        return
    
    doc_ids, scores = zip(*top_results)
    
    # Get titles or use IDs
    if document_titles:
        labels = [document_titles.get(doc_id, f"Doc {doc_id}")[:50] 
                 for doc_id in doc_ids]
    else:
        labels = [f"Doc {doc_id}" for doc_id in doc_ids]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(len(labels)), scores, color='steelblue')
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Similarity Score')
    plt.title(title)
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(score, i, f' {score:.3f}', va='center')
    
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_dict: Dict[str, List[Tuple[int, float]]],
                         document_titles: Dict[int, str] = None,
                         top_k: int = 5) -> None:
    """
    Compare similarity scores across different retrieval models.
    
    Args:
        results_dict: Dictionary mapping model names to result lists
        document_titles: Dictionary mapping document_id to title
        top_k: Number of top results to compare
    """
    fig, axes = plt.subplots(1, len(results_dict), figsize=(15, 5))
    if len(results_dict) == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        top_results = results[:top_k]
        if not top_results:
            continue
        
        doc_ids, scores = zip(*top_results)
        
        if document_titles:
            labels = [document_titles.get(doc_id, f"Doc {doc_id}")[:30] 
                     for doc_id in doc_ids]
        else:
            labels = [f"Doc {doc_id}" for doc_id in doc_ids]
        
        axes[idx].barh(range(len(labels)), scores, color='steelblue')
        axes[idx].set_yticks(range(len(labels)))
        axes[idx].set_yticklabels(labels)
        axes[idx].set_xlabel('Score')
        axes[idx].set_title(f'{model_name}\n(Top {top_k})')
        axes[idx].invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def perform_lda_topic_modeling(documents: List[str], num_topics: int = 5,
                              num_words: int = 10) -> LdaModel:
    """
    Perform LDA topic modeling on documents.
    
    Args:
        documents: List of preprocessed document strings
        num_topics: Number of topics to extract
        num_words: Number of top words per topic to return
        
    Returns:
        Trained LDA model
    """
    # Tokenize documents
    tokenized_docs = [doc.split() for doc in documents]
    
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    
    # Train LDA model
    lda_model = LdaModel(corpus=corpus,
                        id2word=dictionary,
                        num_topics=num_topics,
                        random_state=42,
                        passes=10,
                        alpha='auto',
                        per_word_topics=True)
    
    return lda_model


def visualize_lda_topics(lda_model: LdaModel, num_words: int = 10) -> None:
    """
    Visualize LDA topics.
    
    Args:
        lda_model: Trained LDA model
        num_words: Number of words per topic to display
    """
    num_topics = lda_model.num_topics
    
    fig, axes = plt.subplots(1, num_topics, figsize=(15, 5))
    if num_topics == 1:
        axes = [axes]
    
    for topic_idx in range(num_topics):
        topic_words = lda_model.show_topic(topic_idx, topn=num_words)
        words, weights = zip(*topic_words) if topic_words else ([], [])
        
        axes[topic_idx].barh(range(len(words)), weights, color='steelblue')
        axes[topic_idx].set_yticks(range(len(words)))
        axes[topic_idx].set_yticklabels(words)
        axes[topic_idx].set_xlabel('Weight')
        axes[topic_idx].set_title(f'Topic {topic_idx + 1}')
        axes[topic_idx].invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def print_lda_topics(lda_model: LdaModel, num_words: int = 10) -> None:
    """
    Print LDA topics in text format.
    
    Args:
        lda_model: Trained LDA model
        num_words: Number of words per topic to display
    """
    print("\n" + "="*80)
    print("LDA TOPIC MODELING RESULTS")
    print("="*80)
    
    for topic_idx in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(topic_idx, topn=num_words)
        print(f"\nTopic {topic_idx + 1}:")
        for word, weight in topic_words:
            print(f"  {word}: {weight:.4f}")
    
    print("="*80)


def plot_topic_distribution(lda_model: LdaModel, corpus: List,
                           document_titles: List[str] = None,
                           num_docs: int = 10) -> None:
    """
    Plot topic distribution for documents.
    
    Args:
        lda_model: Trained LDA model
        corpus: Document corpus (list of bow vectors)
        document_titles: List of document titles
        num_docs: Number of documents to display
    """
    num_docs = min(num_docs, len(corpus))
    num_topics = lda_model.num_topics
    
    # Get topic distributions for documents
    topic_distributions = []
    for doc_idx in range(num_docs):
        doc_topics = lda_model.get_document_topics(corpus[doc_idx])
        topic_dist = [0.0] * num_topics
        for topic_id, prob in doc_topics:
            topic_dist[topic_id] = prob
        topic_distributions.append(topic_dist)
    
    # Create DataFrame
    df = pd.DataFrame(topic_distributions, 
                     columns=[f'Topic {i+1}' for i in range(num_topics)])
    
    if document_titles:
        df.index = [title[:30] for title in document_titles[:num_docs]]
    else:
        df.index = [f'Doc {i}' for i in range(num_docs)]
    
    # Plot
    plt.figure(figsize=(12, 6))
    df.plot(kind='bar', stacked=True, colormap='tab20')
    plt.xlabel('Documents')
    plt.ylabel('Topic Probability')
    plt.title('Topic Distribution Across Documents')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_evaluation_metrics(metrics_dict: Dict[str, Dict[str, float]],
                            metric_name: str = 'MAP') -> None:
    """
    Plot evaluation metrics comparison across models.
    
    Args:
        metrics_dict: Dictionary mapping model names to metrics dictionaries
        metric_name: Name of metric to plot (e.g., 'MAP', 'P@10')
    """
    model_names = list(metrics_dict.keys())
    metric_values = [metrics_dict[model].get(metric_name, 0.0) 
                    for model in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, metric_values, color='steelblue')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Comparison Across Models')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(retrieved: List[int], relevant: Set[int]) -> None:
    """
    Plot precision-recall curve for a single query.
    
    Args:
        retrieved: List of retrieved document IDs (ranked)
        relevant: Set of relevant document IDs
    """
    from evaluation import precision_at_k, recall_at_k
    
    precisions = []
    recalls = []
    
    for k in range(1, len(retrieved) + 1):
        prec = precision_at_k(retrieved, relevant, k)
        rec = recall_at_k(retrieved, relevant, k)
        precisions.append(prec)
        recalls.append(rec)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='o', linewidth=2, markersize=4)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()

