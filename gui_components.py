"""
Reusable GUI components for the Information Retrieval system.
"""

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from typing import List, Dict, Tuple, Optional, Callable
from gui_utils import format_document_preview, format_score, create_tooltip


class SearchPanel(ttk.Frame):
    """Search interface panel."""
    
    def __init__(self, parent, on_search: Callable):
        """
        Initialize search panel.
        
        Args:
            parent: Parent widget
            on_search: Callback function(query, model, top_k)
        """
        super().__init__(parent)
        self.on_search = on_search
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI components."""
        # Query input
        query_frame = ttk.LabelFrame(self, text="Enter Query", padding=10)
        query_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.query_entry = ttk.Entry(query_frame, font=("Arial", 11))
        self.query_entry.pack(fill=tk.X, pady=5)
        self.query_entry.bind('<Return>', lambda e: self.perform_search())
        
        # Model selection
        model_frame = ttk.LabelFrame(self, text="Retrieval Model", padding=10)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.model_var = tk.StringVar(value="VSM")
        ttk.Radiobutton(model_frame, text="Vector Space Model (VSM)", 
                       variable=self.model_var, value="VSM").pack(anchor=tk.W)
        ttk.Radiobutton(model_frame, text="BM25", 
                       variable=self.model_var, value="BM25").pack(anchor=tk.W)
        ttk.Radiobutton(model_frame, text="Boolean", 
                       variable=self.model_var, value="Boolean").pack(anchor=tk.W)
        
        # Search options
        options_frame = ttk.LabelFrame(self, text="Search Options", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(options_frame, text="Number of results (leave empty for all):").pack(anchor=tk.W)
        self.top_k_var = tk.StringVar(value="")
        top_k_entry = ttk.Entry(options_frame, textvariable=self.top_k_var, width=10)
        top_k_entry.pack(anchor=tk.W, pady=2)
        ttk.Label(options_frame, text="(Empty = show all results)", font=("Arial", 8), foreground="gray").pack(anchor=tk.W)
        
        # Search button
        search_btn = ttk.Button(self, text="Search", command=self.perform_search)
        search_btn.pack(pady=10)
        create_tooltip(search_btn, "Press Enter or click to search")
    
    def perform_search(self):
        """Perform the search."""
        query = self.query_entry.get().strip()
        if not query:
            return
        
        model = self.model_var.get()
        top_k_str = self.top_k_var.get().strip()
        if top_k_str == "":
            top_k = None  # Show all results
        else:
            try:
                top_k = int(top_k_str)
                if top_k <= 0:
                    top_k = None  # Invalid number, show all
            except ValueError:
                top_k = None  # Invalid input, show all
        
        self.on_search(query, model, top_k)
    
    def get_query(self) -> str:
        """Get the current query."""
        return self.query_entry.get().strip()
    
    def set_query(self, query: str):
        """Set the query text."""
        self.query_entry.delete(0, tk.END)
        self.query_entry.insert(0, query)


class ResultsPanel(ttk.Frame):
    """Results display panel."""
    
    def __init__(self, parent, on_document_select: Optional[Callable] = None):
        """
        Initialize results panel.
        
        Args:
            parent: Parent widget
            on_document_select: Callback when document is selected
        """
        super().__init__(parent)
        self.on_document_select = on_document_select
        self.results = []
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI components."""
        # Results list
        list_frame = ttk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview for results
        columns = ("Rank", "Document", "Score")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        self.tree.heading("Rank", text="Rank")
        self.tree.heading("Document", text="Document Title")
        self.tree.heading("Score", text="Score")
        
        self.tree.column("Rank", width=50)
        self.tree.column("Document", width=300)
        self.tree.column("Score", width=100)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree.bind('<Double-1>', self.on_double_click)
        
        # Document preview
        preview_frame = ttk.LabelFrame(self, text="Document Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.preview_text = tk.Text(preview_frame, wrap=tk.WORD, height=8, 
                                   font=("Arial", 10))
        preview_scroll = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, 
                                      command=self.preview_text.yview)
        self.preview_text.configure(yscrollcommand=preview_scroll.set)
        
        self.preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preview_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def display_results(self, results: List[Tuple[int, float]], 
                       document_titles: Dict[int, str],
                       document_texts: Optional[Dict[int, str]] = None):
        """
        Display search results.
        
        Args:
            results: List of (doc_id, score) tuples
            document_titles: Dictionary mapping doc_id to title
            document_texts: Optional dictionary mapping doc_id to text
        """
        # Clear existing results
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.results = results
        self.document_titles = document_titles
        self.document_texts = document_texts or {}
        
        # Add results to tree
        for rank, (doc_id, score) in enumerate(results, 1):
            title = document_titles.get(doc_id, f"Document {doc_id}")
            self.tree.insert("", tk.END, values=(rank, title[:80], format_score(score)))
    
    def on_double_click(self, event):
        """Handle double-click on result."""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        rank = int(item['values'][0]) - 1
        
        if rank < len(self.results):
            doc_id, score = self.results[rank]
            if self.on_document_select:
                self.on_document_select(doc_id)
            
            # Show preview
            text = self.document_texts.get(doc_id, "")
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(1.0, format_document_preview(text, 500))
    
    def clear(self):
        """Clear all results."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.preview_text.delete(1.0, tk.END)
        self.results = []


class VisualizationPanel(ttk.Frame):
    """Visualization panel for charts and plots."""
    
    def __init__(self, parent):
        """
        Initialize visualization panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI components."""
        # Notebook for different visualizations
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Word cloud tab
        self.wordcloud_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.wordcloud_frame, text="Word Cloud")
        
        self.wordcloud_label = tk.Label(self.wordcloud_frame, text="No word cloud generated")
        self.wordcloud_label.pack(expand=True)
        
        # Frequency chart tab
        self.freq_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.freq_frame, text="Frequency")
        
        self.freq_fig = Figure(figsize=(6, 4), dpi=100)
        self.freq_canvas = FigureCanvasTkAgg(self.freq_fig, self.freq_frame)
        self.freq_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Similarity scores tab
        self.sim_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sim_frame, text="Similarity Scores")
        
        self.sim_fig = Figure(figsize=(6, 4), dpi=100)
        self.sim_canvas = FigureCanvasTkAgg(self.sim_fig, self.sim_frame)
        self.sim_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Model comparison tab
        self.comp_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.comp_frame, text="Model Comparison")
        
        self.comp_fig = Figure(figsize=(6, 4), dpi=100)
        self.comp_canvas = FigureCanvasTkAgg(self.comp_fig, self.comp_frame)
        self.comp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # LDA Topic Clustering tab
        self.lda_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.lda_frame, text="LDA Topics")
        
        # Create scrollable frame for LDA visualization
        lda_scroll_frame = ttk.Frame(self.lda_frame)
        lda_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar
        lda_scrollbar = ttk.Scrollbar(lda_scroll_frame)
        lda_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Canvas for scrolling
        self.lda_canvas = tk.Canvas(lda_scroll_frame, yscrollcommand=lda_scrollbar.set)
        self.lda_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lda_scrollbar.config(command=self.lda_canvas.yview)
        
        # Frame inside canvas for LDA content
        self.lda_content_frame = ttk.Frame(self.lda_canvas)
        self.lda_canvas_window = self.lda_canvas.create_window((0, 0), window=self.lda_content_frame, anchor="nw")
        
        def configure_lda_scroll_region(event):
            self.lda_canvas.configure(scrollregion=self.lda_canvas.bbox("all"))
        
        def configure_lda_canvas_width(event):
            canvas_width = event.width
            self.lda_canvas.itemconfig(self.lda_canvas_window, width=canvas_width)
        
        self.lda_content_frame.bind("<Configure>", configure_lda_scroll_region)
        self.lda_canvas.bind("<Configure>", configure_lda_canvas_width)
        
        # Bind mousewheel
        def on_lda_mousewheel(event):
            self.lda_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        self.lda_canvas.bind_all("<MouseWheel>", on_lda_mousewheel)
        
        # LDA figure for topic words visualization
        self.lda_fig = Figure(figsize=(8, 6), dpi=100)
        self.lda_canvas_fig = FigureCanvasTkAgg(self.lda_fig, self.lda_content_frame)
        self.lda_canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # LDA topic distribution figure
        self.lda_dist_fig = Figure(figsize=(10, 6), dpi=100)
        self.lda_dist_canvas = FigureCanvasTkAgg(self.lda_dist_fig, self.lda_content_frame)
        self.lda_dist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def display_wordcloud(self, image: Image.Image):
        """Display a word cloud image."""
        from gui_utils import display_pil_image_in_label
        display_pil_image_in_label(self.wordcloud_label, image, max_size=(500, 400))
    
    def plot_frequency(self, words: List[str], frequencies: List[int], title: str = "Word Frequency"):
        """Plot word frequency chart."""
        self.freq_fig.clear()
        ax = self.freq_fig.add_subplot(111)
        ax.barh(words, frequencies)
        ax.set_xlabel('Frequency')
        ax.set_title(title)
        ax.invert_yaxis()
        self.freq_canvas.draw()
    
    def plot_similarity_scores(self, doc_ids: List[int], scores: List[float], 
                               titles: List[str], title: str = "Similarity Scores"):
        """Plot similarity scores."""
        self.sim_fig.clear()
        ax = self.sim_fig.add_subplot(111)
        ax.barh(range(len(titles)), scores)
        ax.set_yticks(range(len(titles)))
        ax.set_yticklabels([t[:30] for t in titles])
        ax.set_xlabel('Similarity Score')
        ax.set_title(title)
        ax.invert_yaxis()
        self.sim_canvas.draw()
    
    def plot_model_comparison(self, results_dict: Dict[str, List[Tuple[int, float]]],
                             document_titles: Dict[int, str], top_k: int = 5):
        """Plot model comparison for all 3 models."""
        self.comp_fig.clear()
        
        if not results_dict:
            return
        
        ax = self.comp_fig.add_subplot(111)
        
        # Ensure we have all 3 models
        model_order = ['VSM', 'BM25', 'Boolean']
        model_names = []
        top_scores = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for model_name in model_order:
            if model_name in results_dict and results_dict[model_name]:
                model_names.append(model_name)
                top_score = results_dict[model_name][0][1]
                top_scores.append(top_score)
        
        if model_names and top_scores:
            x_positions = list(range(len(model_names)))
            bars = ax.bar(x_positions, top_scores, alpha=0.7, color=colors[:len(model_names)])
            
            # Add value labels on bars
            for bar, score in zip(bars, top_scores):
                if score > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{score:.3f}', ha='center', va='bottom')
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels(model_names)
            ax.set_ylabel('Top Score')
            ax.set_title('Model Comparison - All Models')
            ax.grid(True, alpha=0.3, axis='y')
        
        self.comp_canvas.draw()
    
    def plot_lda_topics(self, lda_model, num_words: int = 10):
        """
        Plot LDA topics visualization.
        
        Args:
            lda_model: Trained LDA model
            num_words: Number of words per topic to display
        """
        from gensim.models import LdaModel
        
        if not lda_model:
            return
        
        self.lda_fig.clear()
        
        num_topics = lda_model.num_topics
        
        # Create subplots for each topic
        if num_topics <= 3:
            rows, cols = 1, num_topics
        elif num_topics <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        for topic_idx in range(num_topics):
            ax = self.lda_fig.add_subplot(rows, cols, topic_idx + 1)
            
            topic_words = lda_model.show_topic(topic_idx, topn=num_words)
            if topic_words:
                words, weights = zip(*topic_words)
                
                ax.barh(range(len(words)), weights, color='steelblue', alpha=0.7)
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words, fontsize=8)
                ax.set_xlabel('Weight')
                ax.set_title(f'Topic {topic_idx + 1}', fontsize=10, fontweight='bold')
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3, axis='x')
        
        self.lda_fig.suptitle('LDA Topic Clustering - Topic Words', fontsize=12, fontweight='bold')
        self.lda_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.lda_canvas_fig.draw()
    
    def plot_lda_topic_distribution(self, lda_model, corpus, document_titles: Dict[int, str] = None, 
                                    num_docs: int = 20):
        """
        Plot topic distribution across documents.
        
        Args:
            lda_model: Trained LDA model
            corpus: Document corpus (list of bow vectors)
            document_titles: Dictionary mapping doc_id to title
            num_docs: Number of documents to display
        """
        from gensim.models import LdaModel
        import pandas as pd
        import numpy as np
        
        if not lda_model or not corpus:
            return
        
        self.lda_dist_fig.clear()
        ax = self.lda_dist_fig.add_subplot(111)
        
        num_docs = min(num_docs, len(corpus))
        num_topics = lda_model.num_topics
        
        # Get topic distributions for documents
        topic_distributions = []
        doc_labels = []
        
        for doc_idx in range(num_docs):
            doc_topics = lda_model.get_document_topics(corpus[doc_idx])
            topic_dist = [0.0] * num_topics
            for topic_id, prob in doc_topics:
                topic_dist[topic_id] = prob
            topic_distributions.append(topic_dist)
            
            # Get document label
            if document_titles:
                # Find doc_id from index (assuming corpus index matches document order)
                # This is a simplification - you may need to adjust based on your data structure
                doc_labels.append(f"Doc {doc_idx + 1}")
            else:
                doc_labels.append(f"Doc {doc_idx + 1}")
        
        # Create DataFrame
        df = pd.DataFrame(topic_distributions, 
                         columns=[f'Topic {i+1}' for i in range(num_topics)],
                         index=doc_labels)
        
        # Plot stacked bar chart
        df.plot(kind='bar', stacked=True, ax=ax, colormap='tab20', width=0.8)
        ax.set_xlabel('Documents', fontsize=10)
        ax.set_ylabel('Topic Probability', fontsize=10)
        ax.set_title('Topic Distribution Across Documents', fontsize=12, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        self.lda_dist_fig.tight_layout()
        self.lda_dist_canvas.draw()
        
        # Update scroll region
        self.lda_content_frame.update_idletasks()
        self.lda_canvas.configure(scrollregion=self.lda_canvas.bbox("all"))


class EvaluationPanel(ttk.Frame):
    """Evaluation metrics panel."""
    
    def __init__(self, parent):
        """
        Initialize evaluation panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI components."""
        # Metrics display
        metrics_frame = ttk.LabelFrame(self, text="Evaluation Metrics", padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview for metrics - support 3 models
        columns = ("Metric", "VSM", "BM25", "Boolean")
        self.metrics_tree = ttk.Treeview(metrics_frame, columns=columns, show="headings", height=10)
        
        self.metrics_tree.heading("Metric", text="Metric")
        self.metrics_tree.heading("VSM", text="VSM")
        self.metrics_tree.heading("BM25", text="BM25")
        self.metrics_tree.heading("Boolean", text="Boolean")
        
        self.metrics_tree.column("Metric", width=150)
        self.metrics_tree.column("VSM", width=80)
        self.metrics_tree.column("BM25", width=80)
        self.metrics_tree.column("Boolean", width=80)
        
        scrollbar = ttk.Scrollbar(metrics_frame, orient=tk.VERTICAL, 
                                  command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscrollcommand=scrollbar.set)
        
        self.metrics_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Precision-Recall curve
        pr_frame = ttk.LabelFrame(self, text="Precision-Recall Curve", padding=10)
        pr_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.pr_fig = Figure(figsize=(5, 4), dpi=100)
        self.pr_canvas = FigureCanvasTkAgg(self.pr_fig, pr_frame)
        self.pr_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize with empty plot
        ax = self.pr_fig.add_subplot(111)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        self.pr_canvas.draw()
    
    def display_metrics(self, vsm_metrics: Dict[str, float], 
                       bm25_metrics: Dict[str, float],
                       boolean_metrics: Dict[str, float] = None):
        """
        Display evaluation metrics.
        
        Args:
            vsm_metrics: Dictionary of VSM metrics
            bm25_metrics: Dictionary of BM25 metrics
            boolean_metrics: Optional dictionary of Boolean metrics
        """
        # Clear existing
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
        
        # Add metrics
        all_keys = set(vsm_metrics.keys()) | set(bm25_metrics.keys())
        if boolean_metrics:
            all_keys |= set(boolean_metrics.keys())
        
        for key in sorted(all_keys):
            vsm_val = vsm_metrics.get(key, 0.0)
            bm25_val = bm25_metrics.get(key, 0.0)
            
            if boolean_metrics:
                bool_val = boolean_metrics.get(key, 0.0)
                self.metrics_tree.insert("", tk.END, values=(
                    key, format_score(vsm_val), format_score(bm25_val), format_score(bool_val)
                ))
            else:
                self.metrics_tree.insert("", tk.END, values=(
                    key, format_score(vsm_val), format_score(bm25_val)
                ))
    
    def plot_precision_recall(self, recalls: List[float], precisions: List[float], 
                             label: str = None):
        """Plot precision-recall curve."""
        self.pr_fig.clear()
        ax = self.pr_fig.add_subplot(111)
        
        if recalls and precisions and len(recalls) == len(precisions):
            ax.plot(recalls, precisions, marker='o', linewidth=2, markersize=4, label=label)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        if label:
            ax.legend()
        self.pr_canvas.draw()
    
    def plot_precision_recall_multiple(self, curves: Dict[str, Tuple[List[float], List[float]]]):
        """Plot multiple precision-recall curves for comparison."""
        self.pr_fig.clear()
        ax = self.pr_fig.add_subplot(111)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for idx, (model_name, (recalls, precisions)) in enumerate(curves.items()):
            if recalls and precisions and len(recalls) == len(precisions):
                color = colors[idx % len(colors)]
                ax.plot(recalls, precisions, marker='o', linewidth=2, markersize=3, 
                       label=model_name, color=color)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve - Model Comparison')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        if curves:
            ax.legend()
        self.pr_canvas.draw()


class PreprocessingPanel(ttk.Frame):
    """Panel for displaying text preprocessing steps for all documents."""
    
    def __init__(self, parent):
        """
        Initialize preprocessing panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.preprocessor = None
        self.vectorizer = None
        self.bow_vectorizer = None
        self.documents = []
        self.document_titles = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI components."""
        # Main scrollable frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(main_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Canvas for scrolling
        canvas = tk.Canvas(main_frame, yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=canvas.yview)
        
        # Frame inside canvas
        self.content_frame = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=self.content_frame, anchor="nw")
        
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def configure_canvas_width(event):
            canvas_width = event.width
            canvas.itemconfig(canvas_window, width=canvas_width)
        
        self.content_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", configure_canvas_width)
        
        # Bind mousewheel
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        self.canvas = canvas
    
    def display_documents(self, documents: List[Dict], document_titles: Dict[int, str],
                         preprocessor, vectorizer):
        """
        Display preprocessing steps for all documents.
        
        Args:
            documents: List of document dictionaries
            document_titles: Dictionary mapping doc_id to title
            preprocessor: TextPreprocessor instance
            vectorizer: DocumentVectorizer instance (TF-IDF)
        """
        # Clear existing content
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.documents = documents
        self.document_titles = document_titles
        
        if not documents or not preprocessor:
            ttk.Label(self.content_frame, text="No documents loaded or preprocessor not set.").pack(pady=20)
            return
        
        # Create BoW vectorizer
        from vectorization import Vectorizer
        self.bow_vectorizer = Vectorizer(method='bow',
                                         max_features=vectorizer.vectorizer.max_features if vectorizer else 5000,
                                         min_df=vectorizer.vectorizer.min_df if vectorizer else 2,
                                         max_df=vectorizer.vectorizer.max_df if vectorizer else 0.95)
        
        # Get all document texts
        doc_texts = [doc['text'] for doc in documents]
        
        # Fit BoW vectorizer
        preprocessed_texts = [preprocessor.preprocess(text) for text in doc_texts]
        self.bow_vectorizer.fit_transform(preprocessed_texts)
        
        # Display each document
        for doc in documents:
            self._display_document_preprocessing(doc, document_titles.get(doc['id'], f"Document {doc['id']}"))
        
        # Update scroll region
        self.content_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _display_document_preprocessing(self, doc: Dict, title: str):
        """Display preprocessing steps for a single document."""
        from preprocessing import TextPreprocessor
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        import string
        
        # Document frame
        doc_frame = ttk.LabelFrame(self.content_frame, text=title, padding=10)
        doc_frame.pack(fill=tk.X, padx=5, pady=5)
        
        text = doc['text']
        
        # 1. Original Text
        orig_frame = ttk.LabelFrame(doc_frame, text="1. Original Text", padding=5)
        orig_frame.pack(fill=tk.X, pady=2)
        orig_text = text[:300] + "..." if len(text) > 300 else text
        orig_label = ttk.Label(orig_frame, text=orig_text, wraplength=600, justify=tk.LEFT)
        orig_label.pack(anchor=tk.W)
        
        # 2. Tokenization
        tokens = word_tokenize(text)
        token_frame = ttk.LabelFrame(doc_frame, text="2. Tokenization", padding=5)
        token_frame.pack(fill=tk.X, pady=2)
        token_text = ", ".join(tokens[:50]) + ("..." if len(tokens) > 50 else "")
        token_label = ttk.Label(token_frame, text=token_text, wraplength=600, justify=tk.LEFT)
        token_label.pack(anchor=tk.W)
        
        # 3. Lowercasing
        lower_tokens = [t.lower() for t in tokens]
        lower_frame = ttk.LabelFrame(doc_frame, text="3. Lowercasing", padding=5)
        lower_frame.pack(fill=tk.X, pady=2)
        lower_text = ", ".join(lower_tokens[:50]) + ("..." if len(lower_tokens) > 50 else "")
        lower_label = ttk.Label(lower_frame, text=lower_text, wraplength=600, justify=tk.LEFT)
        lower_label.pack(anchor=tk.W)
        
        # 4. Stopwords Removal
        stop_words = set(stopwords.words('english'))
        no_stopwords = [t for t in lower_tokens if t.lower() not in stop_words and t not in string.punctuation]
        stop_frame = ttk.LabelFrame(doc_frame, text="4. Stopwords Removal", padding=5)
        stop_frame.pack(fill=tk.X, pady=2)
        stop_text = ", ".join(no_stopwords[:50]) + ("..." if len(no_stopwords) > 50 else "")
        stop_label = ttk.Label(stop_frame, text=stop_text, wraplength=600, justify=tk.LEFT)
        stop_label.pack(anchor=tk.W)
        
        # 5. Stemming/Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(t) for t in no_stopwords]
        lemma_frame = ttk.LabelFrame(doc_frame, text="5. Stemming/Lemmatization", padding=5)
        lemma_frame.pack(fill=tk.X, pady=2)
        lemma_text = ", ".join(lemmatized[:50]) + ("..." if len(lemmatized) > 50 else "")
        lemma_label = ttk.Label(lemma_frame, text=lemma_text, wraplength=600, justify=tk.LEFT)
        lemma_label.pack(anchor=tk.W)
        
        # 6. Bag-of-Words (BoW)
        preprocessed = self.preprocessor.preprocess(text)
        bow_vector = self.bow_vectorizer.transform([preprocessed])
        bow_frame = ttk.LabelFrame(doc_frame, text="6. Bag-of-Words (BoW)", padding=5)
        bow_frame.pack(fill=tk.X, pady=2)
        
        # Get top terms
        if hasattr(self.bow_vectorizer, 'vectorizer') and hasattr(self.bow_vectorizer.vectorizer, 'vocabulary_'):
            vocab = self.bow_vectorizer.vectorizer.vocabulary_
            inv_vocab = {v: k for k, v in vocab.items()}
            bow_array = bow_vector.toarray()[0]
            top_indices = bow_array.argsort()[-20:][::-1]
            top_terms = [(inv_vocab.get(idx, ""), int(bow_array[idx])) for idx in top_indices if bow_array[idx] > 0]
            bow_text = ", ".join([f"{term}({count})" for term, count in top_terms[:10]])
        else:
            bow_text = "BoW vector created (sparse representation)"
        bow_label = ttk.Label(bow_frame, text=bow_text, wraplength=600, justify=tk.LEFT)
        bow_label.pack(anchor=tk.W)
        
        # 7. TF-IDF
        if self.vectorizer:
            tfidf_vector = self.vectorizer.transform_query(text)
            tfidf_frame = ttk.LabelFrame(doc_frame, text="7. TF-IDF", padding=5)
            tfidf_frame.pack(fill=tk.X, pady=2)
            
            # Get top terms
            if hasattr(self.vectorizer, 'vectorizer') and hasattr(self.vectorizer.vectorizer, 'vectorizer'):
                vocab = self.vectorizer.vectorizer.vectorizer.vocabulary_
                inv_vocab = {v: k for k, v in vocab.items()}
                tfidf_array = tfidf_vector.toarray()[0]
                top_indices = tfidf_array.argsort()[-20:][::-1]
                top_terms = [(inv_vocab.get(idx, ""), float(tfidf_array[idx])) for idx in top_indices if tfidf_array[idx] > 0]
                tfidf_text = ", ".join([f"{term}({score:.3f})" for term, score in top_terms[:10]])
            else:
                tfidf_text = "TF-IDF vector created (sparse representation)"
            tfidf_label = ttk.Label(tfidf_frame, text=tfidf_text, wraplength=600, justify=tk.LEFT)
            tfidf_label.pack(anchor=tk.W)


class DatasetInfoPanel(ttk.Frame):
    """Dataset information panel."""
    
    def __init__(self, parent):
        """
        Initialize dataset info panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI components."""
        # Info display
        info_frame = ttk.LabelFrame(self, text="Dataset Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.info_text = tk.Text(info_frame, wrap=tk.WORD, height=15, 
                                font=("Courier", 9))
        info_scroll = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, 
                                   command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scroll.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Load button
        self.load_btn = ttk.Button(self, text="Load/Reload Dataset")
        self.load_btn.pack(pady=5)
    
    def display_info(self, stats: Dict):
        """
        Display dataset statistics.
        
        Args:
            stats: Dictionary of statistics
        """
        self.info_text.delete(1.0, tk.END)
        
        info_lines = [
            "Dataset Statistics",
            "=" * 50,
            f"Total Documents: {stats.get('total_documents', 0)}",
            f"Total Characters: {stats.get('total_characters', 0):,}",
            f"Total Words: {stats.get('total_words', 0):,}",
            f"Average Doc Length (chars): {stats.get('avg_doc_length_chars', 0):.2f}",
            f"Average Doc Length (words): {stats.get('avg_doc_length_words', 0):.2f}",
            f"Min Doc Length (words): {stats.get('min_doc_length_words', 0)}",
            f"Max Doc Length (words): {stats.get('max_doc_length_words', 0)}",
        ]
        
        self.info_text.insert(1.0, "\n".join(info_lines))
    
    def set_load_callback(self, callback: Callable):
        """Set callback for load button."""
        self.load_btn.config(command=callback)

