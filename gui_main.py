"""
Main GUI application for Information Retrieval system.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from typing import Optional, Dict, List, Tuple

from data_loader import DataLoader
from preprocessing import TextPreprocessor
from vectorization import DocumentVectorizer
from retrieval_models import RetrievalSystem
from evaluation import Evaluator, create_relevance_set
from visualization import (create_wordcloud, plot_word_frequency, 
                          perform_lda_topic_modeling, print_lda_topics)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import io

from gui_components import (SearchPanel, ResultsPanel, VisualizationPanel,
                           EvaluationPanel, DatasetInfoPanel, PreprocessingPanel)
from gui_utils import (create_progress_window, ThreadedTask, 
                     update_status_label, safe_tk_call)


class IRSystemGUI:
    """Main GUI application for Information Retrieval system."""
    
    def __init__(self, root):
        """
        Initialize the GUI application.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Information Retrieval System")
        self.root.geometry("1400x900")
        
        # System components
        self.data_loader: Optional[DataLoader] = None
        self.preprocessor: Optional[TextPreprocessor] = None
        self.vectorizer: Optional[DocumentVectorizer] = None
        self.retrieval_system: Optional[RetrievalSystem] = None
        self.evaluator = Evaluator()
        
        # Data storage
        self.documents: List[Dict] = []
        self.document_texts: List[str] = []
        self.document_ids: List[int] = []
        self.document_titles: Dict[int, str] = {}
        self.document_texts_dict: Dict[int, str] = {}
        
        # Current results
        self.current_results: Dict[str, List[Tuple[int, float]]] = {}
        
        # Setup UI
        self.setup_menu()
        self.setup_toolbar()
        self.setup_layout()
        self.setup_status_bar()
        
        # Initialize system
        self.root.after(100, self.initialize_system)
    
    def setup_menu(self):
        """Setup menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Dataset...", command=self.load_dataset, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Export Results...", command=self.export_results, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Run Evaluation", command=self.run_evaluation, accelerator="Ctrl+E")
        tools_menu.add_command(label="Generate Word Cloud", command=self.generate_wordcloud, accelerator="Ctrl+W")
        tools_menu.add_command(label="Topic Modeling", command=self.run_topic_modeling, accelerator="Ctrl+T")
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Compare Models", command=self.show_model_comparison)
        view_menu.add_command(label="Dataset Statistics", command=self.show_dataset_stats)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.load_dataset())
        self.root.bind('<Control-s>', lambda e: self.export_results())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
        self.root.bind('<Control-e>', lambda e: self.run_evaluation())
        self.root.bind('<Control-w>', lambda e: self.generate_wordcloud())
        self.root.bind('<Control-t>', lambda e: self.run_topic_modeling())
    
    def setup_toolbar(self):
        """Setup toolbar."""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="Load Dataset", command=self.load_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Search", command=self.focus_search).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Evaluate", command=self.run_evaluation).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Visualize", command=self.generate_wordcloud).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(toolbar, text="Clear", command=self.clear_all).pack(side=tk.LEFT, padx=2)
    
    def setup_layout(self):
        """Setup main layout."""
        # Main container
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Dataset info and search
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=1)
        
        # Dataset info
        self.dataset_panel = DatasetInfoPanel(left_panel)
        self.dataset_panel.pack(fill=tk.BOTH, expand=True)
        self.dataset_panel.set_load_callback(self.load_dataset)
        
        # Search panel
        search_frame = ttk.Frame(left_panel)
        search_frame.pack(fill=tk.X, pady=5)
        self.search_panel = SearchPanel(search_frame, on_search=self.perform_search)
        self.search_panel.pack(fill=tk.BOTH, expand=True)
        
        # Center panel - Results
        center_panel = ttk.Frame(main_container)
        main_container.add(center_panel, weight=2)
        
        # Results notebook
        results_notebook = ttk.Notebook(center_panel)
        results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # VSM results tab
        self.vsm_results = ResultsPanel(results_notebook, on_document_select=self.on_document_select)
        results_notebook.add(self.vsm_results, text="VSM Results")
        
        # BM25 results tab
        self.bm25_results = ResultsPanel(results_notebook, on_document_select=self.on_document_select)
        results_notebook.add(self.bm25_results, text="BM25 Results")
        
        # Boolean results tab
        self.boolean_results = ResultsPanel(results_notebook, on_document_select=self.on_document_select)
        results_notebook.add(self.boolean_results, text="Boolean Results")
        
        # Comparison tab
        self.comparison_frame = ttk.Frame(results_notebook)
        results_notebook.add(self.comparison_frame, text="Comparison")
        self.setup_comparison_view()
        
        # Right panel - Visualizations and evaluation
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=1)
        
        # Right panel notebook
        right_notebook = ttk.Notebook(right_panel)
        right_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Text Preprocessing tab
        self.preprocessing_panel = PreprocessingPanel(right_notebook)
        right_notebook.add(self.preprocessing_panel, text="Text Preprocessing")
        
        # Visualization tab
        self.viz_panel = VisualizationPanel(right_notebook)
        right_notebook.add(self.viz_panel, text="Visualizations")
        
        # Evaluation tab
        self.eval_panel = EvaluationPanel(right_notebook)
        right_notebook.add(self.eval_panel, text="Evaluation")
    
    def setup_comparison_view(self):
        """Setup model comparison view."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        
        self.comp_fig = Figure(figsize=(8, 6), dpi=100)
        self.comp_canvas = FigureCanvasTkAgg(self.comp_fig, self.comparison_frame)
        self.comp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_status_bar(self):
        """Setup status bar."""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_status(self, message: str):
        """Update status bar."""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def initialize_system(self):
        """Initialize the IR system."""
        self.update_status("Initializing system...")
        
        # Create progress window with determinate progress bar
        progress_window, progress_bar, status_label = create_progress_window(
            self.root, "Loading Documents", determinate=True
        )
        
        def update_progress(current: int, total: int, message: str):
            """Update progress in the progress window (thread-safe)."""
            def update_ui():
                if progress_window.winfo_exists():
                    # Update status label
                    status_label.config(text=f"{message} ({current}/{total})")
                    # Update progress bar
                    if progress_bar['mode'] == 'determinate':
                        progress_bar['value'] = (current / total) * 100
                    progress_window.update_idletasks()
            
            # Schedule update on main thread
            self.root.after(0, update_ui)
        
        def init_task():
            try:
                # Load data - use Wikipedia API (falls back to sample documents if needed)
                loader = DataLoader(max_documents=20)
                loader.load_dataset(progress_callback=update_progress)
                
                # Get documents
                docs = loader.get_documents()
                doc_texts = [doc['text'] for doc in docs]
                doc_ids = [doc['id'] for doc in docs]
                doc_titles = {doc['id']: doc['title'] for doc in docs}
                doc_texts_dict = {doc['id']: doc['text'] for doc in docs}
                
                # Initialize preprocessor
                preprocessor = TextPreprocessor(
                    lowercase=True,
                    remove_stopwords=True,
                    remove_punctuation=True,
                    lemmatize=True
                )
                
                # Initialize vectorizer
                vectorizer = DocumentVectorizer(
                    preprocessor=preprocessor,
                    method='tfidf',
                    max_features=5000,
                    min_df=2,
                    max_df=0.95
                )
                vectorizer.fit_transform(doc_texts)
                
                # Initialize retrieval system
                retrieval_system = RetrievalSystem(
                    documents=doc_texts,
                    document_ids=doc_ids,
                    preprocessor=preprocessor,
                    vectorizer=vectorizer
                )
                
                # Get stats
                stats = loader.inspect_dataset()
                
                # Train LDA model for visualization
                lda_model = None
                lda_corpus = None
                try:
                    from gensim.corpora import Dictionary
                    from gensim.models import LdaModel
                    
                    # Prepare documents for LDA (tokenized)
                    tokenized_docs = []
                    for text in doc_texts:
                        preprocessed = preprocessor.preprocess(text)
                        tokens = preprocessed.split()
                        if len(tokens) > 0:
                            tokenized_docs.append(tokens)
                    
                    if len(tokenized_docs) > 0:
                        # Create dictionary and corpus
                        dictionary = Dictionary(tokenized_docs)
                        dictionary.filter_extremes(no_below=2, no_above=0.95)
                        lda_corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
                        
                        # Train LDA model
                        num_topics = min(5, len(docs) // 2) if len(docs) > 10 else 3
                        num_topics = max(2, num_topics)  # At least 2 topics
                        
                        lda_model = LdaModel(corpus=lda_corpus, id2word=dictionary, 
                                             num_topics=num_topics, random_state=42,
                                             passes=10, alpha='auto', per_word_topics=True)
                except Exception as e:
                    print(f"Error training LDA model: {e}")
                    import traceback
                    traceback.print_exc()
                
                return {
                    'loader': loader,
                    'preprocessor': preprocessor,
                    'vectorizer': vectorizer,
                    'retrieval_system': retrieval_system,
                    'docs': docs,
                    'doc_texts': doc_texts,
                    'doc_ids': doc_ids,
                    'doc_titles': doc_titles,
                    'doc_texts_dict': doc_texts_dict,
                    'stats': stats,
                    'lda_model': lda_model,
                    'lda_corpus': lda_corpus
                }
            except Exception as e:
                return {'error': str(e)}
        
        def on_complete(result):
            # Close progress window
            if progress_window.winfo_exists():
                progress_window.destroy()
            
            if result and 'error' in result:
                messagebox.showerror("Error", f"Failed to initialize system: {result['error']}")
                self.update_status("Initialization failed")
            elif result:
                self.data_loader = result['loader']
                self.preprocessor = result['preprocessor']
                self.vectorizer = result['vectorizer']
                self.retrieval_system = result['retrieval_system']
                self.documents = result['docs']
                self.document_texts = result['doc_texts']
                self.document_ids = result['doc_ids']
                self.document_titles = result['doc_titles']
                self.document_texts_dict = result['doc_texts_dict']
                
                # Update UI
                self.dataset_panel.display_info(result['stats'])
                
                # Update preprocessing panel
                self.preprocessing_panel.display_documents(
                    self.documents,
                    self.document_titles,
                    self.preprocessor,
                    self.vectorizer
                )
                
                # Display LDA topic clustering in visualization panel
                if result.get('lda_model') and result.get('lda_corpus'):
                    self.viz_panel.plot_lda_topics(result['lda_model'], num_words=10)
                    self.viz_panel.plot_lda_topic_distribution(
                        result['lda_model'], 
                        result['lda_corpus'],
                        result['doc_titles'],
                        num_docs=min(20, len(self.documents))
                    )
                
                self.update_status(f"System ready - {len(self.documents)} documents loaded")
                messagebox.showinfo("Success", f"System initialized successfully!\nLoaded {len(self.documents)} documents.")
            else:
                self.update_status("Initialization failed")
        
        task = ThreadedTask(init_task, on_complete, self.root)
        task.start()
    
    def load_dataset(self):
        """Load or reload dataset."""
        if not self.retrieval_system:
            self.initialize_system()
        else:
            response = messagebox.askyesno("Reload Dataset", 
                                          "Reload dataset? This will reinitialize the system.")
            if response:
                self.initialize_system()
    
    def perform_search(self, query: str, model: str, top_k: int):
        """Perform search."""
        if not self.retrieval_system:
            messagebox.showwarning("Not Ready", "System not initialized. Please wait...")
            return
        
        self.update_status(f"Searching: {query}...")
        
        def search_task():
            try:
                results = {}
                
                # Always search with all models for comparison
                try:
                    vsm_res = self.retrieval_system.search_vsm(query, top_k=top_k)
                    results['VSM'] = vsm_res if vsm_res else []
                except Exception as e:
                    print(f"VSM search error: {e}")
                    import traceback
                    traceback.print_exc()
                    results['VSM'] = []
                
                try:
                    bm25_res = self.retrieval_system.search_bm25(query, top_k=top_k)
                    results['BM25'] = bm25_res if bm25_res else []
                except Exception as e:
                    print(f"BM25 search error: {e}")
                    results['BM25'] = []
                
                try:
                    bool_res = self.retrieval_system.search_boolean(query, top_k=top_k)
                    # Convert to (doc_id, 1.0) format for consistency
                    bool_res_with_scores = [(doc_id, 1.0) for doc_id in bool_res]
                    results['Boolean'] = bool_res_with_scores
                except Exception as e:
                    print(f"Boolean search error: {e}")
                    results['Boolean'] = []
                
                return results
            except Exception as e:
                print(f"Search task error: {e}")
                import traceback
                traceback.print_exc()
                return {'error': str(e)}
        
        def on_complete(results):
            if results and 'error' in results:
                messagebox.showerror("Search Error", results['error'])
                self.update_status("Search failed")
            else:
                self.current_results = results
                
                # Display results
                if 'VSM' in results and results['VSM']:
                    self.vsm_results.display_results(
                        results['VSM'], self.document_titles, self.document_texts_dict
                    )
                else:
                    self.vsm_results.clear()
                
                if 'BM25' in results and results['BM25']:
                    self.bm25_results.display_results(
                        results['BM25'], self.document_titles, self.document_texts_dict
                    )
                else:
                    self.bm25_results.clear()
                
                if 'Boolean' in results and results['Boolean']:
                    self.boolean_results.display_results(
                        results['Boolean'], self.document_titles, self.document_texts_dict
                    )
                else:
                    self.boolean_results.clear()
                
                # Update comparison
                self.update_comparison_view()
                
                # Update visualizations
                if 'VSM' in results and results['VSM']:
                    self.update_similarity_plot(results['VSM'])
                
                # Update frequency plot
                self.update_frequency_plot(query)
                
                # Update model comparison visualization
                if self.current_results:
                    self.update_model_comparison_viz()
                
                total_results = sum(len(r) for r in results.values())
                self.update_status(f"Found {total_results} results")
        
        task = ThreadedTask(search_task, on_complete, self.root)
        task.start()
    
    def update_comparison_view(self):
        """Update model comparison view."""
        if not self.current_results:
            return
        
        try:
            self.comp_fig.clear()
            ax = self.comp_fig.add_subplot(111)
            
            model_names = []
            top_scores = []
            
            # Get top score from each model
            for model_name in ['VSM', 'BM25', 'Boolean']:
                if model_name in self.current_results and self.current_results[model_name]:
                    model_names.append(model_name)
                    top_score = self.current_results[model_name][0][1] if self.current_results[model_name] else 0.0
                    top_scores.append(top_score)
            
            if model_names and top_scores:
                x_positions = list(range(len(model_names)))
                bars = ax.bar(x_positions, top_scores, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                
                # Add value labels on bars
                for bar, score in zip(bars, top_scores):
                    if score > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{score:.3f}', ha='center', va='bottom')
                
                ax.set_xticks(x_positions)
                ax.set_xticklabels(model_names)
                ax.set_ylabel('Top Score')
                ax.set_title('Model Comparison - Top Results')
                ax.grid(True, alpha=0.3, axis='y')
            
            self.comp_canvas.draw()
        except Exception as e:
            print(f"Error updating comparison view: {e}")
    
    def update_similarity_plot(self, results: List[Tuple[int, float]]):
        """Update similarity scores plot."""
        if not results:
            return
        
        try:
            top_results = results[:10]
            if not top_results:
                return
            
            doc_ids, scores = zip(*top_results)
            titles = [self.document_titles.get(doc_id, f"Doc {doc_id}") for doc_id in doc_ids]
            
            self.viz_panel.plot_similarity_scores(
                list(doc_ids), list(scores), titles, "Similarity Scores"
            )
        except Exception as e:
            print(f"Error updating similarity plot: {e}")
            import traceback
            traceback.print_exc()
    
    def update_model_comparison_viz(self):
        """Update model comparison in visualization panel."""
        if not self.current_results:
            return
        
        try:
            # Get top 5 results from each model - ensure all 3 models
            comparison_data = {}
            for model_name in ['VSM', 'BM25', 'Boolean']:
                if model_name in self.current_results and self.current_results[model_name]:
                    comparison_data[model_name] = self.current_results[model_name][:5]
            
            if comparison_data:
                self.viz_panel.plot_model_comparison(
                    comparison_data, self.document_titles, top_k=5
                )
        except Exception as e:
            print(f"Error updating model comparison viz: {e}")
            import traceback
            traceback.print_exc()
    
    def update_frequency_plot(self, query: str):
        """Update frequency plot based on query."""
        if not self.document_texts or not query:
            return
        
        try:
            # Get documents that match the query
            query_terms = query.lower().split()
            matching_docs = []
            
            for doc_text in self.document_texts[:20]:  # Use first 20 docs
                doc_lower = doc_text.lower()
                if any(term in doc_lower for term in query_terms):
                    matching_docs.append(doc_text)
            
            if matching_docs:
                # Combine matching documents
                combined_text = ' '.join(matching_docs)
                preprocessed = self.preprocessor.preprocess(combined_text)
                
                # Count word frequencies
                from collections import Counter
                words = preprocessed.split()
                word_freq = Counter(words)
                
                # Get top 20 words
                top_words = word_freq.most_common(20)
                if top_words:
                    words_list, freqs_list = zip(*top_words)
                    self.viz_panel.plot_frequency(
                        list(words_list), list(freqs_list), 
                        f"Word Frequency (Query: {query})"
                    )
        except Exception as e:
            print(f"Error updating frequency plot: {e}")
    
    def on_document_select(self, doc_id: int):
        """Handle document selection."""
        if doc_id in self.document_texts_dict:
            text = self.document_texts_dict[doc_id]
            title = self.document_titles.get(doc_id, f"Document {doc_id}")
            
            # Show in a new window
            doc_window = tk.Toplevel(self.root)
            doc_window.title(title)
            doc_window.geometry("800x600")
            
            text_widget = tk.Text(doc_window, wrap=tk.WORD, font=("Arial", 11))
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text_widget.insert(1.0, text)
            text_widget.config(state=tk.DISABLED)
    
    def generate_wordcloud(self):
        """Generate and display word cloud."""
        if not self.document_texts:
            messagebox.showwarning("No Data", "Please load dataset first.")
            return
        
        self.update_status("Generating word cloud...")
        
        def generate_task():
            try:
                combined_text = ' '.join(self.document_texts[:50])
                preprocessed = self.preprocessor.preprocess(combined_text)
                
                wordcloud = WordCloud(width=800, height=400, background_color='white',
                                    max_words=100, colormap='viridis').generate(preprocessed)
                
                return wordcloud.to_image()
            except Exception as e:
                return {'error': str(e)}
        
        def on_complete(image):
            if isinstance(image, dict) and 'error' in image:
                messagebox.showerror("Error", image['error'])
                self.update_status("Word cloud generation failed")
            else:
                self.viz_panel.display_wordcloud(image)
                self.update_status("Word cloud generated")
        
        task = ThreadedTask(generate_task, on_complete, self.root)
        task.start()
    
    def run_evaluation(self):
        """Run evaluation on sample queries."""
        if not self.retrieval_system:
            messagebox.showwarning("Not Ready", "System not initialized.")
            return
        
        sample_queries = ["artificial intelligence", "cybersecurity", "data science", "machine learning"]
        
        self.update_status("Running evaluation...")
        
        def eval_task():
            try:
                all_metrics = {'VSM': [], 'BM25': [], 'Boolean': []}
                pr_curves = {'VSM': ([], []), 'BM25': ([], []), 'Boolean': ([], [])}
                
                for query in sample_queries:
                    vsm_res = [doc_id for doc_id, _ in 
                              self.retrieval_system.search_vsm(query, top_k=20)]
                    bm25_res = [doc_id for doc_id, _ in 
                               self.retrieval_system.search_bm25(query, top_k=20)]
                    
                    try:
                        bool_res = self.retrieval_system.search_boolean(query, top_k=20)
                    except:
                        bool_res = []
                    
                    relevant = create_relevance_set(query, self.documents)
                    
                    if len(relevant) > 0:
                        vsm_metrics = self.evaluator.evaluate_query(vsm_res, relevant)
                        bm25_metrics = self.evaluator.evaluate_query(bm25_res, relevant)
                        bool_metrics = self.evaluator.evaluate_query(bool_res, relevant)
                        
                        all_metrics['VSM'].append(vsm_metrics)
                        all_metrics['BM25'].append(bm25_metrics)
                        all_metrics['Boolean'].append(bool_metrics)
                        
                        # Generate precision-recall curves
                        from evaluation import precision_at_k, recall_at_k
                        
                        # VSM curve
                        vsm_recalls, vsm_precisions = [], []
                        for k in range(1, min(len(vsm_res), 20) + 1):
                            prec = precision_at_k(vsm_res, relevant, k)
                            rec = recall_at_k(vsm_res, relevant, k)
                            vsm_precisions.append(prec)
                            vsm_recalls.append(rec)
                        if vsm_recalls and vsm_precisions:
                            pr_curves['VSM'] = (vsm_recalls, vsm_precisions)
                        
                        # BM25 curve
                        bm25_recalls, bm25_precisions = [], []
                        for k in range(1, min(len(bm25_res), 20) + 1):
                            prec = precision_at_k(bm25_res, relevant, k)
                            rec = recall_at_k(bm25_res, relevant, k)
                            bm25_precisions.append(prec)
                            bm25_recalls.append(rec)
                        if bm25_recalls and bm25_precisions:
                            pr_curves['BM25'] = (bm25_recalls, bm25_precisions)
                        
                        # Boolean curve
                        bool_recalls, bool_precisions = [], []
                        for k in range(1, min(len(bool_res), 20) + 1):
                            prec = precision_at_k(bool_res, relevant, k)
                            rec = recall_at_k(bool_res, relevant, k)
                            bool_precisions.append(prec)
                            bool_recalls.append(rec)
                        if bool_recalls and bool_precisions:
                            pr_curves['Boolean'] = (bool_recalls, bool_precisions)
                
                # Calculate averages
                avg_metrics = {}
                for model_name, metrics_list in all_metrics.items():
                    if metrics_list:
                        avg_metrics[model_name] = {
                            key: sum(m.get(key, 0) for m in metrics_list) / len(metrics_list)
                            for key in metrics_list[0].keys()
                        }
                
                return {'metrics': avg_metrics, 'pr_curves': pr_curves}
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {'error': str(e)}
        
        def on_complete(result):
            if isinstance(result, dict) and 'error' in result:
                messagebox.showerror("Error", result['error'])
                self.update_status("Evaluation failed")
            else:
                if result and 'metrics' in result:
                    avg_metrics = result['metrics']
                    pr_curves = result.get('pr_curves', {})
                    
                    self.eval_panel.display_metrics(
                        avg_metrics.get('VSM', {}),
                        avg_metrics.get('BM25', {}),
                        avg_metrics.get('Boolean', {})
                    )
                    
                    # Plot precision-recall curves
                    if pr_curves:
                        # Convert to format expected by plot function
                        curves_dict = {}
                        for model_name, (recalls, precisions) in pr_curves.items():
                            if recalls and precisions:
                                curves_dict[model_name] = (recalls, precisions)
                        
                        if curves_dict:
                            self.eval_panel.plot_precision_recall_multiple(curves_dict)
                    
                    self.update_status("Evaluation completed")
                else:
                    messagebox.showinfo("Info", "No metrics calculated.")
        
        task = ThreadedTask(eval_task, on_complete, self.root)
        task.start()
    
    def run_topic_modeling(self):
        """Run LDA topic modeling."""
        if not self.retrieval_system:
            messagebox.showwarning("Not Ready", "System not initialized.")
            return
        
        self.update_status("Running topic modeling...")
        
        def topic_task():
            try:
                preprocessed_docs = self.retrieval_system.preprocessed_documents[:100]
                lda_model = perform_lda_topic_modeling(preprocessed_docs, num_topics=5)
                return lda_model
            except Exception as e:
                return {'error': str(e)}
        
        def on_complete(lda_model):
            if isinstance(lda_model, dict) and 'error' in lda_model:
                messagebox.showerror("Error", lda_model['error'])
                self.update_status("Topic modeling failed")
            else:
                # Visualize topics
                self.viz_panel.freq_fig.clear()
                ax = self.viz_panel.freq_fig.add_subplot(111)
                
                num_topics = lda_model.num_topics
                for topic_idx in range(num_topics):
                    topic_words = lda_model.show_topic(topic_idx, topn=10)
                    words, weights = zip(*topic_words) if topic_words else ([], [])
                    
                    # Plot in subplot
                    if topic_idx == 0:
                        ax.barh(range(len(words)), weights)
                        ax.set_yticks(range(len(words)))
                        ax.set_yticklabels(words)
                        ax.set_title(f'Topic {topic_idx + 1}')
                        ax.invert_yaxis()
                
                self.viz_panel.freq_canvas.draw()
                self.update_status("Topic modeling completed")
                messagebox.showinfo("Success", "Topic modeling completed. Check visualizations tab.")
        
        task = ThreadedTask(topic_task, on_complete, self.root)
        task.start()
    
    def show_model_comparison(self):
        """Show model comparison window."""
        if not self.current_results:
            messagebox.showinfo("No Results", "Please perform a search first.")
            return
        # Comparison is already shown in the comparison tab
        messagebox.showinfo("Info", "Model comparison is available in the 'Comparison' tab.")
    
    def show_dataset_stats(self):
        """Show dataset statistics."""
        if not self.data_loader:
            messagebox.showinfo("No Data", "Dataset not loaded.")
            return
        
        stats = self.data_loader.inspect_dataset()
        stats_text = "\n".join([f"{k}: {v}" for k, v in stats.items()])
        messagebox.showinfo("Dataset Statistics", stats_text)
    
    def export_results(self):
        """Export search results to file."""
        if not self.current_results:
            messagebox.showinfo("No Results", "No results to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("Information Retrieval System - Search Results\n")
                    f.write("=" * 60 + "\n\n")
                    
                    for model_name, results in self.current_results.items():
                        f.write(f"\n{model_name} Results:\n")
                        f.write("-" * 60 + "\n")
                        for rank, (doc_id, score) in enumerate(results, 1):
                            title = self.document_titles.get(doc_id, f"Document {doc_id}")
                            f.write(f"{rank}. [{score:.4f}] {title}\n")
                
                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")
    
    def focus_search(self):
        """Focus on search input."""
        if hasattr(self, 'search_panel'):
            self.search_panel.query_entry.focus()
    
    def clear_all(self):
        """Clear all results and visualizations."""
        self.vsm_results.clear()
        self.bm25_results.clear()
        self.boolean_results.clear()
        self.current_results = {}
        self.update_status("Cleared all results")
    
    def show_about(self):
        """Show about dialog."""
        about_text = """Information Retrieval System
        
A comprehensive IR system with:
- Vector Space Model (VSM)
- BM25 Ranking
- Boolean Retrieval
- Evaluation Metrics
- Visualizations
- Topic Modeling

Group Members:
- 1. Mickel Wassef (22010449)
- 2. Amr Khaled (2206159)
- 3. Abdelrahman Jayasundara (2206147)
- 4. Loay Salah (2206155)

Version 1.0"""
        messagebox.showinfo("About", about_text)
    
    def show_shortcuts(self):
        """Show keyboard shortcuts."""
        shortcuts = """Keyboard Shortcuts:

Enter - Perform search
Ctrl+O - Load dataset
Ctrl+E - Run evaluation
Ctrl+V - Generate visualizations
Ctrl+Q - Exit"""
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)


def main():
    """Main entry point."""
    root = tk.Tk()
    app = IRSystemGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

