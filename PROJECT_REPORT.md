# Information Retrieval and Text Analytics System
## Detailed Project Report

---

## Executive Summary

This project implements a comprehensive Information Retrieval (IR) system with advanced text preprocessing, multiple retrieval models, evaluation metrics, and interactive visualization capabilities. The system is designed to retrieve relevant documents from a Wikipedia dataset (filtered for tech-related topics: cybersecurity, AI, and data science) based on user queries using three different retrieval models: Vector Space Model (VSM), Boolean Retrieval, and BM25.

The system features a modern graphical user interface (GUI) built with Tkinter, providing an intuitive way to interact with all system functionalities including search, visualization, evaluation, and text preprocessing analysis.

---

## 1. Project Overview

### 1.1 Purpose
The primary purpose of this project is to build a complete Information Retrieval system that demonstrates:
- Text preprocessing techniques
- Multiple retrieval model implementations
- Performance evaluation methodologies
- Data visualization and topic modeling
- User-friendly interface design

### 1.2 Scope
The system handles:
- **Data Collection**: Loading Wikipedia articles via API (filtered for tech topics)
- **Text Preprocessing**: Complete pipeline from raw text to processed tokens
- **Vectorization**: Bag-of-Words (BoW) and TF-IDF representations
- **Retrieval**: Three different retrieval models for document ranking
- **Evaluation**: Comprehensive metrics for model comparison
- **Visualization**: Multiple visualization types for insights
- **Topic Modeling**: LDA-based topic clustering

### 1.3 Dataset
- **Source**: Wikipedia API (real-time fetching)
- **Topics**: Cybersecurity, Artificial Intelligence, Data Science
- **Size**: Configurable (default: 20-100 documents)
- **Content**: Full Wikipedia article texts with titles

---

## 2. Objectives

### 2.1 Primary Objectives
1. ✅ Implement a complete text preprocessing pipeline
2. ✅ Build and compare multiple retrieval models (VSM, Boolean, BM25)
3. ✅ Evaluate model performance using standard IR metrics
4. ✅ Create interactive visualizations for data insights
5. ✅ Develop a user-friendly GUI application
6. ✅ Implement topic modeling using LDA

### 2.2 Secondary Objectives
1. ✅ Real-time data loading from Wikipedia API
2. ✅ Comprehensive text preprocessing visualization
3. ✅ Model comparison and evaluation
4. ✅ Export capabilities for results
5. ✅ Progress tracking for long operations

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GUI Application Layer                      │
│  (gui_main.py, gui_components.py, gui_utils.py)            │
└──────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────┴──────────────────────────────────────┐
│                  Core System Components                      │
├──────────────────────────────────────────────────────────────┤
│  • Data Loader (data_loader.py)                             │
│  • Text Preprocessor (preprocessing.py)                     │
│  • Vectorizer (vectorization.py)                            │
│  • Retrieval Models (retrieval_models.py)                   │
│  • Evaluator (evaluation.py)                                │
│  • Visualizer (visualization.py)                            │
└──────────────────────────────────────────────────────────────┘
                        │
┌───────────────────────┴──────────────────────────────────────┐
│                    Data Layer                                 │
│  • Wikipedia API / HuggingFace Datasets                      │
│  • Document Storage (in-memory)                              │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Component Breakdown

#### 3.2.1 Data Loading Module (`data_loader.py`)
- **Functionality**: 
  - Fetches Wikipedia articles via Wikipedia API
  - Filters articles by tech-related keywords
  - Falls back to HuggingFace datasets if API fails
  - Provides progress callbacks for GUI
- **Key Features**:
  - Real-time article fetching
  - Topic filtering (cybersecurity, AI, data science)
  - Rate limiting to avoid API throttling
  - Document validation and filtering

#### 3.2.2 Text Preprocessing Module (`preprocessing.py`)
- **Pipeline Steps**:
  1. **Tokenization**: Splits text into individual words
  2. **Lowercasing**: Converts all text to lowercase
  3. **Stopwords Removal**: Removes common non-informative words
  4. **Punctuation Removal**: Strips punctuation marks
  5. **Stemming/Lemmatization**: Reduces words to root forms
- **Implementation**: Uses NLTK library for NLP operations

#### 3.2.3 Vectorization Module (`vectorization.py`)
- **Methods**:
  - **Bag-of-Words (BoW)**: Term frequency representation
  - **TF-IDF**: Term Frequency-Inverse Document Frequency weighting
- **Features**:
  - Configurable vocabulary size
  - Min/max document frequency filtering
  - Sparse matrix representation for efficiency

#### 3.2.4 Retrieval Models Module (`retrieval_models.py`)
- **Vector Space Model (VSM)**:
  - Uses cosine similarity for document-query matching
  - TF-IDF vector representation
  - Returns ranked results by similarity score
  
- **Boolean Retrieval Model**:
  - Supports AND, OR, NOT operators
  - Exact term matching
  - Set-based operations
  
- **BM25 (Best Matching 25)**:
  - Probabilistic ranking function
  - Handles term frequency and document length normalization
  - Industry-standard retrieval algorithm

#### 3.2.5 Evaluation Module (`evaluation.py`)
- **Metrics**:
  - **Precision@k**: Fraction of retrieved documents that are relevant
  - **Recall@k**: Fraction of relevant documents that are retrieved
  - **Mean Average Precision (MAP)**: Average precision across all queries
- **Features**:
  - Relevance set creation
  - Per-query and aggregate metrics
  - Precision-Recall curve generation

#### 3.2.6 Visualization Module (`visualization.py`)
- **Visualizations**:
  - Word clouds for top keywords
  - Frequency distribution charts
  - Document-query similarity scores
  - Model comparison charts
  - LDA topic modeling visualizations
  - Precision-Recall curves

---

## 4. Implementation Details

### 4.1 Text Preprocessing Pipeline

The preprocessing pipeline transforms raw text into a format suitable for retrieval:

```python
Raw Text → Tokenization → Lowercasing → Stopwords Removal 
→ Punctuation Removal → Lemmatization → Processed Text
```

**Example Transformation**:
- **Input**: "The Artificial Intelligence system processes data efficiently."
- **Output**: "artificial intelligence system process data efficiently"

### 4.2 Vectorization Process

#### Bag-of-Words (BoW)
- Creates a vocabulary from all documents
- Represents each document as a vector of term frequencies
- Example: `[0, 2, 1, 0, 3, ...]` where each index represents a term

#### TF-IDF
- Calculates term frequency (TF) × inverse document frequency (IDF)
- Formula: `TF-IDF(t,d) = TF(t,d) × log(N/DF(t))`
- Weights rare terms higher than common terms

### 4.3 Retrieval Model Algorithms

#### Vector Space Model (VSM)
```
1. Convert query to TF-IDF vector
2. Calculate cosine similarity with all document vectors
3. Rank documents by similarity score
4. Return top-k results
```

#### Boolean Retrieval
```
1. Parse query for AND, OR, NOT operators
2. Find documents containing query terms
3. Apply set operations (intersection, union, difference)
4. Return matching documents
```

#### BM25
```
1. For each query term:
   - Calculate term frequency in document
   - Apply BM25 scoring formula
2. Sum scores across all query terms
3. Rank documents by total score
4. Return top-k results
```

### 4.4 GUI Architecture

The GUI is built using Tkinter with a modular design:

- **Main Window** (`IRSystemGUI`): Orchestrates all components
- **Search Panel**: Query input and model selection
- **Results Panel**: Displays search results in tabbed view
- **Visualization Panel**: Multiple visualization tabs
- **Evaluation Panel**: Metrics display and comparison
- **Preprocessing Panel**: Step-by-step preprocessing visualization
- **Dataset Info Panel**: Dataset statistics and information

**Threading**: Long-running operations (data loading, LDA training) run in background threads to keep UI responsive.

---

## 5. Features

### 5.1 Search Features
- ✅ Multi-model search (VSM, BM25, Boolean)
- ✅ Configurable result limit (or show all results)
- ✅ Real-time search execution
- ✅ Results displayed with scores and titles
- ✅ Document preview functionality

### 5.2 Visualization Features
- ✅ **Word Cloud**: Visual representation of top keywords
- ✅ **Frequency Charts**: Bar charts for word frequencies
- ✅ **Similarity Scores**: Visual comparison of document-query similarities
- ✅ **Model Comparison**: Side-by-side comparison of all three models
- ✅ **LDA Topics**: 
  - Topic words visualization (bar charts)
  - Topic distribution across documents (stacked bar chart)

### 5.3 Evaluation Features
- ✅ Precision@k calculation
- ✅ Recall@k calculation
- ✅ Mean Average Precision (MAP)
- ✅ Precision-Recall curves for all models
- ✅ Comparison tables

### 5.4 Preprocessing Analysis
- ✅ Step-by-step preprocessing visualization
- ✅ Original text snippets
- ✅ Tokenization results
- ✅ Lowercasing transformation
- ✅ Stopwords removal results
- ✅ Stemming/Lemmatization output
- ✅ BoW representation (top terms)
- ✅ TF-IDF representation (top weighted terms)
- ✅ LDA topic assignment per document

### 5.5 User Interface Features
- ✅ Modern tabbed interface
- ✅ Progress bars for long operations
- ✅ Keyboard shortcuts
- ✅ Export functionality
- ✅ Status bar with real-time updates
- ✅ Responsive layout

---

## 6. Methodology

### 6.1 Data Collection Methodology
1. **Wikipedia API Integration**:
   - Search for tech-related articles using keywords
   - Filter results by relevance (keyword matching in title/content)
   - Limit document count to manage processing time
   - Handle rate limiting and errors gracefully

2. **Data Validation**:
   - Minimum document length requirement (200 characters)
   - Disambiguation page filtering
   - Duplicate detection

### 6.2 Preprocessing Methodology
- **Tokenization**: NLTK's `word_tokenize()` for accurate word splitting
- **Stopwords**: NLTK's English stopwords list
- **Lemmatization**: WordNet lemmatizer for morphological normalization
- **Consistency**: Same preprocessing applied to queries and documents

### 6.3 Retrieval Methodology
- **Query Processing**: Same preprocessing pipeline as documents
- **Vectorization**: Consistent vocabulary between documents and queries
- **Ranking**: Score-based ranking for VSM and BM25
- **Set Operations**: Efficient boolean operations for Boolean model

### 6.4 Evaluation Methodology
- **Relevance Sets**: Manual or automatic creation
- **Metrics Calculation**: Standard IR evaluation metrics
- **Comparison**: Side-by-side model performance analysis
- **Visualization**: Precision-Recall curves for detailed analysis

---

## 7. Technologies and Libraries

### 7.1 Core Libraries
- **Python 3.8+**: Programming language
- **scikit-learn (≥1.3.0)**: Machine learning and vectorization
- **NLTK (≥3.8)**: Natural language processing
- **Gensim (≥4.3.0)**: Topic modeling (LDA)
- **NumPy (≥1.24.0)**: Numerical computations
- **Pandas (≥2.0.0)**: Data manipulation

### 7.2 Visualization Libraries
- **Matplotlib (≥3.7.0)**: Plotting and visualization
- **Seaborn (≥0.12.0)**: Statistical visualizations
- **Wordcloud (≥1.9.0)**: Word cloud generation

### 7.3 GUI Libraries
- **Tkinter**: Built-in Python GUI framework
- **PIL/Pillow**: Image processing for word clouds

### 7.4 Data Libraries
- **HuggingFace Datasets (≥2.14.0)**: Dataset loading
- **Wikipedia (≥1.4.0)**: Wikipedia API access

### 7.5 Development Tools
- **Jupyter (≥1.0.0)**: Interactive notebooks
- **ipykernel (≥6.25.0)**: Jupyter kernel

---

## 8. System Workflow

### 8.1 Initialization Flow
```
1. User launches GUI application
2. System automatically loads documents from Wikipedia API
3. Progress bar shows loading status
4. Documents are preprocessed
5. Vectorizers are fitted
6. Retrieval models are initialized
7. LDA model is trained
8. Visualizations are generated
9. System ready for queries
```

### 8.2 Search Flow
```
1. User enters query
2. User selects retrieval model (or all models)
3. User sets result limit (optional)
4. System preprocesses query
5. Selected model(s) execute search
6. Results are ranked and displayed
7. Visualizations update automatically
```

### 8.3 Evaluation Flow
```
1. User provides relevance judgments (or uses automatic)
2. System calculates metrics for each model
3. Precision@k, Recall@k, MAP computed
4. Precision-Recall curves generated
5. Results displayed in comparison table
6. Visualizations updated
```

---

## 9. Results and Evaluation

### 9.1 Model Performance Characteristics

#### Vector Space Model (VSM)
- **Strengths**:
  - Good for semantic similarity
  - Handles partial matches well
  - Fast retrieval with sparse matrices
- **Use Cases**: General-purpose search, similarity-based retrieval

#### Boolean Retrieval
- **Strengths**:
  - Exact matching
  - Supports complex logical queries
  - Fast for small result sets
- **Use Cases**: Structured queries, exact term matching

#### BM25
- **Strengths**:
  - Industry-standard algorithm
  - Handles term frequency and document length
  - Good ranking quality
- **Use Cases**: Production search systems, ranked retrieval

### 9.2 Evaluation Metrics

The system provides comprehensive evaluation through:
- **Precision@k**: Measures accuracy of top-k results
- **Recall@k**: Measures coverage of relevant documents
- **MAP**: Overall system performance metric
- **Precision-Recall Curves**: Detailed performance visualization

### 9.3 Visualization Insights

- **Word Clouds**: Identify dominant themes in documents
- **Frequency Charts**: Understand term distribution
- **LDA Topics**: Discover latent topics in document collection
- **Model Comparison**: Compare retrieval effectiveness

---

## 10. User Interface Design

### 10.1 Layout Structure
```
┌─────────────────────────────────────────────────────────┐
│                    Menu Bar                              │
├──────────┬──────────────────────────┬───────────────────┤
│          │                          │                   │
│  Left    │      Center              │    Right          │
│  Panel   │      Panel               │    Panel         │
│          │                          │                   │
│  • Search│  • Results (Tabs)        │  • Preprocessing  │
│  • Dataset│  • VSM Results          │  • Visualizations│
│  Info    │  • BM25 Results          │  • Evaluation     │
│          │  • Boolean Results       │                   │
│          │                          │                   │
└──────────┴──────────────────────────┴───────────────────┘
│                    Status Bar                            │
└─────────────────────────────────────────────────────────┘
```

### 10.2 Key UI Components

1. **Search Panel**:
   - Query input field
   - Model selection (VSM, BM25, Boolean, All)
   - Result limit option
   - Search button

2. **Results Panel**:
   - Tabbed interface for each model
   - Document list with scores
   - Document preview
   - Scrollable results

3. **Visualization Panel**:
   - Word Cloud tab
   - Frequency tab
   - Similarity Scores tab
   - Model Comparison tab
   - LDA Topics tab

4. **Evaluation Panel**:
   - Metrics table
   - Precision-Recall curves
   - Model comparison charts

5. **Preprocessing Panel**:
   - Document-by-document preprocessing steps
   - Original text snippets
   - Transformation at each step
   - Vectorization results (BoW, TF-IDF)
   - LDA topic assignments

---

## 11. Technical Challenges and Solutions

### 11.1 Challenge: Real-time Wikipedia API Loading
**Problem**: API rate limiting and network issues
**Solution**: 
- Implemented progress callbacks
- Added fallback to HuggingFace datasets
- Rate limiting with delays between requests
- Error handling and retry logic

### 11.2 Challenge: GUI Responsiveness
**Problem**: Long-running operations freeze UI
**Solution**:
- Implemented threading for background tasks
- Progress windows with status updates
- Thread-safe UI updates using `root.after()`

### 11.3 Challenge: Large Document Processing
**Problem**: Memory and performance issues with large datasets
**Solution**:
- Sparse matrix representations
- Configurable document limits
- Efficient vectorization
- Lazy loading where possible

### 11.4 Challenge: Model Comparison Visualization
**Problem**: Different result lengths and score ranges
**Solution**:
- Normalized score handling
- Flexible plotting with padding
- Multiple visualization types

---

## 12. Future Enhancements

### 12.1 Potential Improvements
1. **Query Expansion**: Add synonym and related term expansion
2. **Relevance Feedback**: Implement user feedback for result improvement
3. **Advanced Preprocessing**: Named entity recognition, phrase detection
4. **More Retrieval Models**: Add Language Models, Neural IR models
5. **Distributed Processing**: Support for larger document collections
6. **Caching**: Cache results and models for faster subsequent queries
7. **Export Formats**: Support for JSON, CSV, PDF exports
8. **Multi-language Support**: Extend to other languages
9. **Advanced Visualizations**: Interactive plots, 3D visualizations
10. **Performance Optimization**: Further speed improvements

### 12.2 Scalability Considerations
- Current system handles 20-100 documents efficiently
- Can be extended to 1000+ documents with optimizations
- Vectorization uses sparse matrices for memory efficiency
- LDA training can be optimized for larger corpora

---

## 13. Group Members

- **1. Mickel Wassef** (22010449)
- **2. Amr Khaled** (2206159)
- **3. Abdelrahman Jayasundara** (2206147)
- **4. Loay Salah** (2206155)

---

## 14. Conclusion

This Information Retrieval and Text Analytics system successfully implements a comprehensive solution for document retrieval with multiple models, evaluation metrics, and visualization capabilities. The system demonstrates:

1. **Complete IR Pipeline**: From data loading to result presentation
2. **Multiple Retrieval Models**: VSM, Boolean, and BM25 implementations
3. **Comprehensive Evaluation**: Standard IR metrics with visualization
4. **User-Friendly Interface**: Modern GUI with all features accessible
5. **Advanced Features**: LDA topic modeling, preprocessing visualization
6. **Real-World Application**: Real-time Wikipedia data integration

The project successfully combines theoretical IR concepts with practical implementation, providing a valuable tool for understanding and comparing different retrieval approaches. The modular architecture allows for easy extension and enhancement.

---

## 15. References

### Academic References
- Salton, G., & McGill, M. J. (1986). *Introduction to Modern Information Retrieval*. McGraw-Hill.
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
- Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.

### Technical References
- **Scikit-learn**: Machine learning library for vectorization and utilities
  - Documentation: https://scikit-learn.org/
- **NLTK**: Natural Language Toolkit for text preprocessing
  - Documentation: https://www.nltk.org/
- **Gensim**: Topic modeling and document similarity
  - Documentation: https://radimrehurek.com/gensim/
- **Matplotlib/Seaborn**: Data visualization
  - Documentation: https://matplotlib.org/, https://seaborn.pydata.org/
- **Wordcloud**: Word cloud generation
  - Documentation: https://github.com/amueller/word_cloud
- **HuggingFace Datasets**: Dataset loading and management
  - Documentation: https://huggingface.co/docs/datasets/
- **Wikipedia API**: Python Wikipedia library
  - Documentation: https://github.com/goldsmith/Wikipedia

---

## 16. Appendix

### 16.1 Installation Instructions

1. **Prerequisites**:
   ```bash
   Python 3.8 or higher
   pip package manager
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK Data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

4. **Run the Application**:
   ```bash
   python gui_main.py
   ```

### 16.2 Project Structure
```
ir project/
├── data_loader.py          # Data loading and management
├── preprocessing.py        # Text preprocessing pipeline
├── vectorization.py        # BoW and TF-IDF vectorization
├── retrieval_models.py     # VSM, Boolean, BM25 models
├── evaluation.py           # Evaluation metrics
├── visualization.py        # Visualization functions
├── gui_main.py             # Main GUI application
├── gui_components.py       # GUI components
├── gui_utils.py            # GUI utilities
├── main.py                 # Command-line interface
├── utils.py                # Helper functions
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── PROJECT_REPORT.md       # This report
└── notebooks/
    └── visualization.ipynb # Interactive notebook
```

### 16.3 Usage Examples

#### Example 1: Basic Search
1. Launch GUI: `python gui_main.py`
2. Wait for documents to load
3. Enter query: "machine learning"
4. Select model: "VSM"
5. Click "Search"
6. View results in Results Panel

#### Example 2: Model Comparison
1. Enter query: "cybersecurity threats"
2. Select model: "All"
3. Click "Search"
4. Compare results across all three models
5. View Model Comparison visualization

#### Example 3: Evaluation
1. Perform searches with different queries
2. Go to Evaluation tab
3. Click "Run Evaluation"
4. Review Precision, Recall, MAP metrics
5. Analyze Precision-Recall curves

#### Example 4: Topic Modeling
1. Go to Visualizations tab
2. Click on "LDA Topics" tab
3. View discovered topics
4. Analyze topic distribution across documents

---

**Report Version**: 1.0  
**Last Updated**: 2024  
**Project Status**: Complete

---

*This report provides a comprehensive overview of the Information Retrieval and Text Analytics system. For technical details, please refer to the source code and inline documentation.*

