# Information Retrieval and Text Analytics Project

## Project Overview

This project implements a comprehensive Information Retrieval (IR) system with text preprocessing, multiple retrieval models, evaluation metrics, and visualization capabilities. The system retrieves relevant documents from a Wikipedia dataset based on user queries.

## Objectives

- Implement text preprocessing pipeline (tokenization, lowercasing, stopwords removal, stemming/lemmatization)
- Build multiple retrieval models: Vector Space Model (VSM), Boolean Retrieval, and BM25
- Evaluate model performance using Precision, Recall, and Mean Average Precision (MAP)
- Visualize results with word clouds, frequency distributions, similarity scores, and topic modeling

## Installation

1. Clone or download this repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Project Structure

### Core Modules
- `main.py` - Main entry point to run the IR system (command line)
- `gui_main.py` - GUI application entry point
- `run_gui.py` - Simple GUI launcher script
- `data_loader.py` - Dataset loading and inspection utilities
- `preprocessing.py` - Text preprocessing pipeline
- `vectorization.py` - BoW and TF-IDF vectorization
- `retrieval_models.py` - VSM, Boolean, and BM25 retrieval models
- `evaluation.py` - Evaluation metrics (Precision, Recall, MAP)
- `visualization.py` - Visualization functions
- `utils.py` - Helper functions and utilities

### GUI Modules
- `gui_main.py` - Main GUI application window
- `gui_components.py` - Reusable GUI components (panels, widgets)
- `gui_utils.py` - GUI utility functions (threading, plotting helpers)

### Other Files
- `test_system.py` - Quick test script
- `notebooks/visualization.ipynb` - Interactive visualization notebook

## Usage

### GUI Application (Recommended)
Launch the graphical user interface:
```bash
python gui_main.py
```
or
```bash
python run_gui.py
```

The GUI provides:
- **Search Interface**: Query input with model selection (VSM, BM25, Boolean)
- **Results Display**: Tabbed view showing results from all models
- **Visualizations**: Word clouds, frequency charts, similarity scores, model comparison
- **Evaluation Metrics**: Precision, Recall, MAP with comparison tables
- **Topic Modeling**: LDA topic visualization
- **Dataset Management**: Load/reload dataset with statistics display

**Keyboard Shortcuts:**
- `Ctrl+O`: Load dataset
- `Ctrl+S`: Export results
- `Ctrl+E`: Run evaluation
- `Ctrl+W`: Generate word cloud
- `Ctrl+T`: Topic modeling
- `Ctrl+Q`: Exit
- `Enter`: Perform search

### Quick Test
Run the quick test script to verify everything works:
```bash
python test_system.py
```

### Main Application (Command Line)
Run the main application:
```bash
python main.py
```

Options:
- `--max-docs N`: Load N documents (default: 1000)
- `--mode {interactive|eval|visualize|all}`: Run mode (default: all)
- `--use-bow`: Use Bag-of-Words instead of TF-IDF

Examples:
```bash
# Interactive search only
python main.py --mode interactive --max-docs 500

# Evaluation only
python main.py --mode eval --max-docs 1000

# All features
python main.py --mode all --max-docs 500
```

### Interactive Notebook
Use the interactive notebook for visualization:
```bash
jupyter notebook notebooks/visualization.ipynb
```

## Dataset

This project uses the Wikipedia dataset from HuggingFace's `datasets` library. The dataset is automatically downloaded on first run.

## Features

### Retrieval Models
- **Vector Space Model (VSM)**: Cosine similarity-based document ranking
- **Boolean Retrieval**: Logical query processing with AND, OR, NOT operators
- **BM25**: Probabilistic ranking algorithm

### Evaluation Metrics
- Precision@k
- Recall@k
- Mean Average Precision (MAP)

### Visualizations
- Word clouds for top keywords
- Frequency distribution plots
- Document-query similarity score charts
- LDA topic modeling visualization

## References

- **Scikit-learn**: Machine learning library for vectorization and utilities
- **NLTK**: Natural Language Toolkit for text preprocessing
- **Gensim**: Topic modeling and document similarity
- **Matplotlib/Seaborn**: Data visualization
- **Wordcloud**: Word cloud generation
- **HuggingFace Datasets**: Dataset loading and management

## License

This project is for educational purposes.