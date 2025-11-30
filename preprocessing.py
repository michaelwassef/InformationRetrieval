"""
Text preprocessing pipeline for the Information Retrieval system.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import List, Callable
import string

# Download required NLTK data (will be done on first import if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


class TextPreprocessor:
    """Text preprocessing pipeline for IR system."""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_stopwords: bool = True,
                 remove_punctuation: bool = True,
                 stem: bool = False,
                 lemmatize: bool = True):
        """
        Initialize the text preprocessor.
        
        Args:
            lowercase: Whether to convert text to lowercase
            remove_stopwords: Whether to remove stopwords
            remove_punctuation: Whether to remove punctuation
            stem: Whether to apply stemming (mutually exclusive with lemmatize)
            lemmatize: Whether to apply lemmatization (mutually exclusive with stem)
        """
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.stem = stem
        self.lemmatize = lemmatize
        
        # Initialize stopwords set
        if self.remove_stopwords:
            try:
                self.stopwords_set = set(stopwords.words('english'))
            except LookupError:
                nltk.download('stopwords', quiet=True)
                self.stopwords_set = set(stopwords.words('english'))
        else:
            self.stopwords_set = set()
        
        # Initialize stemmer and lemmatizer
        if self.stem:
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = None
            
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None
        
        # Punctuation set
        self.punctuation_set = set(string.punctuation)
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess a single text document.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string
        """
        # Lowercasing
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation (before tokenization to handle punctuation attached to words)
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords_set]
        
        # Remove empty tokens and tokens that are only punctuation
        tokens = [token for token in tokens if token.strip() and token.isalnum()]
        
        # Stemming or Lemmatization
        if self.stem and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into a string
        return ' '.join(tokens)
    
    def preprocess_tokens(self, text: str) -> List[str]:
        """
        Preprocess text and return as a list of tokens.
        
        Args:
            text: Input text string
            
        Returns:
            List of preprocessed tokens
        """
        # Lowercasing
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords_set]
        
        # Remove empty tokens and tokens that are only punctuation
        tokens = [token for token in tokens if token.strip() and token.isalnum()]
        
        # Stemming or Lemmatization
        if self.stem and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def preprocess_documents(self, documents: List[str]) -> List[str]:
        """
        Preprocess a list of documents.
        
        Args:
            documents: List of document text strings
            
        Returns:
            List of preprocessed document strings
        """
        return [self.preprocess(doc) for doc in documents]
    
    def preprocess_documents_to_tokens(self, documents: List[str]) -> List[List[str]]:
        """
        Preprocess a list of documents and return as token lists.
        
        Args:
            documents: List of document text strings
            
        Returns:
            List of token lists
        """
        return [self.preprocess_tokens(doc) for doc in documents]


def create_preprocessor(config: dict = None) -> TextPreprocessor:
    """
    Create a preprocessor with custom configuration.
    
    Args:
        config: Dictionary with preprocessor configuration options
        
    Returns:
        Configured TextPreprocessor instance
    """
    if config is None:
        config = {}
    
    return TextPreprocessor(
        lowercase=config.get('lowercase', True),
        remove_stopwords=config.get('remove_stopwords', True),
        remove_punctuation=config.get('remove_punctuation', True),
        stem=config.get('stem', False),
        lemmatize=config.get('lemmatize', True)
    )

