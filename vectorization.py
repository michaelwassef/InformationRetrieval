"""
Vectorization module for converting text to numerical representations.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import List, Union
import numpy as np
from scipy.sparse import csr_matrix


class Vectorizer:
    """Handles text vectorization using BoW and TF-IDF."""
    
    def __init__(self, method: str = 'tfidf', max_features: int = None, 
                 min_df: int = 1, max_df: float = 1.0):
        """
        Initialize the vectorizer.
        
        Args:
            method: 'bow' for Bag-of-Words or 'tfidf' for TF-IDF
            max_features: Maximum number of features (vocabulary size)
            min_df: Minimum document frequency for a term to be included
            max_df: Maximum document frequency for a term to be included
        """
        self.method = method.lower()
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.vocabulary_ = None
        self._initialize_vectorizer()
    
    def _initialize_vectorizer(self) -> None:
        """Initialize the appropriate vectorizer based on method."""
        if self.method == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                lowercase=False  # Assume preprocessing already done
            )
        elif self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                lowercase=False  # Assume preprocessing already done
            )
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'bow' or 'tfidf'")
    
    def fit(self, documents: List[str]) -> None:
        """
        Fit the vectorizer on the documents.
        
        Args:
            documents: List of preprocessed document strings
        """
        self.vectorizer.fit(documents)
        self.vocabulary_ = self.vectorizer.vocabulary_
    
    def transform(self, documents: List[str]) -> csr_matrix:
        """
        Transform documents into vector representation.
        
        Args:
            documents: List of preprocessed document strings
            
        Returns:
            Sparse matrix of document vectors
        """
        return self.vectorizer.transform(documents)
    
    def fit_transform(self, documents: List[str]) -> csr_matrix:
        """
        Fit the vectorizer and transform documents.
        
        Args:
            documents: List of preprocessed document strings
            
        Returns:
            Sparse matrix of document vectors
        """
        vectors = self.vectorizer.fit_transform(documents)
        self.vocabulary_ = self.vectorizer.vocabulary_
        return vectors
    
    def transform_query(self, query: str) -> csr_matrix:
        """
        Transform a single query into vector representation.
        
        Args:
            query: Preprocessed query string
            
        Returns:
            Sparse matrix of query vector
        """
        return self.vectorizer.transform([query])
    
    def get_feature_names(self) -> List[str]:
        """Get the feature names (vocabulary terms)."""
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_vocabulary_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.vocabulary_) if self.vocabulary_ else 0


class DocumentVectorizer:
    """Wrapper class for managing document vectorization."""
    
    def __init__(self, preprocessor, method: str = 'tfidf', 
                 max_features: int = None, min_df: int = 1, max_df: float = 1.0):
        """
        Initialize the document vectorizer.
        
        Args:
            preprocessor: TextPreprocessor instance
            method: 'bow' or 'tfidf'
            max_features: Maximum number of features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        self.preprocessor = preprocessor
        self.vectorizer = Vectorizer(method=method, max_features=max_features,
                                    min_df=min_df, max_df=max_df)
        self.document_vectors = None
        self.preprocessed_documents = None
    
    def fit_transform(self, documents: List[str]) -> csr_matrix:
        """
        Preprocess and vectorize documents.
        
        Args:
            documents: List of raw document strings
            
        Returns:
            Sparse matrix of document vectors
        """
        # Preprocess documents
        self.preprocessed_documents = self.preprocessor.preprocess_documents(documents)
        
        # Fit and transform
        self.document_vectors = self.vectorizer.fit_transform(self.preprocessed_documents)
        
        return self.document_vectors
    
    def transform_query(self, query: str) -> csr_matrix:
        """
        Preprocess and vectorize a query.
        
        Args:
            query: Raw query string
            
        Returns:
            Sparse matrix of query vector
        """
        preprocessed_query = self.preprocessor.preprocess(query)
        # Check if query became empty after preprocessing
        if not preprocessed_query or not preprocessed_query.strip():
            # Return zero vector with same shape as document vectors
            if self.document_vectors is not None:
                return csr_matrix((1, self.document_vectors.shape[1]))
            else:
                return csr_matrix((1, 0))
        return self.vectorizer.transform_query(preprocessed_query)
    
    def get_document_vectors(self) -> csr_matrix:
        """Get the document vectors."""
        return self.document_vectors
    
    def get_preprocessed_documents(self) -> List[str]:
        """Get the preprocessed documents."""
        return self.preprocessed_documents
    
    def get_vocabulary(self) -> List[str]:
        """Get the vocabulary."""
        return self.vectorizer.get_feature_names()
    
    def get_vocabulary_size(self) -> int:
        """Get the vocabulary size."""
        return self.vectorizer.get_vocabulary_size()

