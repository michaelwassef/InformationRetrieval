"""
Retrieval models for Information Retrieval system.
Implements Vector Space Model, Boolean Retrieval, and BM25.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Set, Union
import re
from collections import defaultdict, Counter
import math


class VectorSpaceModel:
    """Vector Space Model using cosine similarity."""
    
    def __init__(self, document_vectors: csr_matrix, document_ids: List[int] = None):
        """
        Initialize the Vector Space Model.
        
        Args:
            document_vectors: Sparse matrix of document vectors
            document_ids: List of document IDs (if None, uses indices)
        """
        self.document_vectors = document_vectors
        if document_ids is None:
            self.document_ids = list(range(document_vectors.shape[0]))
        else:
            self.document_ids = document_ids
    
    def search(self, query_vector: csr_matrix, top_k: int = None) -> List[Tuple[int, float]]:
        """
        Search for documents using cosine similarity.
        
        Args:
            query_vector: Query vector (sparse matrix)
            top_k: Number of top results to return (None for all results)
            
        Returns:
            List of (document_id, similarity_score) tuples, sorted by score
        """
        # Check if query vector is empty or all zeros
        if query_vector.shape[1] == 0 or query_vector.nnz == 0:
            return []
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Check if all similarities are zero or NaN
        if np.all(np.isnan(similarities)) or np.all(similarities == 0):
            return []
        
        # Replace NaN with 0
        similarities = np.nan_to_num(similarities, nan=0.0)
        
        # Get all results sorted by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in sorted_indices:
            doc_id = self.document_ids[idx]
            score = float(similarities[idx])
            results.append((doc_id, score))
        
        # Filter out zero scores only if we have non-zero results
        non_zero_results = [(doc_id, score) for doc_id, score in results if score > 0]
        if non_zero_results:
            if top_k is None:
                return non_zero_results
            return non_zero_results[:top_k]
        else:
            # If all scores are zero, return all or top_k
            if top_k is None:
                return results
            return results[:top_k]
    
    def get_similarity_scores(self, query_vector: csr_matrix) -> np.ndarray:
        """
        Get similarity scores for all documents.
        
        Args:
            query_vector: Query vector (sparse matrix)
            
        Returns:
            Array of similarity scores
        """
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        return similarities


class BooleanRetrievalModel:
    """Boolean Retrieval Model with inverted index."""
    
    def __init__(self, documents: List[str], document_ids: List[int] = None):
        """
        Initialize the Boolean Retrieval Model.
        
        Args:
            documents: List of preprocessed document strings
            document_ids: List of document IDs (if None, uses indices)
        """
        self.documents = documents
        if document_ids is None:
            self.document_ids = list(range(len(documents)))
        else:
            self.document_ids = document_ids
        
        # Build inverted index
        self.inverted_index = self._build_inverted_index()
    
    def _build_inverted_index(self) -> Dict[str, Set[int]]:
        """Build inverted index from documents."""
        inverted_index = defaultdict(set)
        
        for doc_idx, doc in enumerate(self.documents):
            # Tokenize document
            tokens = doc.lower().split()
            # Add document ID to posting list for each term
            for token in tokens:
                inverted_index[token].add(self.document_ids[doc_idx])
        
        return dict(inverted_index)
    
    def _parse_boolean_query(self, query: str) -> List[Union[str, List]]:
        """
        Parse a boolean query into tokens and operators.
        Simple parser that handles AND, OR, NOT operators.
        
        Args:
            query: Boolean query string (e.g., "python AND machine OR learning")
            
        Returns:
            Parsed query structure
        """
        # Tokenize query
        tokens = query.upper().split()
        parsed = []
        i = 0
        
        while i < len(tokens):
            if tokens[i] in ['AND', 'OR', 'NOT']:
                parsed.append(tokens[i])
                i += 1
            else:
                # It's a term
                parsed.append(tokens[i].lower())
                i += 1
        
        return parsed
    
    def _evaluate_boolean_query(self, parsed_query: List[Union[str, List]]) -> Set[int]:
        """
        Evaluate a parsed boolean query.
        
        Args:
            parsed_query: Parsed query structure
            
        Returns:
            Set of document IDs matching the query
        """
        if not parsed_query:
            return set()
        
        # Start with first term
        result = self.inverted_index.get(parsed_query[0], set())
        i = 1
        
        while i < len(parsed_query):
            if parsed_query[i] == 'AND':
                if i + 1 < len(parsed_query):
                    term = parsed_query[i + 1]
                    term_docs = self.inverted_index.get(term, set())
                    result = result.intersection(term_docs)
                    i += 2
            elif parsed_query[i] == 'OR':
                if i + 1 < len(parsed_query):
                    term = parsed_query[i + 1]
                    term_docs = self.inverted_index.get(term, set())
                    result = result.union(term_docs)
                    i += 2
            elif parsed_query[i] == 'NOT':
                if i + 1 < len(parsed_query):
                    term = parsed_query[i + 1]
                    term_docs = self.inverted_index.get(term, set())
                    result = result.difference(term_docs)
                    i += 2
            else:
                # Default to AND if no operator
                term = parsed_query[i]
                term_docs = self.inverted_index.get(term, set())
                result = result.intersection(term_docs)
                i += 1
        
        return result
    
    def search(self, query: str, top_k: int = None) -> List[int]:
        """
        Search for documents using boolean query.
        
        Args:
            query: Boolean query string
            top_k: Number of results to return (None returns all)
            
        Returns:
            List of document IDs matching the query
        """
        # Preprocess query (lowercase)
        query = query.lower()
        
        # Parse query
        parsed_query = self._parse_boolean_query(query)
        
        # Evaluate query
        result_docs = self._evaluate_boolean_query(parsed_query)
        
        # Convert to list and limit if needed
        result_list = list(result_docs)
        if top_k is not None:
            result_list = result_list[:top_k]
        
        return result_list


class BM25:
    """BM25 (Best Matching 25) ranking algorithm."""
    
    def __init__(self, documents: List[str], document_ids: List[int] = None,
                 k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 model.
        
        Args:
            documents: List of preprocessed document strings
            document_ids: List of document IDs (if None, uses indices)
            k1: BM25 parameter k1 (controls term frequency saturation)
            b: BM25 parameter b (controls length normalization)
        """
        self.documents = documents
        if document_ids is None:
            self.document_ids = list(range(len(documents)))
        else:
            self.document_ids = document_ids
        
        self.k1 = k1
        self.b = b
        
        # Preprocess documents into token lists
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Calculate document lengths
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_docs]
        self.avg_doc_length = np.mean(self.doc_lengths) if self.doc_lengths else 0
        
        # Build inverted index with term frequencies
        self.inverted_index = self._build_inverted_index()
        
        # Calculate IDF for all terms
        self.idf = self._calculate_idf()
    
    def _build_inverted_index(self) -> Dict[str, Dict[int, int]]:
        """
        Build inverted index with term frequencies.
        
        Returns:
            Dictionary mapping terms to {doc_id: term_frequency}
        """
        inverted_index = defaultdict(dict)
        
        for doc_idx, tokens in enumerate(self.tokenized_docs):
            doc_id = self.document_ids[doc_idx]
            # Count term frequencies in document
            term_freq = Counter(tokens)
            for term, freq in term_freq.items():
                inverted_index[term][doc_id] = freq
        
        return dict(inverted_index)
    
    def _calculate_idf(self) -> Dict[str, float]:
        """
        Calculate Inverse Document Frequency (IDF) for all terms.
        
        Returns:
            Dictionary mapping terms to IDF values
        """
        N = len(self.documents)
        idf = {}
        
        for term, doc_freqs in self.inverted_index.items():
            df = len(doc_freqs)  # Document frequency
            # IDF formula: log((N - df + 0.5) / (df + 0.5))
            idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        
        return idf
    
    def _calculate_bm25_score(self, query_terms: List[str], doc_id: int) -> float:
        """
        Calculate BM25 score for a query-document pair.
        
        Args:
            query_terms: List of query terms
            doc_id: Document ID
            
        Returns:
            BM25 score
        """
        score = 0.0
        doc_idx = self.document_ids.index(doc_id)
        doc_length = self.doc_lengths[doc_idx]
        
        # Count term frequencies in query
        query_term_freq = Counter(query_terms)
        
        for term in set(query_terms):  # Unique terms only
            if term not in self.inverted_index:
                continue
            
            # Term frequency in document
            tf = self.inverted_index[term].get(doc_id, 0)
            if tf == 0:
                continue
            
            # IDF
            idf = self.idf.get(term, 0)
            
            # Query term frequency
            qf = query_term_freq[term]
            
            # BM25 formula
            # score += IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length))) * qf
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            score += idf * (numerator / denominator) * qf
        
        return score
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        """
        Search for documents using BM25 ranking.
        
        Args:
            query: Query string (preprocessed)
            top_k: Number of top results to return (None for all results)
            
        Returns:
            List of (document_id, bm25_score) tuples, sorted by score
        """
        # Tokenize query
        query_terms = query.lower().split()
        
        # Calculate BM25 scores for all documents
        scores = {}
        for doc_id in self.document_ids:
            score = self._calculate_bm25_score(query_terms, doc_id)
            if score > 0:
                scores[doc_id] = score
        
        # Sort by score and return top k or all if top_k is None
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if top_k is None:
            return sorted_results
        return sorted_results[:top_k]
    
    def get_scores(self, query: str) -> Dict[int, float]:
        """
        Get BM25 scores for all documents.
        
        Args:
            query: Query string (preprocessed)
            
        Returns:
            Dictionary mapping document_id to BM25 score
        """
        query_terms = query.lower().split()
        scores = {}
        
        for doc_id in self.document_ids:
            score = self._calculate_bm25_score(query_terms, doc_id)
            scores[doc_id] = score
        
        return scores


class RetrievalSystem:
    """Unified retrieval system that supports multiple models."""
    
    def __init__(self, documents: List[str], document_ids: List[int] = None,
                 preprocessor=None, vectorizer=None):
        """
        Initialize the retrieval system.
        
        Args:
            documents: List of raw document strings
            document_ids: List of document IDs
            preprocessor: TextPreprocessor instance
            vectorizer: DocumentVectorizer instance
        """
        self.documents = documents
        self.document_ids = document_ids if document_ids else list(range(len(documents)))
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        
        # Initialize models
        self.vsm_model = None
        self.boolean_model = None
        self.bm25_model = None
        
        # Preprocessed documents
        self.preprocessed_documents = None
        
        # Initialize models if preprocessor is provided
        if preprocessor:
            self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize all retrieval models."""
        # Preprocess documents
        self.preprocessed_documents = self.preprocessor.preprocess_documents(self.documents)
        
        # Initialize Boolean model
        self.boolean_model = BooleanRetrievalModel(
            self.preprocessed_documents, 
            self.document_ids
        )
        
        # Initialize BM25 model
        self.bm25_model = BM25(
            self.preprocessed_documents,
            self.document_ids
        )
        
        # Initialize VSM if vectorizer is provided
        if self.vectorizer:
            document_vectors = self.vectorizer.get_document_vectors()
            self.vsm_model = VectorSpaceModel(document_vectors, self.document_ids)
    
    def search_vsm(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search using Vector Space Model."""
        if not self.vsm_model:
            raise ValueError("VSM model not initialized. Provide a vectorizer.")
        
        query_vector = self.vectorizer.transform_query(query)
        return self.vsm_model.search(query_vector, top_k)
    
    def search_boolean(self, query: str, top_k: int = None) -> List[int]:
        """Search using Boolean Retrieval Model."""
        if not self.boolean_model:
            raise ValueError("Boolean model not initialized. Provide a preprocessor.")
        
        return self.boolean_model.search(query, top_k)
    
    def search_bm25(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        """Search using BM25."""
        if not self.bm25_model:
            raise ValueError("BM25 model not initialized. Provide a preprocessor.")
        
        # Preprocess query
        preprocessed_query = self.preprocessor.preprocess(query)
        return self.bm25_model.search(preprocessed_query, top_k)

