"""
Data loading and inspection utilities for the Information Retrieval system.
"""

from datasets import load_dataset
import pandas as pd
from typing import List, Dict, Tuple, Set, Optional, Callable
import numpy as np
import wikipedia
import time


class DataLoader:
    """Handles loading and inspection of Wikipedia dataset."""
    
    def __init__(self, dataset_name: str = "wikipedia", 
                 language: str = "20220301.en",
                 max_documents: int = 100):
        """
        Initialize the data loader.
        
        Args:
            dataset_name: Name of the dataset to load
            language: Language/version of Wikipedia to load
            max_documents: Maximum number of documents to load (for faster processing)
        """
        self.dataset_name = dataset_name
        self.language = language
        self.max_documents = max_documents
        self.dataset = None
        self.documents = []
        self.document_ids = []
        
    def load_dataset(self, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> None:
        """
        Load Wikipedia articles using Wikipedia API, filtered to tech topics.
        
        Args:
            progress_callback: Optional callback function(current, total, message) for progress updates
        """
        print("Loading Wikipedia articles for tech topics (cybersecurity, AI, data science)...")
        if progress_callback:
            progress_callback(0, self.max_documents, "Starting to load Wikipedia articles...")
        
        # Try Wikipedia API first
        try:
            self._load_from_wikipedia_api(progress_callback)
            if len(self.documents) > 0:
                print(f"Loaded {len(self.documents)} Wikipedia articles successfully.")
                if progress_callback:
                    progress_callback(len(self.documents), self.max_documents, 
                                    f"Successfully loaded {len(self.documents)} articles!")
                return
        except Exception as e:
            print(f"Error loading from Wikipedia API: {e}")
            print("Falling back to HuggingFace datasets...")
        
        # Fallback to HuggingFace
        try:
            # Try loading from a pre-existing dataset on HuggingFace Hub
            # Using a smaller, more accessible dataset
            try:
                # Try loading a Wikipedia dump from HuggingFace
                dataset = load_dataset("wikipedia", "20220301.en", 
                                     split=f"train[:{self.max_documents}]",
                                     trust_remote_code=True)
            except:
                # Fallback: Use a different approach - load from a text dataset
                try:
                    # Try using a simpler text dataset
                    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", 
                                         split=f"train[:{self.max_documents}]")
                    # Wikitext has different structure, need to adapt
                    self.documents = []
                    self.document_ids = []
                    
                    current_doc = {'id': 0, 'title': 'Document 0', 'text': ''}
                    doc_id = 0
                    
                    for idx, item in enumerate(dataset):
                        text = item.get('text', '')
                        if text and len(text.strip()) > 10:  # Skip very short lines
                            if text.startswith(' = ') and ' = ' in text:
                                # This might be a title
                                if current_doc['text']:
                                    self.documents.append(current_doc)
                                    self.document_ids.append(doc_id)
                                    doc_id += 1
                                current_doc = {'id': doc_id, 'title': text.strip(' ='), 'text': ''}
                            else:
                                current_doc['text'] += ' ' + text
                    
                    # Add last document
                    if current_doc['text']:
                        self.documents.append(current_doc)
                        self.document_ids.append(doc_id)
                    
                    print(f"Loaded {len(self.documents)} documents successfully from wikitext.")
                    return
                except Exception as e3:
                    # Final fallback: Create sample documents
                    print(f"Could not load from HuggingFace: {e3}")
                    print("Creating tech-focused sample documents (cybersecurity, AI, data science)...")
                    self._create_sample_documents()
                    return
            
            # Process standard Wikipedia dataset
            self.dataset = dataset
            self.documents = []
            self.document_ids = []
            
            for idx, item in enumerate(dataset):
                # Wikipedia dataset structure: 'text' field contains the article text
                # 'title' field contains the article title
                text = item.get('text', '')
                title = item.get('title', f'Document {idx}')
                
                if text and len(text.strip()) > 0:
                    self.documents.append({
                        'id': idx,
                        'title': title,
                        'text': text
                    })
                    self.document_ids.append(idx)
            
            print(f"Loaded {len(self.documents)} documents successfully.")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using tech-focused sample documents (cybersecurity, AI, data science)...")
            self._create_sample_documents()
    
    def _load_from_wikipedia_api(self, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> None:
        """
        Load Wikipedia articles using Wikipedia API, filtered to tech topics.
        
        Args:
            progress_callback: Optional callback function(current, total, message) for progress updates
        """
        wikipedia.set_lang("en")
        
        # Define search terms for each tech category
        search_terms = {
            'cybersecurity': [
                'cybersecurity', 'information security', 'network security', 
                'encryption', 'penetration testing', 'threat intelligence',
                'security operations center', 'zero trust security', 'firewall',
                'intrusion detection system', 'vulnerability assessment',
                'security audit', 'cryptography', 'secure coding'
            ],
            'ai': [
                'artificial intelligence', 'machine learning', 'deep learning',
                'neural networks', 'natural language processing', 
                'reinforcement learning', 'supervised learning', 
                'unsupervised learning', 'computer vision', 'speech recognition',
                'expert systems', 'genetic algorithms', 'fuzzy logic'
            ],
            'data_science': [
                'data science', 'big data', 'data analytics', 'data mining',
                'data visualization', 'predictive analytics', 'statistical analysis',
                'business intelligence', 'data warehouse', 'data lake',
                'machine learning in data science', 'data preprocessing'
            ]
        }
        
        # Tech-related keywords for filtering
        tech_keywords = {
            'cybersecurity': ['security', 'cyber', 'threat', 'attack', 'vulnerability', 
                            'encryption', 'firewall', 'malware', 'phishing', 'breach'],
            'ai': ['artificial intelligence', 'machine learning', 'neural', 'algorithm',
                  'intelligence', 'learning', 'model', 'training', 'prediction'],
            'data_science': ['data', 'analytics', 'analysis', 'statistics', 'mining',
                           'visualization', 'dataset', 'processing', 'insight']
        }
        
        self.documents = []
        self.document_ids = []
        seen_titles: Set[str] = set()
        
        doc_id = 0
        
        # Fetch articles for each category
        for category, terms in search_terms.items():
            category_msg = f"Fetching {category} articles..."
            print(category_msg)
            if progress_callback:
                progress_callback(len(self.documents), self.max_documents, category_msg)
            
            for search_term in terms:
                if len(self.documents) >= self.max_documents:
                    break
                
                try:
                    # Search for articles
                    search_results = wikipedia.search(search_term, results=10)
                    
                    for title in search_results:
                        if len(self.documents) >= self.max_documents:
                            break
                        
                        # Skip if already seen
                        if title.lower() in seen_titles:
                            continue
                        
                        # Skip disambiguation pages
                        if 'disambiguation' in title.lower():
                            continue
                        
                        try:
                            # Fetch article
                            page = wikipedia.page(title, auto_suggest=False)
                            
                            # Check if article is relevant
                            title_lower = title.lower()
                            content_lower = page.content.lower()
                            
                            # Check if title or content contains tech keywords
                            is_relevant = False
                            category_keywords = tech_keywords[category]
                            
                            # Check title
                            for keyword in category_keywords:
                                if keyword in title_lower:
                                    is_relevant = True
                                    break
                            
                            # Check content summary (first 500 chars)
                            if not is_relevant:
                                summary = content_lower[:500]
                                for keyword in category_keywords:
                                    if keyword in summary:
                                        is_relevant = True
                                        break
                            
                            if is_relevant and len(page.content) > 200:  # Minimum length
                                # Limit content length
                                content = page.content[:10000]  # First 10000 characters
                                
                                self.documents.append({
                                    'id': doc_id,
                                    'title': page.title,
                                    'text': content
                                })
                                self.document_ids.append(doc_id)
                                seen_titles.add(title.lower())
                                doc_id += 1
                                
                                # Update progress
                                if progress_callback:
                                    progress_callback(len(self.documents), self.max_documents, 
                                                    f"Loaded: {page.title}")
                                
                                # Small delay to avoid rate limiting
                                time.sleep(0.1)
                        
                        except wikipedia.exceptions.DisambiguationError:
                            # Skip disambiguation pages
                            continue
                        except wikipedia.exceptions.PageError:
                            # Page not found, skip
                            continue
                        except Exception as e:
                            # Other errors, skip
                            print(f"Error fetching page '{title}': {e}")
                            continue
                
                except Exception as e:
                    print(f"Error searching for '{search_term}': {e}")
                    continue
                
                # Small delay between searches
                time.sleep(0.2)
            
            if len(self.documents) >= self.max_documents:
                break
        
        if len(self.documents) == 0:
            raise Exception("No Wikipedia articles could be loaded")
    
    def _create_sample_documents(self) -> None:
        """Create sample documents focused on tech topics (cybersecurity, AI, data science)."""
        sample_docs = [
            {
                'id': 0,
                'title': 'Artificial Intelligence',
                'text': 'Artificial intelligence is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. Machine learning algorithms enable AI systems to learn from data and improve their performance over time. Deep learning and neural networks are key technologies driving modern AI applications.'
            },
            {
                'id': 1,
                'title': 'Machine Learning',
                'text': 'Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. Deep learning uses neural networks with multiple layers to process complex data patterns.'
            },
            {
                'id': 2,
                'title': 'Cybersecurity Fundamentals',
                'text': 'Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks. These cyberattacks are usually aimed at accessing, changing, or destroying sensitive information, extorting money from users, or interrupting normal business processes. Common threats include malware, phishing, ransomware, and social engineering attacks.'
            },
            {
                'id': 3,
                'title': 'Network Security',
                'text': 'Network security consists of policies and practices adopted to prevent and monitor unauthorized access, misuse, modification, or denial of a computer network and network-accessible resources. Network security involves the authorization of access to data in a network, which is controlled by the network administrator.'
            },
            {
                'id': 4,
                'title': 'Data Science',
                'text': 'Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data. Data scientists combine domain expertise, programming skills, and knowledge of mathematics and statistics to extract meaningful insights from data.'
            },
            {
                'id': 5,
                'title': 'Big Data Analytics',
                'text': 'Big data analytics is the process of examining large and varied data sets to uncover hidden patterns, unknown correlations, market trends, customer preferences and other useful information. Big data analytics helps organizations harness their data and use it to identify new opportunities and make better business decisions.'
            },
            {
                'id': 6,
                'title': 'Deep Learning',
                'text': 'Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, and recurrent neural networks have been applied to fields including computer vision, speech recognition, and natural language processing.'
            },
            {
                'id': 7,
                'title': 'Cybersecurity Threats',
                'text': 'Cybersecurity threats are malicious acts that seek to damage data, steal data, or disrupt digital life in general. Threats include computer viruses, data breaches, Denial of Service attacks, and other attack vectors. Organizations must implement security measures including firewalls, encryption, and access controls to protect against these threats.'
            },
            {
                'id': 8,
                'title': 'Neural Networks',
                'text': 'A neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of artificial neurons or nodes. The connections of the biological neuron are modeled as weights. Neural networks are used in machine learning and artificial intelligence to solve complex problems in pattern recognition, classification, and prediction.'
            },
            {
                'id': 9,
                'title': 'Python for Data Science',
                'text': 'Python is a high-level programming language widely used in data science and machine learning. Python libraries like NumPy, Pandas, Scikit-learn, and TensorFlow provide powerful tools for data analysis, machine learning, and artificial intelligence. Python is dynamically typed and garbage-collected, making it ideal for rapid development and prototyping.'
            },
            {
                'id': 10,
                'title': 'Encryption and Cryptography',
                'text': 'Encryption is the process of converting information or data into a code, especially to prevent unauthorized access. Cryptography is the practice and study of techniques for secure communication in the presence of adversarial behavior. Modern cryptography uses algorithms and mathematical techniques to secure data transmission and storage.'
            },
            {
                'id': 11,
                'title': 'Cloud Security',
                'text': 'Cloud security refers to a broad set of policies, technologies, and controls deployed to protect data, applications, and the associated infrastructure of cloud computing. Cloud security includes identity and access management, data encryption, network security, and compliance with regulatory requirements.'
            },
            {
                'id': 12,
                'title': 'Data Mining',
                'text': 'Data mining is the process of discovering patterns in large data sets involving methods at the intersection of machine learning, statistics, and database systems. Data mining techniques include clustering, classification, regression, and association rule learning to extract valuable insights from data.'
            },
            {
                'id': 13,
                'title': 'Information Security',
                'text': 'Information security, sometimes shortened to InfoSec, is the practice of protecting information by mitigating information risks. It is part of information risk management and involves protecting information systems from unauthorized access, use, disclosure, disruption, modification, or destruction.'
            },
            {
                'id': 14,
                'title': 'Reinforcement Learning',
                'text': 'Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning.'
            },
            {
                'id': 15,
                'title': 'Penetration Testing',
                'text': 'Penetration testing, also called pen testing or ethical hacking, is the practice of testing a computer system, network or web application to find security vulnerabilities that an attacker could exploit. Penetration testing can be automated with software applications or performed manually.'
            },
            {
                'id': 16,
                'title': 'Natural Language Processing',
                'text': 'Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. NLP techniques enable computers to understand, interpret, and generate human language in a valuable way. Applications include chatbots, translation, and sentiment analysis.'
            },
            {
                'id': 17,
                'title': 'Data Visualization',
                'text': 'Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data. Effective visualization helps communicate data insights clearly and efficiently.'
            },
            {
                'id': 18,
                'title': 'Zero Trust Security',
                'text': 'Zero Trust is a security model based on the principle of maintaining strict access controls and not trusting anyone by default, even those already inside the network perimeter. Zero Trust requires verification of every user and device attempting to access resources, regardless of their location.'
            },
            {
                'id': 19,
                'title': 'Supervised Learning',
                'text': 'Supervised learning is a type of machine learning where algorithms learn from labeled training data to make predictions or decisions. The algorithm learns a mapping from inputs to outputs based on example input-output pairs. Common supervised learning tasks include classification and regression.'
            }
        ]
        
        # Repeat to reach max_documents if needed
        while len(sample_docs) < self.max_documents:
            for doc in sample_docs[:]:
                if len(sample_docs) >= self.max_documents:
                    break
                new_doc = doc.copy()
                new_doc['id'] = len(sample_docs)
                sample_docs.append(new_doc)
        
        self.documents = sample_docs[:self.max_documents]
        self.document_ids = [doc['id'] for doc in self.documents]
        print(f"Created {len(self.documents)} tech-focused documents (cybersecurity, AI, data science).")
    
    def get_documents(self) -> List[Dict]:
        """Get the list of loaded documents."""
        return self.documents
    
    def get_document_by_id(self, doc_id: int) -> Dict:
        """Get a specific document by its ID."""
        for doc in self.documents:
            if doc['id'] == doc_id:
                return doc
        return None
    
    def get_document_texts(self) -> List[str]:
        """Get a list of all document texts."""
        return [doc['text'] for doc in self.documents]
    
    def get_document_titles(self) -> List[str]:
        """Get a list of all document titles."""
        return [doc['title'] for doc in self.documents]
    
    def inspect_dataset(self) -> Dict:
        """
        Inspect the dataset and return statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if not self.documents:
            return {"error": "No documents loaded"}
        
        stats = {
            "total_documents": len(self.documents),
            "total_characters": sum(len(doc['text']) for doc in self.documents),
            "total_words": sum(len(doc['text'].split()) for doc in self.documents),
            "avg_doc_length_chars": np.mean([len(doc['text']) for doc in self.documents]),
            "avg_doc_length_words": np.mean([len(doc['text'].split()) for doc in self.documents]),
            "min_doc_length_words": np.min([len(doc['text'].split()) for doc in self.documents]),
            "max_doc_length_words": np.max([len(doc['text'].split()) for doc in self.documents])
        }
        
        return stats
    
    def print_dataset_info(self) -> None:
        """Print dataset information and statistics."""
        print("\n" + "="*80)
        print("DATASET INFORMATION")
        print("="*80)
        
        if not self.documents:
            print("No documents loaded.")
            return
        
        stats = self.inspect_dataset()
        
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Characters: {stats['total_characters']:,}")
        print(f"Total Words: {stats['total_words']:,}")
        print(f"Average Document Length (chars): {stats['avg_doc_length_chars']:.2f}")
        print(f"Average Document Length (words): {stats['avg_doc_length_words']:.2f}")
        print(f"Min Document Length (words): {stats['min_doc_length_words']}")
        print(f"Max Document Length (words): {stats['max_doc_length_words']}")
        
        print("\n" + "-"*80)
        print("SAMPLE DOCUMENTS")
        print("-"*80)
        
        # Show first 3 documents
        for i, doc in enumerate(self.documents[:3]):
            print(f"\nDocument {i+1}:")
            print(f"Title: {doc['title']}")
            print(f"ID: {doc['id']}")
            preview = doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text']
            print(f"Preview: {preview}")
        
        print("\n" + "="*80)

