"""
Utility functions for the Information Retrieval system.
"""

import os
import pickle
from typing import List, Dict, Any


def save_object(obj: Any, filepath: str) -> None:
    """Save an object to disk using pickle."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_object(filepath: str) -> Any:
    """Load an object from disk using pickle."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def ensure_dir(directory: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def format_document_preview(text: str, max_length: int = 200) -> str:
    """Format a document text for preview display."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def print_separator(char: str = "=", length: int = 80) -> None:
    """Print a separator line."""
    print(char * length)

