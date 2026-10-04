"""
File and I/O Utilities for Full-Triple-Matcher (FTM)
Provides safe loading, saving, and directory management for RDF, CSV, and JSON data.
"""

import os
import json
from typing import Any, Dict, Optional
import pandas as pd


def ensure_dir(path: str) -> str:
    """
    Ensures that directory exists for a given file path or folder path.
    """
    if os.path.splitext(path)[1]:
        dir_name = os.path.dirname(path)
    else:
        dir_name = path
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    return path


def load_rdf_graph(file_path: str, format: str = "turtle"):
    """
    Safely loads and parses an RDF graph from file using rdflib.
    Returns rdflib.Graph or None if loading fails.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"RDF file not found: {file_path}")
    try:
        from rdflib import Graph
        g = Graph()
        g.parse(file_path, format=format)
        return g
    except Exception as e:
        print(f"Warning: Failed to parse RDF graph {file_path}: {e}")
        return None


def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    return pd.read_csv(file_path, **kwargs)


def save_csv(df: pd.DataFrame, file_path: str, index: bool = False, **kwargs) -> str:
    """
    Saves a DataFrame to CSV, automatically creating parent directories.
    """
    ensure_dir(file_path)
    df.to_csv(file_path, index=index, **kwargs)
    return file_path


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Loads JSON file into Python dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 4) -> str:
    """
    Serializes data to JSON format, creating parent directories.
    """
    ensure_dir(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    return file_path
