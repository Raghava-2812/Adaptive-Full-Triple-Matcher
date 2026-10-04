"""
Predicate Matching and Functionality / Inversability Calculation
for Full-Triple-Matcher (FTM).

Implements:
1. Functionality and Inverse Functionality computation over RDF graphs.
2. Inversability metric calculation: inversability(p) = functionality(p) * inverse_functionality(p).
3. Predicate alignment between heterogeneous Knowledge Graphs based on label and functional similarity.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from rdflib import Graph, URIRef
from rapidfuzz import fuzz

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
utils_dir = os.path.abspath(os.path.join(current_dir, "..", "utils"))
for p in [current_dir, utils_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from preprocessing import extract_local_name, clean_text
    from normalization import clamp_score
except ImportError:
    from codes.utils.preprocessing import extract_local_name, clean_text
    from codes.utils.normalization import clamp_score


def calculate_predicate_metrics(graph: Graph, predicate: Union[str, URIRef]) -> Tuple[float, float, float]:
    """
    Computes (functionality, inverse_functionality, inversability) for a given predicate in an RDF graph.
    
    Definitions:
      functionality(p)         = |{s | (s, p, o) in G}| / |{(s, p, o) in G}|
      inverse_functionality(p) = |{o | (s, p, o) in G}| / |{(s, p, o) in G}|
      inversability(p)         = functionality(p) * inverse_functionality(p)
    """
    pred_ref = URIRef(str(predicate)) if not isinstance(predicate, URIRef) else predicate
    
    triples = list(graph.triples((None, pred_ref, None)))
    total_count = len(triples)
    if total_count == 0:
        return 0.0, 0.0, 0.0
    
    unique_subjects = len({s for s, _, _ in triples})
    unique_objects = len({o for _, _, o in triples})
    
    func = unique_subjects / total_count
    inv_func = unique_objects / total_count
    inversability = func * inv_func
    
    return round(func, 4), round(inv_func, 4), round(inversability, 4)


def compute_inversability_table(graph: Graph) -> pd.DataFrame:
    """
    Iterates over all distinct predicates in an RDF graph and returns a DataFrame of functionality metrics.
    """
    predicates = {p for _, p, _ in graph}
    rows = []
    for p in predicates:
        f, inv_f, inv_score = calculate_predicate_metrics(graph, p)
        rows.append({
            'predicate': str(p),
            'functionality': f,
            'inverse_functionality': inv_f,
            'inversability': inv_score
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by='inversability', ascending=False).reset_index(drop=True)
    return df


def match_predicates(
    kg1_inv_df: pd.DataFrame,
    kg2_inv_df: pd.DataFrame,
    threshold: float = 0.60
) -> pd.DataFrame:
    """
    Matches predicates across two KGs by combining lexical label similarity
    and difference in inversability scores.
    """
    matches = []
    for _, row1 in kg1_inv_df.iterrows():
        p1 = row1['predicate']
        label1 = extract_local_name(p1)
        inv1 = float(row1.get('inversability', 1.0))
        
        for _, row2 in kg2_inv_df.iterrows():
            p2 = row2['predicate']
            label2 = extract_local_name(p2)
            inv2 = float(row2.get('inversability', 1.0))
            
            # String similarity on predicate labels
            sim_str = fuzz.token_sort_ratio(clean_text(label1), clean_text(label2)) / 100.0
            
            # Functionality agreement
            diff_inv = abs(inv1 - inv2)
            func_agreement = max(0.0, 1.0 - diff_inv)
            
            # Combined similarity
            score = (0.7 * sim_str) + (0.3 * func_agreement)
            if score >= threshold:
                matches.append({
                    'p1': p1,
                    'p2': p2,
                    'label1': label1,
                    'label2': label2,
                    'similarity': round(score, 4),
                    'inversability_diff': round(diff_inv, 4)
                })
                
    res_df = pd.DataFrame(matches)
    if not res_df.empty:
        res_df = res_df.sort_values(by='similarity', ascending=False).reset_index(drop=True)
    return res_df
