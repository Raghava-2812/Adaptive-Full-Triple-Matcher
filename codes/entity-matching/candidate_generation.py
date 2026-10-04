"""
Candidate Generation and Similarity Computation for Full-Triple-Matcher (FTM).

Implements:
1. Entity extraction from RDF graphs.
2. Pairwise candidate generation using fuzzy string matching and blocking.
3. Similarity matrix computation between source and target entities.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
from rdflib import Graph, URIRef
from rapidfuzz import fuzz

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
utils_dir = os.path.abspath(os.path.join(current_dir, "..", "utils"))

for p in [current_dir, utils_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from preprocessing import extract_local_name, clean_text, extract_entity_label_from_graph
from normalization import clamp_score
from triple_matching import calculate_structural_similarity
try:
    from gpu_utils import gpu_generate_candidates_ngrams, is_cuda_available
except ImportError:
    gpu_generate_candidates_ngrams = None
    is_cuda_available = lambda: False


def extract_entities(graph: Graph) -> Set[str]:
    """
    Extracts all distinct subject and entity object URIs from an RDF graph.
    """
    entities = set()
    for s, _, o in graph:
        if isinstance(s, URIRef):
            entities.add(str(s))
        if isinstance(o, URIRef):
            entities.add(str(o))
    return entities


def generate_candidates(
    graph1: Optional[Graph] = None,
    graph2: Optional[Graph] = None,
    entities1: Optional[List[str]] = None,
    entities2: Optional[List[str]] = None,
    min_similarity: float = 0.30,
    top_k: int = 5,
    use_gpu: bool = False
) -> pd.DataFrame:
    """
    Generates candidate pairs between entities of KG1 and KG2 using lexical similarity
    and structural triple evidence.
    Supports GPU acceleration via cuBLAS tensor matrix multiplication for large entity sets.
    """
    if entities1 is None and graph1 is not None:
        entities1 = list(extract_entities(graph1))
    if entities2 is None and graph2 is not None:
        entities2 = list(extract_entities(graph2))

    if not entities1 or not entities2:
        return pd.DataFrame(columns=['e1', 'e2', 'sim', 'triple_sim'])

    # Pre-extract labels
    labels1 = {e: extract_entity_label_from_graph(graph1, e) for e in entities1}
    labels2 = {e: extract_entity_label_from_graph(graph2, e) for e in entities2}

    # GPU-accelerated candidate generation for large entity sets
    if use_gpu and is_cuda_available() and gpu_generate_candidates_ngrams is not None and len(entities1) > 200 and len(entities2) > 200:
        clean_l1 = [clean_text(labels1[e]) for e in entities1]
        clean_l2 = [clean_text(labels2[e]) for e in entities2]
        cand_df = gpu_generate_candidates_ngrams(
            labels1=clean_l1,
            labels2=clean_l2,
            entities1=entities1,
            entities2=entities2,
            min_similarity=min_similarity,
            top_k=top_k
        )
        if graph1 is not None and graph2 is not None and not cand_df.empty:
            t_sims = [calculate_structural_similarity(r['e1'], r['e2'], graph1, graph2) for _, r in cand_df.iterrows()]
            cand_df['triple_sim'] = [round(s, 4) for s in t_sims]
        return cand_df

    records = []
    for e1 in entities1:
        l1 = labels1[e1]
        c1 = clean_text(l1)
        scored = []
        for e2 in entities2:
            l2 = labels2[e2]
            c2 = clean_text(l2)
            
            # Weighted string similarity
            ratio = fuzz.ratio(c1, c2) / 100.0
            token_ratio = fuzz.token_sort_ratio(c1, c2) / 100.0
            sim = 0.5 * ratio + 0.5 * token_ratio

            if sim >= min_similarity:
                triple_sim = calculate_structural_similarity(e1, e2, graph1, graph2)
                scored.append((e2, round(sim, 4), round(triple_sim, 4)))

        # Keep top-k candidates for e1
        scored.sort(key=lambda x: x[1], reverse=True)
        for e2, sim_val, t_sim_val in scored[:top_k]:
            records.append({
                'e1': e1,
                'e2': e2,
                'sim': sim_val,
                'triple_sim': t_sim_val
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by=['e1', 'sim'], ascending=[True, False]).reset_index(drop=True)
    return df
