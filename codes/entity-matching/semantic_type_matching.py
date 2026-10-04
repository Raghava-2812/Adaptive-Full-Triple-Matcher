"""
Semantic Type Matching for Full-Triple-Matcher (FTM)
Improvement 1: Generic semantic category and ontological type identification,
compatibility evaluation, and evidence generation for Knowledge Graph entity alignment.

This module implements:
1. Generic Semantic Type Extraction:
   - Explicit RDF types (rdf:type, wdt:P31, wdt:P279, schema:additionalType, dcterms:type, etc.)
   - Explicit type-like predicates (category, genre, class, type)
   - Implicit relational roles from incoming edges (range typing, e.g. pointed to by director -> Director, genre -> Genre)
   - Implicit structural domain profiles from outgoing predicate signatures (e.g. director + runtime + year -> Movie)
   - Entity URI disambiguation patterns (e.g., parenthesized suffixes like (film), (album), (city))
   - Support for precomputed semantic probabilities (e.g., entity_sem_prob.csv for large graphs)

2. Generic Semantic Type Compatibility & Similarity:
   - Ontological cluster alignment across heterogeneous schemas (e.g. Movie <-> Film, Director <-> directedBy, Genre <-> Category)
   - RapidFuzz token-level and substring similarity for unseen categories
   - Mutually-exclusive category conflict detection (e.g. Movie vs. Director -> Incompatible)

3. Semantic Evidence Generation:
   - Output structured semantic type features per candidate pair:
     * semantic_type_kg1: Extracted semantic type for entity 1
     * semantic_type_kg2: Extracted semantic type for entity 2
     * semantic_type_similarity: Normalized semantic type similarity [0.0, 1.0]
     * semantic_type_match: Boolean compatibility flag
     * semantic_type_bonus: Bounded confidence bonus passed to Improvement 2
     * semantic_evidence_source: Provenance explanation

Author: CSE Project - Full-Triple-Matcher Enhancements (Improvement 1)
"""

import os
import sys
import re
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import pandas as pd
from rdflib import Graph, URIRef, Literal
from rapidfuzz import fuzz

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
utils_dir = os.path.abspath(os.path.join(current_dir, "..", "utils"))
for p in [current_dir, utils_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from preprocessing import extract_local_name, clean_text
from normalization import clamp_score
try:
    from gpu_utils import gpu_accelerate_semantic_bonuses, is_cuda_available
except ImportError:
    gpu_accelerate_semantic_bonuses = None
    is_cuda_available = lambda: False

# ==============================================================================
# CONFIGURATION SECTION
# ==============================================================================
# Setting USE_SEMANTIC_TYPE_MATCHING = False preserves baseline FTM behavior
USE_SEMANTIC_TYPE_MATCHING: bool = True

SEMANTIC_BONUS_WEIGHT: float = 0.10     # Max bonus awarded for strong semantic type match
SEMANTIC_COMPATIBILITY_THRESHOLD: float = 0.70  # Min type similarity to consider compatible
SEMANTIC_INCOMPATIBILITY_THRESHOLD: float = 0.25 # Below this, types are considered conflicting

# Common RDF predicates used for explicit typing across ontologies
EXPLICIT_TYPE_PREDICATES: List[str] = [
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://www.wikidata.org/entity/P31",       # instance of
    "http://www.wikidata.org/entity/P279",      # subclass of
    "http://schema.org/additionalType",
    "http://schema.org/genre",
    "http://purl.org/dc/terms/type",
    "http://purl.org/dc/elements/1.1/type",
    "http://complex-kg1.org/ontology/type",
    "http://complex-kg2.org/ontology/category"
]

# Generic cross-domain ontological synonym clusters
GENERIC_SEMANTIC_CLUSTERS: Dict[str, Set[str]] = {
    "movie": {
        "movie", "film", "motion picture", "cinema", "feature film", "documentary",
        "film title", "creative work", "production", "video", "broadcast"
    },
    "director": {
        "director", "directedby", "directed by", "filmmaker", "filmmakers",
        "creator", "filmdirector", "stage director"
    },
    "genre": {
        "genre", "category", "theme", "classification", "type", "categories", "genres"
    },
    "actor": {
        "actor", "actress", "star", "cast", "starring", "performer", "voice actor"
    },
    "writer": {
        "writer", "author", "screenwriter", "writtenby", "written by", "scriptwriter"
    },
    "person": {
        "person", "human", "individual", "director", "actor", "writer", "producer",
        "artist", "agent", "musician", "creator", "fullname", "name"
    },
    "location": {
        "location", "place", "city", "country", "nation", "state", "region",
        "town", "settlement", "origincountry", "origin country", "birthplace", "deathplace"
    },
    "city": {
        "city", "town", "municipality", "metropolis", "capital", "urban area", "largestcity"
    },
    "country": {
        "country", "nation", "sovereign state", "state", "origincountry", "origin country"
    },
    "album": {
        "album", "record", "music album", "disc", "ep", "lp", "studio album", "musical work"
    },
    "song": {
        "song", "track", "single", "musical composition", "sound recording"
    },
    "book": {
        "book", "novel", "literary work", "publication", "volume", "story"
    },
    "organization": {
        "organization", "company", "airline", "publisher", "studio", "institution", "agency"
    },
    "scifi": {
        "scifi", "sci fi", "sci-fi", "science fiction", "sciencefiction", "speculative fiction"
    },
    "action": {
        "action", "action film", "action movie", "action genre"
    },
    "romance": {
        "romance", "romantic", "romantic drama", "romantic comedy", "romance film"
    },
    "drama": {
        "drama", "dramatic film", "romantic drama", "period drama"
    },
    "thriller": {
        "thriller", "suspense", "mystery", "crime thriller", "psychological thriller"
    }
}

# Mutually exclusive broad categories
DISJOINT_CATEGORIES: List[Tuple[Set[str], Set[str]]] = [
    ({"movie", "film"}, {"person", "director", "actor", "writer"}),
    ({"movie", "film"}, {"genre", "category"}),
    ({"movie", "film"}, {"location", "city", "country"}),
    ({"person", "director", "actor"}, {"genre", "category"}),
    ({"person", "director", "actor"}, {"location", "city", "country"}),
    ({"genre", "category"}, {"location", "city", "country"}),
    ({"album", "song"}, {"person", "director", "actor"}),
    ({"book"}, {"person", "director", "actor"}),
    ({"director"}, {"actor"}),
    ({"city"}, {"country"})
]


# ==============================================================================
# 1. GENERIC SEMANTIC TYPE EXTRACTION
# ==============================================================================

def extract_entity_semantic_types(
    graph: Optional[Graph],
    entity_uri: str,
    precomputed_types: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Extracts semantic categories, classes, and roles for an entity in an RDF graph.

    Hierarchical derivation:
    1. Precomputed dictionary lookup
    2. Incoming relational roles (range typing, e.g. pointed to by director or genre)
    3. Outgoing relational domain signatures (e.g. has director, runtime, year -> movie)
    4. Explicit rdf:type and type-defining predicates
    5. URI lexical naming and parenthesized disambiguation (e.g. She_(1965_film) -> film)
    """
    if precomputed_types and entity_uri in precomputed_types:
        val = clean_text(precomputed_types[entity_uri])
        return {
            "primary_type": val,
            "all_types": {val},
            "source": "precomputed"
        }

    types: Set[str] = set()
    source: str = "unknown"

    if graph is not None:
        ref = URIRef(str(entity_uri))

        # 1. Explicit rdf:type, category, and well-known type predicates (Highest confidence)
        for p_str in EXPLICIT_TYPE_PREDICATES:
            for _, _, o in graph.triples((ref, URIRef(p_str), None)):
                t_name = clean_text(extract_local_name(str(o)))
                if t_name:
                    types.add(t_name)
                    source = "explicit_rdf_type"

        for _, p, o in graph.triples((ref, None, None)):
            p_local = extract_local_name(str(p)).lower()
            if p_local in ["type", "typeof", "class", "category", "classification"]:
                t_name = clean_text(extract_local_name(str(o)))
                if t_name:
                    types.add(t_name)
                    if source in ["unknown", "uri_disambiguation"]:
                        source = "explicit_predicate"

        # Expand explicit type names into ontology clusters
        cluster_expansions = set()
        for t in types:
            for c_name, c_set in GENERIC_SEMANTIC_CLUSTERS.items():
                if t in c_set or any(s in t for s in c_set):
                    cluster_expansions.add(c_name)
        types.update(cluster_expansions)

        # Check outgoing & incoming predicate local names
        out_preds = {extract_local_name(str(p)).lower() for _, p, _ in graph.triples((ref, None, None))}
        in_preds = {extract_local_name(str(p)).lower() for _, p, _ in graph.triples((None, None, ref))}

        # 2. Incoming predicate roles (Range typing)
        if in_preds.intersection({"genre", "category"}):
            types.add("genre")
            types.add("category")
            if source == "unknown":
                source = "incoming_relation"

        if in_preds.intersection({"director", "directedby", "directed_by"}):
            types.add("director")
            types.add("person")
            if source == "unknown":
                source = "incoming_relation"

        if in_preds.intersection({"writer", "writtenby", "author"}):
            types.add("writer")
            types.add("person")
            if source == "unknown":
                source = "incoming_relation"

        if in_preds.intersection({"actor", "starring", "cast"}):
            types.add("actor")
            types.add("person")
            if source == "unknown":
                source = "incoming_relation"

        if in_preds.intersection({"city", "capital", "largestcity", "filminglocation", "birthplace"}):
            types.add("city")
            types.add("location")
            if source == "unknown":
                source = "incoming_relation"

        if in_preds.intersection({"country", "origincountry", "sourcecountry", "nation", "citizenship"}):
            types.add("country")
            types.add("location")
            if source == "unknown":
                source = "incoming_relation"

        # 3. Outgoing predicate domain signatures
        movie_features = {"director", "directedby", "runtime", "durationminutes", "releaseyear", "year"}
        if len(out_preds.intersection(movie_features)) >= 2 or ("director" in out_preds or "directedby" in out_preds):
            types.add("movie")
            types.add("film")
            if source == "unknown":
                source = "outgoing_signature"
        elif not types and out_preds.intersection({"birthdate", "deathdate", "citizenship", "nationality", "occupation"}):
            types.add("person")
            if source == "unknown":
                source = "outgoing_signature"
        elif not types and out_preds.intersection({"fullname"}):
            types.add("person")
            if source == "unknown":
                source = "outgoing_signature"

    # 4. URI naming / parenthesized suffix extraction (e.g. She_(1965_film) -> 'film')
    uri_str = str(entity_uri)
    m = re.search(r'\(([^)]+)\)$', uri_str)
    if m:
        suffix_type = clean_text(m.group(1))
        if suffix_type and not suffix_type.isdigit():
            types.add(suffix_type)
            if source == "unknown":
                source = "uri_disambiguation"

    # Fallback to local name token if entity URI contains keyword cues
    local_name = extract_local_name(uri_str).lower()
    for cluster_name, keywords in GENERIC_SEMANTIC_CLUSTERS.items():
        if any(kw in local_name for kw in keywords):
            types.add(cluster_name)
            if source == "unknown":
                source = "uri_keyword"

    # Select primary semantic type
    primary_type = "unknown"
    if types:
        priority_keys = ["genre", "category", "director", "movie", "film", "actor", "writer", "city", "country", "location", "album", "person"]
        for pk in priority_keys:
            if any(pk in t for t in types):
                primary_type = pk
                break
        if primary_type == "unknown":
            primary_type = sorted(list(types))[0]

    return {
        "primary_type": primary_type,
        "all_types": types,
        "source": source
    }


def extract_graph_semantic_type_dict(
    graph: Optional[Graph],
    entities: Optional[Set[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Precomputes semantic types for all entities in a graph.
    """
    if graph is None:
        return {}

    if entities is None:
        entities = set()
        for s, _, o in graph:
            if isinstance(s, URIRef):
                entities.add(str(s))
            if isinstance(o, URIRef):
                entities.add(str(o))

    type_dict = {}
    for e in entities:
        type_dict[e] = extract_entity_semantic_types(graph, e)
    return type_dict


# ==============================================================================
# 2. GENERIC SEMANTIC TYPE SIMILARITY & COMPATIBILITY
# ==============================================================================

def are_types_disjoint(types1: Set[str], types2: Set[str]) -> bool:
    """
    Checks if any type in types1 is mutually exclusive with any type in types2.
    """
    for cat_a, cat_b in DISJOINT_CATEGORIES:
        match_a1 = bool(types1.intersection(cat_a))
        match_b2 = bool(types2.intersection(cat_b))
        if match_a1 and match_b2:
            return True

        match_b1 = bool(types1.intersection(cat_b))
        match_a2 = bool(types2.intersection(cat_a))
        if match_b1 and match_a2:
            return True
    return False


def calculate_semantic_type_similarity(
    type1: str,
    type2: str,
    all_types1: Optional[Set[str]] = None,
    all_types2: Optional[Set[str]] = None
) -> Tuple[float, bool]:
    """
    Calculates semantic similarity and compatibility between two entity types.

    Parameters:
        type1: Primary semantic type of KG1 entity.
        type2: Primary semantic type of KG2 entity.
        all_types1: Full set of inferred types for KG1 entity (optional).
        all_types2: Full set of inferred types for KG2 entity (optional).

    Returns:
        Tuple (similarity, is_compatible):
            similarity: float in [0.0, 1.0]
            is_compatible: boolean True if types represent compatible meanings
    """
    t1_clean = clean_text(type1)
    t2_clean = clean_text(type2)

    set1 = set(clean_text(t) for t in (all_types1 or {type1}))
    set2 = set(clean_text(t) for t in (all_types2 or {type2}))

    # If either type is unknown, treat as neutral compatibility
    if t1_clean in ["unknown", ""] or t2_clean in ["unknown", ""]:
        return 0.50, True

    # 1. Exact string match
    if t1_clean == t2_clean:
        return 1.0, True

    # 2. Check disjoint mutually exclusive categories
    if are_types_disjoint(set1, set2):
        return 0.0, False

    # 3. Check generic semantic synonym clusters
    for cluster_name, syn_set in GENERIC_SEMANTIC_CLUSTERS.items():
        in_cluster_1 = bool(set1.intersection(syn_set)) or any(s in t1_clean for s in syn_set)
        in_cluster_2 = bool(set2.intersection(syn_set)) or any(s in t2_clean for s in syn_set)
        if in_cluster_1 and in_cluster_2:
            return 1.0, True

    # 4. Token set ratio / fuzzy token similarity
    sim_ratio = fuzz.token_set_ratio(t1_clean, t2_clean) / 100.0
    if sim_ratio >= SEMANTIC_COMPATIBILITY_THRESHOLD:
        return round(sim_ratio, 4), True

    # Maximum similarity across all type combinations
    max_sim = sim_ratio
    for t_a in set1:
        for t_b in set2:
            s_ab = fuzz.token_set_ratio(t_a, t_b) / 100.0
            if s_ab > max_sim:
                max_sim = s_ab

    is_comp = max_sim >= SEMANTIC_COMPATIBILITY_THRESHOLD
    if max_sim < SEMANTIC_INCOMPATIBILITY_THRESHOLD:
        is_comp = False

    return round(clamp_score(max_sim), 4), is_comp


def calculate_semantic_type_bonus(
    semantic_similarity: float,
    is_compatible: bool,
    weight: float = SEMANTIC_BONUS_WEIGHT
) -> float:
    """
    Computes bounded bonus to reduce threshold in Improvement 2.
    Only awarded if semantic types are compatible.
    """
    if not is_compatible or semantic_similarity < SEMANTIC_COMPATIBILITY_THRESHOLD:
        return 0.0
    bonus = float(weight) * float(semantic_similarity)
    return round(clamp_score(bonus), 4)


# ==============================================================================
# 3. PIPELINE INTEGRATION: APPLY SEMANTIC TYPE MATCHING
# ==============================================================================

def apply_semantic_type_matching(
    candidates_df: pd.DataFrame,
    graph1: Optional[Graph] = None,
    graph2: Optional[Graph] = None,
    entity_sem_prob_df: Optional[pd.DataFrame] = None,
    entity_col1: str = 'e1',
    entity_col2: str = 'e2',
    sim_col: str = 'sim',
    semantic_weight: float = SEMANTIC_BONUS_WEIGHT,
    use_semantic_type: bool = USE_SEMANTIC_TYPE_MATCHING,
    use_gpu: bool = False
) -> pd.DataFrame:
    """
    Enhances candidates DataFrame with structured semantic type information.
    Supports GPU tensor acceleration when precomputed probabilities or large batches are available.
    """
    if candidates_df.empty:
        df = candidates_df.copy()
        for col in ['semantic_type_kg1', 'semantic_type_kg2', 'semantic_type_similarity',
                    'semantic_type_match', 'semantic_type_bonus', 'semantic_evidence_source']:
            df[col] = []
        return df

    df = candidates_df.copy()

    # When disabled, provide neutral baseline passthrough
    if not use_semantic_type:
        df['semantic_type_kg1'] = 'unspecified'
        df['semantic_type_kg2'] = 'unspecified'
        df['semantic_type_similarity'] = 0.0
        df['semantic_type_match'] = True
        df['semantic_type_bonus'] = 0.0
        df['semantic_evidence_source'] = 'disabled'
        return df

    # Check for precomputed semantic probabilities lookup table (e.g. DW-NB)
    if entity_sem_prob_df is not None and not entity_sem_prob_df.empty:
        e1_c = 'e1' if 'e1' in entity_sem_prob_df.columns else entity_sem_prob_df.columns[0]
        e2_c = 'e2' if 'e2' in entity_sem_prob_df.columns else entity_sem_prob_df.columns[1]
        s_c = 'sim' if 'sim' in entity_sem_prob_df.columns else entity_sem_prob_df.columns[2]

        merged = df[[entity_col1, entity_col2]].merge(
            entity_sem_prob_df[[e1_c, e2_c, s_c]].rename(columns={e1_c: entity_col1, e2_c: entity_col2, s_c: '_sem_sim'}),
            on=[entity_col1, entity_col2],
            how='left'
        )

        if merged['_sem_sim'].notna().all():
            sem_sim_arr = merged['_sem_sim'].values.astype(np.float32)
            is_comp_arr = sem_sim_arr >= SEMANTIC_COMPATIBILITY_THRESHOLD

            if use_gpu and is_cuda_available() and gpu_accelerate_semantic_bonuses is not None:
                bonuses = gpu_accelerate_semantic_bonuses(sem_sim_arr, is_comp_arr, semantic_weight)
            else:
                bonuses = np.where(is_comp_arr, np.clip(sem_sim_arr * semantic_weight, 0.0, semantic_weight), 0.0)

            df['semantic_type_kg1'] = 'entity'
            df['semantic_type_kg2'] = 'entity'
            df['semantic_type_similarity'] = np.round(sem_sim_arr, 4)
            df['semantic_type_match'] = is_comp_arr
            df['semantic_type_bonus'] = bonuses
            df['semantic_evidence_source'] = 'precomputed_sem_prob'
            return df

    precomputed_lookup = {}
    if entity_sem_prob_df is not None and not entity_sem_prob_df.empty:
        e1_c = 'e1' if 'e1' in entity_sem_prob_df.columns else entity_sem_prob_df.columns[0]
        e2_c = 'e2' if 'e2' in entity_sem_prob_df.columns else entity_sem_prob_df.columns[1]
        s_c = 'sim' if 'sim' in entity_sem_prob_df.columns else entity_sem_prob_df.columns[2]
        for _, r in entity_sem_prob_df.iterrows():
            precomputed_lookup[(str(r[e1_c]).strip(), str(r[e2_c]).strip())] = float(r[s_c])

    # Pre-extract types for distinct entities in candidates
    distinct_e1 = set(df[entity_col1].dropna().astype(str).unique())
    distinct_e2 = set(df[entity_col2].dropna().astype(str).unique())

    types_g1 = extract_graph_semantic_type_dict(graph1, distinct_e1) if graph1 is not None else {}
    types_g2 = extract_graph_semantic_type_dict(graph2, distinct_e2) if graph2 is not None else {}

    t1_list = []
    t2_list = []
    sim_list = []
    match_list = []
    bonus_list = []
    source_list = []

    for _, row in df.iterrows():
        e1_val = str(row[entity_col1]).strip()
        e2_val = str(row[entity_col2]).strip()

        # Check precomputed lookup first
        if (e1_val, e2_val) in precomputed_lookup:
            sem_sim = precomputed_lookup[(e1_val, e2_val)]
            is_comp = sem_sim >= SEMANTIC_COMPATIBILITY_THRESHOLD
            bonus = calculate_semantic_type_bonus(sem_sim, is_comp, semantic_weight)

            t1_info = types_g1.get(e1_val, extract_entity_semantic_types(None, e1_val))
            t2_info = types_g2.get(e2_val, extract_entity_semantic_types(None, e2_val))

            t1_list.append(t1_info["primary_type"])
            t2_list.append(t2_info["primary_type"])
            sim_list.append(round(sem_sim, 4))
            match_list.append(is_comp)
            bonus_list.append(bonus)
            source_list.append("precomputed_sem_prob")
            continue

        # Extract from graph or URI heuristics
        t1_info = types_g1.get(e1_val)
        if t1_info is None:
            t1_info = extract_entity_semantic_types(graph1, e1_val)
            types_g1[e1_val] = t1_info

        t2_info = types_g2.get(e2_val)
        if t2_info is None:
            t2_info = extract_entity_semantic_types(graph2, e2_val)
            types_g2[e2_val] = t2_info

        sem_sim, is_comp = calculate_semantic_type_similarity(
            t1_info["primary_type"],
            t2_info["primary_type"],
            t1_info["all_types"],
            t2_info["all_types"]
        )
        bonus = calculate_semantic_type_bonus(sem_sim, is_comp, semantic_weight)

        src = f"{t1_info['source']}+{t2_info['source']}"

        t1_list.append(t1_info["primary_type"])
        t2_list.append(t2_info["primary_type"])
        sim_list.append(sem_sim)
        match_list.append(is_comp)
        bonus_list.append(bonus)
        source_list.append(src)

    df['semantic_type_kg1'] = t1_list
    df['semantic_type_kg2'] = t2_list
    df['semantic_type_similarity'] = sim_list
    df['semantic_type_match'] = match_list
    df['semantic_type_bonus'] = bonus_list
    df['semantic_evidence_source'] = source_list

    return df
