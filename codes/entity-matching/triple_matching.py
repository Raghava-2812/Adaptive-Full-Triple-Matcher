"""
Triple Matching and Datatype-Aware Similarity for Full-Triple-Matcher (FTM).

Implements:
1. Object datatype detection (Entity, String, Number, Date).
2. Cross-datatype similarity metrics (Euclidean, Indel, Levenshtein, RapidFuzz WRatio).
3. Full triple similarity: sim(t1, t2) = w_s * sim(s1, s2) + w_p * sim(p1, p2) + w_o * sim(o1, o2).
4. Structural neighborhood consistency between entities across graphs.
"""

import math
import re
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from rdflib import URIRef, Literal, Graph
from rapidfuzz import fuzz, distance
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
utils_dir = os.path.abspath(os.path.join(current_dir, "..", "utils"))
for p in [current_dir, utils_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from preprocessing import extract_local_name, get_timestamp
    from normalization import clamp_score
except ImportError:
    from codes.utils.preprocessing import extract_local_name, get_timestamp
    from codes.utils.normalization import clamp_score

# Datatype categories
ENTITY = 'entity'
NUMBER = 'numbers'
DATE = 'date'
STRING = 'string'

NUMBER_TYPES = (int, float)


def check_object_type(o: Any) -> str:
    """
    Identifies datatype category of an RDF object.
    """
    if isinstance(o, URIRef):
        return ENTITY
    if isinstance(o, Literal):
        py_val = o.toPython()
        if isinstance(py_val, NUMBER_TYPES):
            return NUMBER
        if isinstance(py_val, (datetime.date, datetime.datetime)):
            return DATE
        return STRING
    if isinstance(o, NUMBER_TYPES):
        return NUMBER
    if isinstance(o, (datetime.date, datetime.datetime)):
        return DATE
    return STRING


def number_number_similarity(x: float, y: float) -> float:
    """
    Computes Euclidean similarity for two numeric values.
    """
    if x == y:
        return 1.0
    diff = abs(x - y)
    max_d = abs(x) + abs(y)
    if max_d == 0:
        return 1.0
    return clamp_score(1.0 - (diff / max_d))


def string_entity_similarity(s1: Any, e2: Any) -> float:
    """
    Computes fuzzy token ratio between a string literal and an entity URI label.
    """
    label2 = extract_local_name(str(e2))
    if not label2:
        return 0.0
    return fuzz.WRatio(str(s1), label2) / 100.0


def string_number_similarity(s1: Any, n2: float) -> float:
    """
    Extracts embedded numbers from string and calculates Euclidean & Indel similarity.
    """
    s1_str = str(s1)
    numbers = [float(n) for n in re.findall(r'-?\d+\.?\d*', s1_str)]
    max_sim = 0.0
    for n1 in numbers:
        sim = number_number_similarity(n1, n2)
        if sim > max_sim:
            max_sim = sim
    
    indel_sim = distance.Indel.normalized_similarity(s1_str, str(n2))
    return max(max_sim, indel_sim)


def date_date_similarity(d1: Any, d2: Any) -> float:
    """
    Compares two dates by timestamp distance.
    """
    t1 = get_timestamp(d1)
    t2 = get_timestamp(d2)
    return number_number_similarity(t1, t2)


def string_date_similarity(s1: Any, d2: Any) -> float:
    """
    Parses date from string if possible, otherwise falls back to string similarity.
    """
    try:
        parsed_d1 = dparser.parse(str(s1), fuzzy=True)
        return date_date_similarity(parsed_d1, d2)
    except Exception:
        return string_string_similarity(str(s1), str(d2))


def number_date_similarity(n1: float, d2: Any) -> float:
    """
    Compares a number to date via timestamp or string representation.
    """
    sim_num = number_number_similarity(float(n1), get_timestamp(d2))
    sim_str = string_string_similarity(str(n1), str(d2))
    return max(sim_num, sim_str)


def string_string_similarity(s1: Any, s2: Any) -> float:
    """
    Fuzzy ratio string similarity in [0.0, 1.0].
    """
    return fuzz.WRatio(str(s1), str(s2)) / 100.0


def get_objects_similarity(o1: Any, o2: Any) -> float:
    """
    Calculates polymorphic similarity between two RDF objects based on their types.
    """
    if str(o1) == str(o2):
        return 1.0

    t1 = check_object_type(o1)
    t2 = check_object_type(o2)

    # Entity to String / String to Entity
    if t1 == ENTITY and t2 == STRING:
        return string_entity_similarity(o2, o1)
    if t2 == ENTITY and t1 == STRING:
        return string_entity_similarity(o1, o2)
    if t1 == ENTITY and t2 == ENTITY:
        return string_string_similarity(extract_local_name(str(o1)), extract_local_name(str(o2)))

    val1 = o1.toPython() if isinstance(o1, Literal) else o1
    val2 = o2.toPython() if isinstance(o2, Literal) else o2

    # Number comparisons
    if t1 == NUMBER and t2 == NUMBER:
        return number_number_similarity(val1, val2)
    if t1 == NUMBER and t2 == STRING:
        return string_number_similarity(val2, val1)
    if t2 == NUMBER and t1 == STRING:
        return string_number_similarity(val1, val2)
    if t1 == NUMBER and t2 == DATE:
        return number_date_similarity(val1, val2)
    if t2 == NUMBER and t1 == DATE:
        return number_date_similarity(val2, val1)

    # Date comparisons
    if t1 == DATE and t2 == DATE:
        return date_date_similarity(val1, val2)
    if t1 == DATE and t2 == STRING:
        return string_date_similarity(val2, val1)
    if t2 == DATE and t1 == STRING:
        return string_date_similarity(val1, val2)

    # Fallback to string similarity
    return string_string_similarity(str(val1), str(val2))


def calculate_triple_similarity(
    s_sim: float,
    p_sim: float,
    o_sim: float,
    w_s: float = 0.40,
    w_p: float = 0.20,
    w_o: float = 0.40
) -> float:
    """
    Computes overall triple similarity from subject, predicate, and object similarities.
    """
    return clamp_score(w_s * s_sim + w_p * p_sim + w_o * o_sim)


def calculate_structural_similarity(
    e1_uri: str,
    e2_uri: str,
    graph1: Optional[Graph],
    graph2: Optional[Graph]
) -> float:
    """
    Evaluates graph neighborhood agreement: compares the outgoing predicates and objects
    of e1 in graph1 against e2 in graph2.
    """
    if graph1 is None or graph2 is None:
        return 0.5  # Neutral default when graphs are not available

    t1_list = list(graph1.triples((URIRef(e1_uri), None, None)))
    t2_list = list(graph2.triples((URIRef(e2_uri), None, None)))

    if not t1_list or not t2_list:
        return 0.5

    total_sim = 0.0
    comparisons = 0

    for _, p1, o1 in t1_list:
        p1_name = extract_local_name(str(p1)).lower()
        for _, p2, o2 in t2_list:
            p2_name = extract_local_name(str(p2)).lower()
            p_match = fuzz.token_sort_ratio(p1_name, p2_name) / 100.0
            if p_match >= 0.5:
                o_sim = get_objects_similarity(o1, o2)
                sim = 0.3 * p_match + 0.7 * o_sim
                total_sim += sim
                comparisons += 1

    if comparisons == 0:
        return 0.4
    return clamp_score(total_sim / comparisons)
