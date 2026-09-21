from rdflib import URIRef, Literal

import string

import dateutil.parser as dparser

import datetime

from rapidfuzz import fuzz, distance

import math

import re


# ============================================================
# Semantic Type-Aware Matching
# ============================================================

try:
    from semantic_type_matcher import (
        USE_SEMANTIC_TYPE_MATCHING,
        SEMANTIC_TYPE_WEIGHT,
        get_semantic_type_bonus,
    )

except ModuleNotFoundError:

    import os
    import importlib.util

    module_path = os.path.join(
        os.path.dirname(__file__),
        "semantic_type_matcher.py"
    )

    spec = importlib.util.spec_from_file_location(
        "semantic_type_matcher",
        module_path
    )

    semantic_type_matcher = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(semantic_type_matcher)

    USE_SEMANTIC_TYPE_MATCHING = (
        semantic_type_matcher.USE_SEMANTIC_TYPE_MATCHING
    )

    SEMANTIC_TYPE_WEIGHT = (
        semantic_type_matcher.SEMANTIC_TYPE_WEIGHT
    )

    get_semantic_type_bonus = (
        semantic_type_matcher.get_semantic_type_bonus
    )


# ============================================================
# Original FTM constants
# ============================================================

string_types = [
    string,
    Literal,
    None,
    str
]

number_types = [
    int,
    float
]

LABEL_PREDICATE = (
    "http://www.w3.org/2000/01/rdf-schema#label"
)

UNDEFINED = "undefined"

STRING = "string"

ENTITY = "entity"

NUMBER = "numbers"

DATE = "date"

CATEGORICAL = "categorical"


# ============================================================
# URL utility
# ============================================================

def get_last_part_url(uri):
    """
    Extract the last part of a URI.

    Example:
        http://example.org/Raj
        -> Raj
    """

    return uri.rsplit("/", 1)[-1]


# ============================================================
# Original FTM object type detection
# ============================================================

def check_object_type(o):
    """
    Determine the original FTM object type.

    Returns:
        STRING
        ENTITY
        NUMBER
        DATE
    """

    if o is None:
        return STRING

    if type(o) == URIRef:
        return ENTITY

    native_value = o

    if hasattr(o, "toPython"):

        try:
            native_value = o.toPython()

        except Exception:
            native_value = o

    datatype = type(native_value)

    if datatype in number_types:
        return NUMBER

    if datatype == datetime.date:
        return DATE

    return STRING


# ============================================================
# Number similarity
# ============================================================

def number_number_similarity(x, y):
    """
    Calculate similarity between two numbers.
    """

    if x == y:
        return 1.0

    difference = x - y

    squared_difference = difference ** 2

    distance_value = math.sqrt(
        squared_difference
    )

    maximum_distance = abs(x) + abs(y)

    # Prevent division by zero
    if maximum_distance == 0:
        return 1.0

    scaled_distance = (
        distance_value / maximum_distance
    )

    return 1 - scaled_distance


# ============================================================
# String-Entity similarity
# ============================================================

def string_entity_similarity(s1, e2):
    """
    Compare a string with an RDF entity URI.
    """

    entity_label = get_last_part_url(e2)

    if not entity_label:
        return 0.0

    s1_str = str(s1)

    max_ratio = 0.0

    new_ratio = (
        fuzz.WRatio(
            s1_str,
            entity_label
        ) / 100
    )

    if max_ratio < new_ratio:
        max_ratio = new_ratio

    return max_ratio


# ============================================================
# String-Number similarity
# ============================================================

def string_number_similarity(s1, n2):
    """
    Compare a string containing a number with a number.
    """

    s1_str = str(s1)

    # Extract integer / decimal numbers from text.
    number_list = [
        float(s)
        for s in re.findall(
            r"-?\d+(?:\.\d+)?",
            s1_str
        )
    ]

    # --------------------------------------------------------
    # Numeric similarity
    # --------------------------------------------------------

    max_sim = 0.0

    for n1 in number_list:

        new_sim = number_number_similarity(
            n1,
            n2
        )

        if max_sim < new_sim:
            max_sim = new_sim

    # --------------------------------------------------------
    # String edit similarity
    # --------------------------------------------------------

    edit_similarity = (
        distance.Indel.normalized_similarity(
            s1_str,
            str(n2)
        )
    )

    if edit_similarity > max_sim:
        max_sim = edit_similarity

    return max_sim


# ============================================================
# Date timestamp
# ============================================================

def get_timestamp(d):
    """
    Convert a date into a timestamp.
    """

    datetime_obj = datetime.datetime.combine(
        d,
        datetime.time.min
    )

    try:

        return datetime_obj.timestamp()

    except Exception as e:

        print(e)

        return 0.0


# ============================================================
# String-Date similarity
# ============================================================

def string_date_similarity(s1, d2):
    """
    Compare a date string with a date.
    """

    try:

        d1 = dparser.parse(
            s1,
            fuzzy=True
        )

        return date_date_similarity(
            d1,
            d2
        )

    except Exception:

        return string_string_similarity(
            s1,
            str(d2)
        )


# ============================================================
# Number-Date similarity
# ============================================================

def number_date_similarity(n1, d2):
    """
    Compare a number with a date.
    """

    sim_as_number = number_number_similarity(
        n1,
        get_timestamp(d2)
    )

    sim_as_string = string_string_similarity(
        str(n1),
        str(d2)
    )

    if sim_as_number > sim_as_string:
        return sim_as_number

    return sim_as_string


# ============================================================
# Date-Date similarity
# ============================================================

def date_date_similarity(d1, d2):
    """
    Compare two dates.
    """

    return number_number_similarity(
        get_timestamp(d1),
        get_timestamp(d2)
    )


# ============================================================
# String-String similarity
# ============================================================

def string_string_similarity(o1, o2):
    """
    Calculate string similarity using RapidFuzz WRatio.
    """

    return (
        fuzz.WRatio(
            str(o1),
            str(o2)
        ) / 100
    )


# ============================================================
# Safe Python value
# ============================================================

def _safe_python_value(value):
    """
    Safely convert RDFLib values to native Python values.
    """

    if value is None:
        return None

    if hasattr(value, "toPython"):

        try:

            return value.toPython()

        except Exception:

            pass

    return value


# ============================================================
# Semantic Type Bonus
# ============================================================

def apply_semantic_type_bonus(
    base_similarity,
    o1,
    o2,
    use_semantic_type_matching=None,
    semantic_type_weight=None,
    graph_1=None,
    graph_2=None,
    type_1=None,
    type_2=None,
):
    """
    Apply the Semantic Type-Aware Matching improvement.

    Formula:

        improved_score =
            base_score
            +
            semantic_type_weight
            *
            semantic_compatibility

    The final score is capped at 1.0.

    Important:
        The original FTM score is never replaced.
        Semantic type is only additional evidence.
    """

    # --------------------------------------------------------
    # Determine whether semantic matching is enabled
    # --------------------------------------------------------

    if use_semantic_type_matching is None:

        use_semantic_type_matching = (
            USE_SEMANTIC_TYPE_MATCHING
        )

    # --------------------------------------------------------
    # Get configured semantic weight
    # --------------------------------------------------------

    if semantic_type_weight is None:

        semantic_type_weight = (
            SEMANTIC_TYPE_WEIGHT
        )

    # --------------------------------------------------------
    # Baseline mode
    # --------------------------------------------------------

    if not use_semantic_type_matching:

        return base_similarity

    # --------------------------------------------------------
    # Calculate semantic compatibility
    # --------------------------------------------------------

    semantic_bonus = get_semantic_type_bonus(
        value_a=o1,
        value_b=o2,
        graph_a=graph_1,
        graph_b=graph_2,
        type_a=type_1,
        type_b=type_2,
    )

    # --------------------------------------------------------
    # No semantic evidence
    # --------------------------------------------------------

    if semantic_bonus <= 0:

        return base_similarity

    # --------------------------------------------------------
    # Add semantic evidence
    # --------------------------------------------------------

    improved_similarity = (
        base_similarity
        +
        semantic_type_weight * semantic_bonus
    )

    # --------------------------------------------------------
    # Keep similarity in [0, 1]
    # --------------------------------------------------------

    return min(
        1.0,
        improved_similarity
    )


# ============================================================
# Main Object Similarity
# ============================================================

def get_objects_similarity(
    o1,
    o2,
    use_semantic_type_matching=None,
    semantic_type_weight=None,
    graph_1=None,
    graph_2=None,
    type_1=None,
    type_2=None,
):
    """
    Calculate similarity between two FTM objects.

    Parameters
    ----------
    o1 : RDF object
        First object.

    o2 : RDF object
        Second object.

    use_semantic_type_matching : bool, optional
        Enable/disable semantic type-aware matching.

    semantic_type_weight : float, optional
        Weight assigned to semantic compatibility.

    graph_1, graph_2 : rdflib.Graph, optional
        Graph context used by the Semantic Type Transformer.

    type_1, type_2 : str, optional
        Explicit semantic type metadata when available.

    Returns
    -------
    float
        Similarity score in [0, 1].

    Pipeline
    --------

        Object pair
             |
             v
        Original FTM
        object similarity
             |
             v
        Base similarity
             |
             v
        Semantic Type
        Compatibility
             |
             v
        Small weighted bonus
             |
             v
        Final similarity
    """

    # ========================================================
    # Exact match
    # ========================================================

    if str(o1) == str(o2):

        return 1.0

    # ========================================================
    # Configuration
    # ========================================================

    if use_semantic_type_matching is None:

        use_semantic_type_matching = (
            USE_SEMANTIC_TYPE_MATCHING
        )

    if semantic_type_weight is None:

        semantic_type_weight = (
            SEMANTIC_TYPE_WEIGHT
        )

    # ========================================================
    # Original FTM object types
    # ========================================================

    o1_type = check_object_type(o1)

    o2_type = check_object_type(o2)

    # ========================================================
    # ENTITY -> STRING
    # ========================================================

    if o1_type == ENTITY:

        if o2_type == STRING:

            base_similarity = (
                string_entity_similarity(
                    o2,
                    o1
                )
            )

            return apply_semantic_type_bonus(
                base_similarity,
                o1,
                o2,
                use_semantic_type_matching=use_semantic_type_matching,
                semantic_type_weight=semantic_type_weight,
                graph_1=graph_1,
                graph_2=graph_2,
                type_1=type_1,
                type_2=type_2,
            )

    # ========================================================
    # Convert RDF values safely
    # ========================================================

    o1_python = _safe_python_value(o1)

    o2_python = _safe_python_value(o2)

    # ========================================================
    # STRING -> ENTITY
    # ========================================================

    if o2_type == ENTITY:

        if o1_type == STRING:

            base_similarity = (
                string_entity_similarity(
                    o1_python,
                    o2
                )
            )

            return apply_semantic_type_bonus(
                base_similarity,
                o1,
                o2,
                use_semantic_type_matching=use_semantic_type_matching,
                semantic_type_weight=semantic_type_weight,
                graph_1=graph_1,
                graph_2=graph_2,
                type_1=type_1,
                type_2=type_2,
            )

    # ========================================================
    # NUMBER
    # ========================================================

    if o1_type == NUMBER:

        if o2_type == NUMBER:

            base_similarity = (
                number_number_similarity(
                    o1_python,
                    o2_python
                )
            )

        elif o2_type == STRING:

            base_similarity = (
                string_number_similarity(
                    o2_python,
                    o1_python
                )
            )

        elif o2_type == DATE:

            base_similarity = (
                number_date_similarity(
                    o1_python,
                    o2_python
                )
            )

        else:

            base_similarity = 0.0

    # ========================================================
    # NUMBER on second side
    # ========================================================

    elif o2_type == NUMBER:

        if o1_type == STRING:

            base_similarity = (
                string_number_similarity(
                    o1_python,
                    o2_python
                )
            )

        elif o1_type == DATE:

            base_similarity = (
                number_date_similarity(
                    o2_python,
                    o1_python
                )
            )

        else:

            base_similarity = 0.0

    # ========================================================
    # DATE
    # ========================================================

    elif o1_type == DATE:

        if o2_type == DATE:

            base_similarity = (
                date_date_similarity(
                    o1_python,
                    o2_python
                )
            )

        elif o2_type == STRING:

            base_similarity = (
                string_date_similarity(
                    o2_python,
                    o1_python
                )
            )

        else:

            base_similarity = 0.0

    # ========================================================
    # DATE on second side
    # ========================================================

    elif o2_type == DATE:

        if o1_type == STRING:

            base_similarity = (
                string_date_similarity(
                    o1_python,
                    o2_python
                )
            )

        else:

            base_similarity = 0.0

    # ========================================================
    # STRING -> STRING
    # ========================================================

    elif (
        o1_type == STRING
        and
        o2_type == STRING
    ):

        base_similarity = (
            string_string_similarity(
                o1_python,
                o2_python
            )
        )

    # ========================================================
    # Unsupported combination
    # ========================================================

    else:

        base_similarity = 0.0

    # ========================================================
    # Apply Semantic Type-Aware Matching
    # ========================================================

    return apply_semantic_type_bonus(
        base_similarity,
        o1,
        o2,
        use_semantic_type_matching=use_semantic_type_matching,
        semantic_type_weight=semantic_type_weight,
        graph_1=graph_1,
        graph_2=graph_2,
        type_1=type_1,
        type_2=type_2,
    )
