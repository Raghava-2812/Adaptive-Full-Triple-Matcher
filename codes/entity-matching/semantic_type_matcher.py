"""
Semantic Type-Aware Matching for Full Triple Matcher (FTM).

This module does NOT replace the original FTM similarity.
It provides an additional semantic compatibility signal.

Priority of type detection:

1. Explicit RDF type information
2. Supplied semantic type metadata
3. URI / label based inference
4. Unknown

The result is used as supporting evidence, not as a hard constraint.
"""

import re
from datetime import date, datetime
from numbers import Number

from rdflib import URIRef, Literal
from rdflib.namespace import RDF, RDFS


# ============================================================
# Configuration
# ============================================================

# Keep False when reproducing the original FTM baseline.
# Set True for the first A-FTM improvement.
USE_SEMANTIC_TYPE_MATCHING = True

# Initial experimental weight.
# This should be tuned experimentally later.
SEMANTIC_TYPE_WEIGHT = 0.10


# ============================================================
# Semantic categories
# ============================================================

TYPE_UNKNOWN = "unknown"
TYPE_ENTITY = "entity"

TYPE_PERSON = "person"
TYPE_ORGANIZATION = "organization"
TYPE_LOCATION = "location"
TYPE_COUNTRY = "country"
TYPE_NATIONALITY = "nationality"

TYPE_MOVIE = "movie"
TYPE_BOOK = "book"
TYPE_SONG = "song"
TYPE_ALBUM = "album"
TYPE_SERIES = "series"
TYPE_GENRE = "genre"

TYPE_NUMBER = "number"
TYPE_DATE = "date"
TYPE_STRING = "string"


# ============================================================
# Keyword dictionaries
# ============================================================

PERSON_KEYWORDS = {
    "person",
    "people",
    "human",
    "actor",
    "actress",
    "author",
    "writer",
    "artist",
    "director",
    "producer",
    "musician",
    "scientist",
    "teacher",
    "student",
    "engineer",
    "athlete",
    "player",
    "politician",
    "king",
    "queen",
    "character",
}

ORGANIZATION_KEYWORDS = {
    "organization",
    "organisation",
    "company",
    "corporation",
    "team",
    "club",
    "school",
    "university",
    "college",
    "institute",
    "agency",
    "department",
    "society",
    "association",
    "foundation",
}

LOCATION_KEYWORDS = {
    "location",
    "place",
    "city",
    "town",
    "village",
    "state",
    "province",
    "region",
    "district",
    "capital",
    "mountain",
    "river",
    "island",
    "continent",
    "park",
    "country",
}

MOVIE_KEYWORDS = {
    "movie",
    "film",
}

BOOK_KEYWORDS = {
    "book",
    "novel",
}

SONG_KEYWORDS = {
    "song",
    "track",
}

ALBUM_KEYWORDS = {
    "album",
}

SERIES_KEYWORDS = {
    "series",
    "show",
    "episode",
}

GENRE_KEYWORDS = {
    "genre",
    "category",
    "action",
    "drama",
    "romance",
    "romantic",
    "thriller",
    "scifi",
    "sci fi",
    "science fiction",
    "film",
}

MOVIE_CONTEXT_PREDICATES = {
    "title",
    "name",
    "director",
    "directed by",
    "release year",
    "year",
    "runtime",
    "duration minutes",
    "country",
    "origin country",
    "genre",
    "category",
}

PERSON_CONTEXT_PREDICATES = {
    "director",
    "directed by",
    "actor",
    "cast",
    "producer",
    "writer",
}

GENRE_CONTEXT_PREDICATES = {
    "genre",
    "category",
}


# ============================================================
# Common aliases
# ============================================================

NATIONALITY_ALIASES = {
    "american",
    "british",
    "french",
    "german",
    "indian",
    "japanese",
    "chinese",
    "italian",
    "spanish",
    "russian",
    "korean",
    "arab",
    "canadian",
    "mexican",
    "brazilian",
    "australian",
    "irish",
    "swedish",
    "dutch",
    "turkish",
    "norwegian",
    "danish",
    "finnish",
    "polish",
    "ukrainian",
}

COUNTRY_ALIASES = {
    "united states",
    "usa",
    "us",
    "united states of america",
    "america",
    "england",
    "uk",
    "united kingdom",
    "great britain",
    "france",
    "germany",
    "india",
    "japan",
    "china",
    "italy",
    "spain",
    "russia",
    "korea",
    "saudi arabia",
    "canada",
    "mexico",
    "brazil",
    "australia",
    "ireland",
    "sweden",
    "netherlands",
    "turkey",
    "norway",
    "denmark",
    "finland",
    "poland",
    "ukraine",
}


# ============================================================
# Text normalization
# ============================================================

def normalize_text(value):
    """
    Convert a value into normalized human-readable text.
    """

    if value is None:
        return ""

    text = str(value).strip()

    # URI separators
    text = re.sub(r"[_\-]+", " ", text)

    # CamelCase -> Camel Case
    text = re.sub(
        r"(?<=[a-z])(?=[A-Z])",
        " ",
        text
    )

    # Remove repeated spaces
    text = re.sub(r"\s+", " ", text)

    return text.lower().strip()


# ============================================================
# URI extraction
# ============================================================

def get_uri_local_name(uri):
    """
    Extract the meaningful part from a URI.

    Example:

    http://example.org/UnitedStates
    ->
    UnitedStates
    """

    if uri is None:
        return ""

    text = str(uri)

    # Handle both / and #
    text = re.split(r"[#/]", text)[-1]

    # Remove fragment artifacts
    text = text.replace("%20", " ")

    return normalize_text(text)


# ============================================================
# Category inference from text
# ============================================================

def infer_category_from_text(value):
    """
    Infer a semantic category from a label or URI local name.

    This is a fallback mechanism only.
    """

    text = normalize_text(value)

    if not text:
        return TYPE_UNKNOWN

    # -----------------------------------------
    # Explicit aliases
    # -----------------------------------------

    if text in NATIONALITY_ALIASES:
        return TYPE_NATIONALITY

    if text in COUNTRY_ALIASES:
        return TYPE_COUNTRY

    # -----------------------------------------
    # Number
    # -----------------------------------------

    if re.fullmatch(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
        text
    ):
        return TYPE_NUMBER

    # -----------------------------------------
    # Person
    # -----------------------------------------

    tokens = set(text.split())

    if tokens & PERSON_KEYWORDS:
        return TYPE_PERSON

    # -----------------------------------------
    # Organization
    # -----------------------------------------

    if tokens & ORGANIZATION_KEYWORDS:
        return TYPE_ORGANIZATION

    # -----------------------------------------
    # Location
    # -----------------------------------------

    if tokens & LOCATION_KEYWORDS:
        return TYPE_LOCATION

    # -----------------------------------------
    # Media
    # -----------------------------------------

    if tokens & MOVIE_KEYWORDS:
        return TYPE_MOVIE

    if tokens & BOOK_KEYWORDS:
        return TYPE_BOOK

    if tokens & SONG_KEYWORDS:
        return TYPE_SONG

    if tokens & ALBUM_KEYWORDS:
        return TYPE_ALBUM

    if tokens & SERIES_KEYWORDS:
        return TYPE_SERIES

    if text in GENRE_KEYWORDS or tokens & GENRE_KEYWORDS:
        return TYPE_GENRE

    # -----------------------------------------
    # Nationality suffixes
    # -----------------------------------------

    if (
        text.endswith("ian")
        or text.endswith("ese")
        or text.endswith("ish")
    ):
        return TYPE_NATIONALITY

    return TYPE_ENTITY


# ============================================================
# RDF type inference
# ============================================================

def infer_category_from_rdf_type(rdf_type):
    """
    Convert an RDF class URI into one of our semantic categories.

    Example:

    http://dbpedia.org/ontology/Person
    ->
    person
    """

    if rdf_type is None:
        return TYPE_UNKNOWN

    local_name = get_uri_local_name(rdf_type)

    if not local_name:
        return TYPE_UNKNOWN

    tokens = set(local_name.split())

    if local_name in {
        "person",
        "human",
        "actor",
        "actress",
        "artist",
        "writer",
        "author",
    }:
        return TYPE_PERSON

    if local_name in {
        "organization",
        "organisation",
        "company",
        "corporation",
        "institution",
        "university",
        "school",
    }:
        return TYPE_ORGANIZATION

    if local_name in {
        "country",
        "nation",
    }:
        return TYPE_COUNTRY

    if local_name in {
        "city",
        "town",
        "village",
        "place",
        "location",
        "state",
        "province",
        "region",
        "island",
        "mountain",
        "river",
    }:
        return TYPE_LOCATION

    if local_name in {
        "movie",
        "film",
    }:
        return TYPE_MOVIE

    if local_name in {
        "book",
        "novel",
    }:
        return TYPE_BOOK

    if local_name in {
        "song",
        "track",
    }:
        return TYPE_SONG

    if local_name == "album":
        return TYPE_ALBUM

    if local_name in {
        "series",
        "tvseries",
        "televisionseries",
    }:
        return TYPE_SERIES

    if local_name in {
        "genre",
        "category",
        "filmgenre",
    }:
        return TYPE_GENRE

    return TYPE_UNKNOWN


# ============================================================
# Context-aware type inference
# ============================================================

def _predicate_context_name(predicate):
    """
    Normalize a predicate URI local name for context rules.
    """

    return get_uri_local_name(predicate)


def get_literal_label(value, graph=None):
    """
    Try to obtain a readable label for a URIRef.
    """

    if graph is None or not isinstance(value, URIRef):
        return ""

    label_predicates = (
        RDFS.label,
        URIRef("http://kg1.example.org/ontology/title"),
        URIRef("http://kg1.example.org/ontology/name"),
        URIRef("http://kg2.example.org/ontology/name"),
        URIRef("http://kg2.example.org/ontology/fullName"),
    )

    try:
        for predicate in label_predicates:
            for label in graph.objects(value, predicate):
                return normalize_text(label)
    except Exception:
        return ""

    return ""


def infer_category_from_context(value, graph=None, max_terms=50):
    """
    Infer type from neighboring triples.

    This is the first A-FTM improvement: the matcher considers what an entity
    is connected to, not only how its URI or label is written.
    """

    if graph is None or not isinstance(value, URIRef):
        return TYPE_UNKNOWN

    outgoing_predicates = set()
    incoming_predicates = set()
    labels = set()

    try:
        for predicate, obj in graph.predicate_objects(value):
            outgoing_predicates.add(_predicate_context_name(predicate))

            if predicate == RDFS.label:
                labels.add(normalize_text(obj))

            if len(outgoing_predicates) + len(labels) >= max_terms:
                break

        for subj, predicate in graph.subject_predicates(value):
            incoming_predicates.add(_predicate_context_name(predicate))

            if len(incoming_predicates) >= max_terms:
                break

    except Exception:
        return TYPE_UNKNOWN

    label_text = get_literal_label(value, graph=graph)
    if label_text:
        labels.add(label_text)

    label_category_votes = [
        infer_category_from_text(label)
        for label in labels
        if label
    ]

    for category in label_category_votes:
        if category not in {TYPE_UNKNOWN, TYPE_ENTITY}:
            return category

    # Movie entities in the small KGs are described by several movie-fact
    # predicates. Requiring at least two signals avoids treating a person with
    # only a name as a movie.
    movie_signals = outgoing_predicates & MOVIE_CONTEXT_PREDICATES
    if len(movie_signals) >= 2:
        return TYPE_MOVIE

    if incoming_predicates & PERSON_CONTEXT_PREDICATES:
        return TYPE_PERSON

    if incoming_predicates & GENRE_CONTEXT_PREDICATES:
        return TYPE_GENRE

    if outgoing_predicates == {"name"} or outgoing_predicates == {"full name"}:
        return TYPE_PERSON

    return TYPE_UNKNOWN


# ============================================================
# Explicit RDF graph lookup
# ============================================================

def get_rdf_semantic_type(value, graph=None):
    """
    Try to obtain an explicit RDF semantic type.

    Example RDF:

        :Raj rdf:type :Person .

    If graph contains this triple, the result is 'person'.

    Returns TYPE_UNKNOWN if no graph/type information is available.
    """

    if graph is None:
        return TYPE_UNKNOWN

    if not isinstance(value, URIRef):
        return TYPE_UNKNOWN

    try:
        rdf_types = graph.objects(value, RDF.type)

        for rdf_type in rdf_types:
            category = infer_category_from_rdf_type(rdf_type)

            if category != TYPE_UNKNOWN:
                return category

    except Exception:
        return TYPE_UNKNOWN

    return TYPE_UNKNOWN


# ============================================================
# Main semantic type detection
# ============================================================

def detect_semantic_category(value, graph=None, explicit_type=None):
    """
    Determine the semantic category of an object.

    Priority:

        explicit_type
            ↓
        RDF graph type
            ↓
        Python datatype
            ↓
        literal/URI label inference
            ↓
        unknown
    """

    # -----------------------------------------
    # 1. Explicit caller-provided type
    # -----------------------------------------

    if explicit_type:
        normalized_type = normalize_text(explicit_type)

        if normalized_type:
            return normalized_type

    # -----------------------------------------
    # None
    # -----------------------------------------

    if value is None:
        return TYPE_UNKNOWN

    # -----------------------------------------
    # 2. RDF type
    # -----------------------------------------

    rdf_category = get_rdf_semantic_type(
        value,
        graph=graph
    )

    if rdf_category != TYPE_UNKNOWN:
        return rdf_category

    # -----------------------------------------
    # 3. Graph neighborhood context
    # -----------------------------------------

    context_category = infer_category_from_context(
        value,
        graph=graph
    )

    if context_category != TYPE_UNKNOWN:
        return context_category

    # -----------------------------------------
    # 4. RDFLib Literal
    # -----------------------------------------

    if isinstance(value, Literal):

        try:
            native_value = value.toPython()
        except Exception:
            native_value = value

        if isinstance(native_value, bool):
            return TYPE_ENTITY

        if isinstance(native_value, Number):
            return TYPE_NUMBER

        if isinstance(native_value, (date, datetime)):
            return TYPE_DATE

        return infer_category_from_text(native_value)

    # -----------------------------------------
    # 5. URIRef
    # -----------------------------------------

    if isinstance(value, URIRef):

        return infer_category_from_text(
            get_uri_local_name(value)
        )

    # -----------------------------------------
    # 6. Normal Python values
    # -----------------------------------------

    if isinstance(value, bool):
        return TYPE_ENTITY

    if isinstance(value, Number):
        return TYPE_NUMBER

    if isinstance(value, (date, datetime)):
        return TYPE_DATE

    # -----------------------------------------
    # 7. String fallback
    # -----------------------------------------

    return infer_category_from_text(value)


# ============================================================
# Semantic compatibility
# ============================================================

def semantic_type_compatibility(
    value_a,
    value_b,
    graph_a=None,
    graph_b=None,
    type_a=None,
    type_b=None,
):
    """
    Calculate semantic compatibility between two objects.

    Range:

        0.0 -> incompatible / unknown
        1.0 -> same semantic category

    This does NOT mean the two objects are equivalent.
    """

    category_a = detect_semantic_category(
        value_a,
        graph=graph_a,
        explicit_type=type_a,
    )

    category_b = detect_semantic_category(
        value_b,
        graph=graph_b,
        explicit_type=type_b,
    )

    # Unknown should not create a bonus.
    if (
        category_a == TYPE_UNKNOWN
        or category_b == TYPE_UNKNOWN
    ):
        return 0.0

    # Same strong semantic category.
    if category_a == category_b:

        # Generic entity is intentionally NOT treated
        # as strong semantic evidence.
        if category_a == TYPE_ENTITY:
            return 0.0

        return 1.0

    # -----------------------------------------
    # Related semantic categories
    # -----------------------------------------

    compatible_pairs = {

        # Country <-> Nationality
        (TYPE_COUNTRY, TYPE_NATIONALITY): 0.90,
        (TYPE_NATIONALITY, TYPE_COUNTRY): 0.90,

        # Country <-> Location
        (TYPE_COUNTRY, TYPE_LOCATION): 0.75,
        (TYPE_LOCATION, TYPE_COUNTRY): 0.75,

        # Person <-> generic entity
        (TYPE_PERSON, TYPE_ENTITY): 0.25,
        (TYPE_ENTITY, TYPE_PERSON): 0.25,

        # Organization <-> generic entity
        (TYPE_ORGANIZATION, TYPE_ENTITY): 0.25,
        (TYPE_ENTITY, TYPE_ORGANIZATION): 0.25,

        # Location <-> generic entity
        (TYPE_LOCATION, TYPE_ENTITY): 0.25,
        (TYPE_ENTITY, TYPE_LOCATION): 0.25,

        # Genre aliases can be modeled as entities or categories.
        (TYPE_GENRE, TYPE_ENTITY): 0.25,
        (TYPE_ENTITY, TYPE_GENRE): 0.25,
    }

    return compatible_pairs.get(
        (category_a, category_b),
        0.0
    )


# ============================================================
# Public API
# ============================================================

def get_semantic_type_bonus(
    value_a,
    value_b,
    graph_a=None,
    graph_b=None,
    type_a=None,
    type_b=None,
):
    """
    Public function used by graph_similarity.py.
    """

    return semantic_type_compatibility(
        value_a=value_a,
        value_b=value_b,
        graph_a=graph_a,
        graph_b=graph_b,
        type_a=type_a,
        type_b=type_b,
    )
