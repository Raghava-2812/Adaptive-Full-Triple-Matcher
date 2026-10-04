"""
Preprocessing Utilities for Full-Triple-Matcher (FTM)
Provides text cleaning, URI normalization, tokenization, and RDF literal handling.
"""

import re
import string
import datetime
from typing import Optional, List, Any, Union
from rdflib import URIRef, Literal
import dateutil.parser as dparser

# Common RDF predicates used for labeling
LABEL_PREDICATES = [
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://schema.org/name",
    "http://xmlns.com/foaf/0.1/name",
    "http://kg1.example.org/ontology/title",
    "http://kg1.example.org/ontology/name",
    "http://kg2.example.org/ontology/film_title",
    "http://kg2.example.org/ontology/full_name",
    "http://complex-kg1.org/ontology/title",
    "http://complex-kg1.org/ontology/name",
    "http://complex-kg2.org/ontology/filmTitle",
    "http://complex-kg2.org/ontology/fullName",
    "http://complex-kg2.org/ontology/name"
]


def clean_text(text: Optional[str]) -> str:
    """
    Standard text normalization: lowercasing, stripping excessive spaces,
    and removing non-alphanumeric punctuation.
    """
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r'[\s_]+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def extract_local_name(uri: Optional[str]) -> str:
    """
    Extracts the human-readable entity name or identifier from a URI.
    Splits on '#' or the last '/' and replaces underscores with spaces.
    
    Examples:
      - 'http://dbpedia.org/resource/Inception' -> 'Inception'
      - 'http://xmlns.com/foaf/0.1/name' -> 'name'
      - 'http://www.wikidata.org/entity/Q251362' -> 'Q251362'
    """
    if not uri:
        return ""
    uri_str = str(uri).strip()
    if '#' in uri_str:
        local = uri_str.rsplit('#', 1)[-1]
    else:
        local = uri_str.rsplit('/', 1)[-1]
    # Replace underscores with spaces for clean label extraction
    clean_local = local.replace('_', ' ').strip()
    return clean_local if clean_local else uri_str


def tokenize(text: Optional[str]) -> List[str]:
    """
    Splits text into lowercased alphanumeric tokens.
    """
    if not text:
        return []
    cleaned = clean_text(text)
    return [tok for tok in cleaned.split() if tok]


def parse_date_value(val: Any) -> Optional[datetime.datetime]:
    """
    Attempts to parse arbitrary date/time representation into datetime.
    """
    if val is None:
        return None
    if isinstance(val, (datetime.datetime, datetime.date)):
        if isinstance(val, datetime.date) and not isinstance(val, datetime.datetime):
            return datetime.datetime.combine(val, datetime.time.min)
        return val
    try:
        return dparser.parse(str(val), fuzzy=True)
    except Exception:
        return None


def get_timestamp(val: Any) -> float:
    """
    Converts datetime or date to unix timestamp float.
    """
    dt = parse_date_value(val)
    if dt:
        try:
            return dt.timestamp()
        except Exception:
            return 0.0
    return 0.0


def extract_entity_label_from_graph(graph, uri: Union[str, URIRef]) -> str:
    """
    Retrieves human-readable label for a URI from an RDF graph.
    Checks common label predicates before falling back to local name.
    """
    if not isinstance(uri, URIRef):
        uri_ref = URIRef(str(uri))
    else:
        uri_ref = uri

    if graph is not None:
        for pred in LABEL_PREDICATES:
            for _, _, o in graph.triples((uri_ref, URIRef(pred), None)):
                if o:
                    return str(o)
        for p, o in graph.predicate_objects(uri_ref):
            p_local = extract_local_name(str(p)).lower().replace('_', '')
            if p_local in ['label', 'preflabel', 'title', 'filmtitle', 'name', 'fullname']:
                if o:
                    return str(o)
    return extract_local_name(str(uri))
