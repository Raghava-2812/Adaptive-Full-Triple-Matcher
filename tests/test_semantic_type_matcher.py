import sys
from pathlib import Path

from rdflib import Graph, URIRef

ROOT = Path(__file__).resolve().parents[1]
ENTITY_MATCHING = ROOT / "codes" / "entity-matching"
sys.path.insert(0, str(ENTITY_MATCHING))

import graph_similarity  # noqa: E402
import semantic_type_matcher as stm  # noqa: E402


def _small_graphs():
    kg1 = Graph()
    kg2 = Graph()
    kg1.parse(ROOT / "smallDatasets" / "small_movie_kg1.ttl")
    kg2.parse(ROOT / "smallDatasets" / "small_movie_kg2.ttl")
    return kg1, kg2


def test_context_infers_movie_person_and_genre():
    kg1, _ = _small_graphs()

    assert (
        stm.detect_semantic_category(
            URIRef("http://kg1.example.org/resource/Inception"),
            graph=kg1,
        )
        == stm.TYPE_MOVIE
    )
    assert (
        stm.detect_semantic_category(
            URIRef("http://kg1.example.org/resource/Christopher_Nolan"),
            graph=kg1,
        )
        == stm.TYPE_PERSON
    )
    assert (
        stm.detect_semantic_category(
            URIRef("http://kg1.example.org/resource/Science_Fiction"),
            graph=kg1,
        )
        == stm.TYPE_GENRE
    )


def test_semantic_compatibility_uses_graph_context():
    kg1, kg2 = _small_graphs()

    score = stm.get_semantic_type_bonus(
        URIRef("http://kg1.example.org/resource/Inception"),
        URIRef("http://kg2.example.org/resource/Inception_2010"),
        graph_a=kg1,
        graph_b=kg2,
    )

    assert score == 1.0


def test_graph_similarity_gets_semantic_context_bonus():
    kg1, kg2 = _small_graphs()
    left = URIRef("http://kg1.example.org/resource/Inception")
    right = URIRef("http://kg2.example.org/resource/Inception_2010")

    baseline = graph_similarity.get_objects_similarity(
        left,
        right,
        use_semantic_type_matching=False,
    )
    improved = graph_similarity.get_objects_similarity(
        left,
        right,
        graph_1=kg1,
        graph_2=kg2,
        use_semantic_type_matching=True,
    )

    assert improved > baseline
    assert improved <= 1.0
