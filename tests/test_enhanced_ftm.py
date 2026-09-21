import sys
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import RDF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))

from enhanced_ftm import (  # noqa: E402
    ConflictCandidate,
    ThresholdContext,
    adaptive_threshold,
    infer_type,
    precision_recall_f1,
    resolve_conflict,
    score_candidate,
)


def test_semantic_type_uses_graph_context():
    graph = Graph()
    entity = URIRef("http://example.org/Raj")
    graph.add((entity, RDF.type, URIRef("http://example.org/Person")))

    evidence = infer_type(entity, graph=graph)

    assert evidence.category == "person"
    assert evidence.confidence > 0


def test_adaptive_threshold_drops_with_semantic_evidence():
    plain = adaptive_threshold(ThresholdContext(scores=(0.91, 0.89), semantic_score=0.0))
    semantic = adaptive_threshold(ThresholdContext(scores=(0.91, 0.89), semantic_score=1.0))

    assert semantic < plain


def test_workflow_adds_semantic_bonus_and_decides():
    decision = score_candidate(
        0.86,
        URIRef("http://example.org/France"),
        URIRef("http://example.org/French"),
        competing_scores=(0.86, 0.42),
        functionality=0.95,
        inverse_functionality=0.95,
    )

    assert decision.semantic_score > 0
    assert decision.final_score > decision.base_score


def test_conflict_resolution_auto_resolves_clear_winner():
    result = resolve_conflict(
        [
            ConflictCandidate("a", "x", similarity=0.98, semantic_score=1.0, threshold=0.90, evidence_count=4),
            ConflictCandidate("a", "y", similarity=0.91, semantic_score=0.2, threshold=0.90, evidence_count=1),
        ]
    )

    assert result.status == "AUTO_RESOLVED"
    assert result.selected is not None
    assert result.selected.right_id == "x"


def test_precision_recall_f1():
    metrics = precision_recall_f1(
        predicted=[("a", "x"), ("b", "y")],
        gold=[("a", "x"), ("c", "z")],
    )

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
