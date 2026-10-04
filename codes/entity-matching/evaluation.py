"""
Evaluation and Metrics Calculation for Full-Triple-Matcher (FTM).

Implements:
1. Precision, Recall, F1-Score calculation against gold standard alignments.
2. Ranking metrics: Hit@1 and Hit@10.
3. Comparative reporting across Baseline, Improvement 2 (Adaptive Threshold),
   and Improvement 3 (Automated Conflict Resolution).
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd
import numpy as np


def evaluate_predictions(
    predictions_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    gold_col1: str = 'e1',
    gold_col2: str = 'e2',
    pred_col1: str = 'entity_kg1',
    pred_col2: str = 'entity_kg2',
    decision_col: str = 'decision',
    sim_col: str = 'similarity',
    one_to_one: bool = True
) -> Dict[str, Any]:
    """
    Evaluates binary matching decisions against gold standard mappings.
    Computes TP, FP, FN, Precision, Recall, F1-score, and Hit@k.
    """
    # 1. Parse gold standard pairs into set of tuples
    gold_pairs: Set[Tuple[str, str]] = set()
    for _, row in gold_df.iterrows():
        g1 = str(row[gold_col1]).strip()
        g2 = str(row[gold_col2]).strip()
        gold_pairs.add((g1, g2))

    total_gold = len(gold_pairs)

    # 2. Filter accepted predictions
    if decision_col in predictions_df.columns:
        accepted_df = predictions_df[predictions_df[decision_col] == True].copy()
    else:
        accepted_df = predictions_df.copy()

    # 3. Enforce 1-to-1 if requested (standard OAEI evaluation)
    if one_to_one and not accepted_df.empty and sim_col in accepted_df.columns:
        accepted_df = accepted_df.sort_values(by=sim_col, ascending=False)
        accepted_df = accepted_df.drop_duplicates(subset=[pred_col1], keep='first')
        accepted_df = accepted_df.drop_duplicates(subset=[pred_col2], keep='first')

    predicted_pairs: Set[Tuple[str, str]] = set()
    for _, row in accepted_df.iterrows():
        p1 = str(row[pred_col1]).strip()
        p2 = str(row[pred_col2]).strip()
        predicted_pairs.add((p1, p2))

    # 4. Compute TP, FP, FN
    tp_pairs = predicted_pairs.intersection(gold_pairs)
    true_positives = len(tp_pairs)
    false_positives = len(predicted_pairs) - true_positives
    false_negatives = total_gold - true_positives

    precision = true_positives / len(predicted_pairs) if len(predicted_pairs) > 0 else 0.0
    recall = true_positives / total_gold if total_gold > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 5. Compute Hit@1 and Hit@10
    hit_at_1, hit_at_10 = compute_hit_at_k(
        predictions_df=predictions_df,
        gold_pairs=gold_pairs,
        entity_col1=pred_col1,
        entity_col2=pred_col2,
        sim_col=sim_col
    )

    return {
        'total_gold': int(total_gold),
        'predicted_matches': int(len(predicted_pairs)),
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'precision': round(precision, 6),
        'recall': round(recall, 6),
        'f1': round(f1, 6),
        'hit_at_1': round(hit_at_1, 6),
        'hit_at_10': round(hit_at_10, 6)
    }


def compute_hit_at_k(
    predictions_df: pd.DataFrame,
    gold_pairs: Set[Tuple[str, str]],
    entity_col1: str = 'entity_kg1',
    entity_col2: str = 'entity_kg2',
    sim_col: str = 'similarity'
) -> Tuple[float, float]:
    """
    Computes Hit@1 and Hit@10 ranking metrics for source entities.
    """
    if predictions_df.empty or not gold_pairs:
        return 0.0, 0.0

    # Group candidate targets by e1 sorted by descending similarity (top 10 per entity)
    sorted_df = predictions_df.sort_values(by=[entity_col1, sim_col], ascending=[True, False])
    top10_df = sorted_df.groupby(entity_col1).head(10)
    grouped = top10_df.groupby(entity_col1)[entity_col2].apply(list).to_dict()

    gold_dict = {}
    for g1, g2 in gold_pairs:
        gold_dict[g1] = g2

    evaluable_entities = [e1 for e1 in gold_dict if e1 in grouped]
    if not evaluable_entities:
        return 0.0, 0.0

    hit1_count = 0
    hit10_count = 0

    for e1 in evaluable_entities:
        true_target = gold_dict[e1]
        ranked_targets = grouped[e1]

        if len(ranked_targets) > 0 and ranked_targets[0] == true_target:
            hit1_count += 1
        if true_target in ranked_targets[:10]:
            hit10_count += 1

    total = len(evaluable_entities)
    return hit1_count / total, hit10_count / total


def build_comparison_dataframe(
    baseline_metrics: Dict[str, Any],
    imp2_metrics: Optional[Dict[str, Any]] = None,
    imp3_metrics: Optional[Dict[str, Any]] = None,
    imp1_metrics: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Builds a structured comparison DataFrame between Baseline, Improvement 1 (Semantic Type Matching),
    Improvement 2 (Adaptive Threshold), and Improvement 3 (Automated Conflict Resolution).
    """
    keys = [
        ('predicted_matches', 'Predicted Matches'),
        ('true_positives', 'True Positives'),
        ('false_positives', 'False Positives'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1-Score'),
        ('hit_at_1', 'Hit@1'),
        ('hit_at_10', 'Hit@10'),
        ('runtime_seconds', 'Runtime (s)')
    ]

    imp2 = imp2_metrics or {}
    imp3 = imp3_metrics or {}
    imp1 = imp1_metrics or {}

    rows = []
    for k, label in keys:
        b_val = baseline_metrics.get(k, 0.0)
        i1_val = imp1.get(k, 0.0) if imp1_metrics is not None else None
        i2_val = imp2.get(k, 0.0)
        i3_val = imp3.get(k, 0.0)

        row_dict = {
            'metric': label,
            'baseline': round(b_val, 6) if isinstance(b_val, float) else b_val,
        }

        if imp1_metrics is not None:
            row_dict['improvement_1_semantic'] = round(i1_val, 6) if isinstance(i1_val, float) else i1_val

        row_dict['improvement_2_adaptive'] = round(i2_val, 6) if isinstance(i2_val, float) else i2_val
        row_dict['improvement_3_conflict_res'] = round(i3_val, 6) if isinstance(i3_val, float) else i3_val

        diff_base = i3_val - b_val if isinstance(i3_val, (int, float)) and isinstance(b_val, (int, float)) else 0.0
        diff_i2 = i3_val - i2_val if isinstance(i3_val, (int, float)) and isinstance(i2_val, (int, float)) else 0.0

        row_dict['diff_imp3_minus_baseline'] = round(diff_base, 6) if isinstance(diff_base, float) else diff_base
        row_dict['diff_imp3_minus_imp2'] = round(diff_i2, 6) if isinstance(diff_i2, float) else diff_i2

        rows.append(row_dict)

    return pd.DataFrame(rows)
