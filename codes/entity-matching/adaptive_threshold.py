"""
Adaptive Threshold Optimization for Full-Triple-Matcher (FTM)
Improvement 2: Candidate-dependent decision thresholding based on uncertainty penalty and confidence bonus.

This module is designed to replace static global thresholding with an explainable,
scientifically defensible adaptive threshold calculated from signals available
during the FTM matching pipeline (candidate similarity, candidate score margin,
and structural triple evidence).

Author: CSE Project - Full-Triple-Matcher Enhancements
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

try:
    from gpu_utils import gpu_accelerate_adaptive_threshold, gpu_compute_candidate_margins, is_cuda_available, get_gpu_info
except ImportError:
    try:
        from codes.utils.gpu_utils import gpu_accelerate_adaptive_threshold, gpu_compute_candidate_margins, is_cuda_available, get_gpu_info
    except ImportError:
        gpu_accelerate_adaptive_threshold = None
        gpu_compute_candidate_margins = None
        is_cuda_available = lambda: False
        get_gpu_info = lambda: {"cuda_available": False}


try:
    import semantic_type_matching as stm
except ImportError:
    try:
        from codes.entity_matching import semantic_type_matching as stm
    except ImportError:
        stm = None


# ==============================================================================
# CONFIGURATION SECTION
# ==============================================================================
# Setting USE_SEMANTIC_TYPE_MATCHING = True activates Improvement 1 (Semantic Type Matching)
USE_SEMANTIC_TYPE_MATCHING: bool = True

# Setting USE_ADAPTIVE_THRESHOLD = False strictly preserves original FTM baseline behavior.
# Setting USE_ADAPTIVE_THRESHOLD = True activates Improvement 2.
USE_ADAPTIVE_THRESHOLD: bool = False

# Setting USE_CONFLICT_RESOLUTION = True activates Improvement 3 (requires Improvement 2).
USE_CONFLICT_RESOLUTION: bool = False

# Default threshold parameters (aligned with standard FTM literature)
BASE_THRESHOLD: float = 0.80           # Base acceptance threshold (fixed baseline cut-off)
MIN_ADAPTIVE_THRESHOLD: float = 0.50   # Clamping lower bound
MAX_ADAPTIVE_THRESHOLD: float = 0.95   # Clamping upper bound

# Parameter weights
UNCERTAINTY_WEIGHT: float = 0.10       # Weight for margin-based uncertainty penalty
CONFIDENCE_WEIGHT: float = 0.10        # Weight for similarity score confidence bonus
MARGIN_WEIGHT: float = 0.10            # Weight for score margin confidence bonus
TRIPLE_WEIGHT: float = 0.05            # Weight for triple agreement confidence bonus
SEMANTIC_BONUS_WEIGHT: float = 0.10     # Weight for semantic type agreement bonus (Improvement 1)


class AdaptiveThresholdConfig:
    """
    Configuration holder for Adaptive Threshold Optimization parameters.
    """
    def __init__(
        self,
        use_adaptive_threshold: bool = USE_ADAPTIVE_THRESHOLD,
        use_semantic_type_matching: bool = USE_SEMANTIC_TYPE_MATCHING,
        base_threshold: float = BASE_THRESHOLD,
        min_adaptive_threshold: float = MIN_ADAPTIVE_THRESHOLD,
        max_adaptive_threshold: float = MAX_ADAPTIVE_THRESHOLD,
        uncertainty_weight: float = UNCERTAINTY_WEIGHT,
        confidence_weight: float = CONFIDENCE_WEIGHT,
        margin_weight: float = MARGIN_WEIGHT,
        triple_weight: float = TRIPLE_WEIGHT,
        semantic_bonus_weight: float = SEMANTIC_BONUS_WEIGHT
    ):
        self.use_adaptive_threshold = bool(use_adaptive_threshold)
        self.use_semantic_type_matching = bool(use_semantic_type_matching)
        self.base_threshold = float(base_threshold)
        self.min_adaptive_threshold = float(min_adaptive_threshold)
        self.max_adaptive_threshold = float(max_adaptive_threshold)
        self.uncertainty_weight = float(uncertainty_weight)
        self.confidence_weight = float(confidence_weight)
        self.margin_weight = float(margin_weight)
        self.triple_weight = float(triple_weight)
        self.semantic_bonus_weight = float(semantic_bonus_weight)

    def to_dict(self) -> Dict[str, Union[bool, float]]:
        return {
            "USE_ADAPTIVE_THRESHOLD": self.use_adaptive_threshold,
            "USE_SEMANTIC_TYPE_MATCHING": self.use_semantic_type_matching,
            "BASE_THRESHOLD": self.base_threshold,
            "MIN_ADAPTIVE_THRESHOLD": self.min_adaptive_threshold,
            "MAX_ADAPTIVE_THRESHOLD": self.max_adaptive_threshold,
            "UNCERTAINTY_WEIGHT": self.uncertainty_weight,
            "CONFIDENCE_WEIGHT": self.confidence_weight,
            "MARGIN_WEIGHT": self.margin_weight,
            "TRIPLE_WEIGHT": self.triple_weight,
            "SEMANTIC_BONUS_WEIGHT": self.semantic_bonus_weight,
        }

    def __repr__(self) -> str:
        return (
            f"AdaptiveThresholdConfig("
            f"use_adaptive={self.use_adaptive_threshold}, "
            f"use_semantic={self.use_semantic_type_matching}, "
            f"base_threshold={self.base_threshold}, "
            f"bounds=[{self.min_adaptive_threshold}, {self.max_adaptive_threshold}], "
            f"weights=[unc={self.uncertainty_weight}, conf={self.confidence_weight}, "
            f"margin={self.margin_weight}, triple={self.triple_weight}, sem={self.semantic_bonus_weight}])"
        )


# ==============================================================================
# CORE MATHEMATICAL FUNCTIONS
# ==============================================================================

def calculate_candidate_margin(best_score: float, second_best_score: float) -> float:
    """
    Calculate the score margin between the top-ranked candidate and runner-up candidate.

    Mathematical formulation:
        margin = max(0.0, min(1.0, best_score - second_best_score))

    A large margin indicates a clear, dominant match (high confidence).
    A small margin indicates competing candidates with similar scores (high ambiguity).

    Parameters:
        best_score: Highest similarity score among candidates for entity e1.
        second_best_score: Second-highest similarity score (0.0 if only 1 candidate exists).

    Returns:
        Normalized margin in range [0.0, 1.0].
    """
    if best_score is None or math.isnan(best_score):
        best_score = 0.0
    if second_best_score is None or math.isnan(second_best_score):
        second_best_score = 0.0

    margin = float(best_score - second_best_score)
    # Ensure range [0.0, 1.0]
    return max(0.0, min(1.0, margin))


def calculate_uncertainty_penalty(
    margin: float,
    uncertainty_weight: float = UNCERTAINTY_WEIGHT
) -> float:
    """
    Calculate the uncertainty penalty based on candidate margin.

    Mathematical formulation:
        uncertainty_penalty = uncertainty_weight * (1.0 - margin)

    Logic:
        When margin -> 0 (ambiguous/uncertain competition):
            penalty -> uncertainty_weight (increases threshold, requiring higher evidence).
        When margin -> 1 (decisive separation):
            penalty -> 0 (no penalty imposed).

    Parameters:
        margin: Candidate score margin in [0.0, 1.0].
        uncertainty_weight: Scaling weight (default 0.10).

    Returns:
        Uncertainty penalty in range [0.0, uncertainty_weight].
    """
    norm_margin = max(0.0, min(1.0, float(margin)))
    penalty = float(uncertainty_weight) * (1.0 - norm_margin)
    return max(0.0, penalty)


def calculate_confidence_bonus(
    similarity: float,
    margin: float,
    confidence_weight: float = CONFIDENCE_WEIGHT,
    margin_weight: float = MARGIN_WEIGHT,
    triple_sim: float = 0.0,
    triple_weight: float = TRIPLE_WEIGHT,
    semantic_bonus: float = 0.0
) -> float:
    """
    Calculate confidence bonus based on intrinsic similarity, margin, triple agreement,
    and semantic type alignment.

    Mathematical formulation:
        confidence_bonus = (confidence_weight * similarity) +
                           (margin_weight * margin) +
                           (triple_weight * triple_sim) +
                           semantic_bonus

    Parameters:
        similarity: Candidate similarity score in [0.0, 1.0].
        margin: Candidate score margin in [0.0, 1.0].
        confidence_weight: Weight for similarity score (default 0.10).
        margin_weight: Weight for candidate margin (default 0.10).
        triple_sim: Structural triple similarity score in [0.0, 1.0] (default 0.0).
        triple_weight: Weight for triple agreement (default 0.05).
        semantic_bonus: Semantic type match bonus from Improvement 1 (default 0.0).

    Returns:
        Confidence bonus in range [0.0, 1.0].
    """
    norm_sim = max(0.0, min(1.0, float(similarity))) if similarity is not None else 0.0
    norm_margin = max(0.0, min(1.0, float(margin))) if margin is not None else 0.0
    norm_triple = max(0.0, min(1.0, float(triple_sim))) if triple_sim is not None else 0.0
    norm_sem = max(0.0, float(semantic_bonus)) if semantic_bonus is not None else 0.0

    bonus = (
        float(confidence_weight) * norm_sim +
        float(margin_weight) * norm_margin +
        float(triple_weight) * norm_triple +
        norm_sem
    )
    return max(0.0, bonus)


def calculate_adaptive_threshold(
    similarity: float,
    margin: float,
    base_threshold: float = BASE_THRESHOLD,
    uncertainty_weight: float = UNCERTAINTY_WEIGHT,
    confidence_weight: float = CONFIDENCE_WEIGHT,
    margin_weight: float = MARGIN_WEIGHT,
    min_threshold: float = MIN_ADAPTIVE_THRESHOLD,
    max_threshold: float = MAX_ADAPTIVE_THRESHOLD,
    triple_sim: float = 0.0,
    triple_weight: float = TRIPLE_WEIGHT,
    semantic_type_similarity: float = 0.0,
    semantic_type_match: bool = True,
    semantic_type_bonus: float = 0.0,
    use_adaptive: bool = True,
    use_semantic_type: bool = True
) -> Tuple[float, float, float]:
    """
    Compute adaptive threshold for a candidate match integrating Improvement 1 semantic evidence.

    Mathematical formulation:
        Adaptive Threshold = clamp(
            Base Threshold + Uncertainty Penalty - Confidence Bonus - Semantic Meaning Match Bonus,
            min_threshold,
            max_threshold
        )

    Parameters:
        similarity: Similarity score of candidate in [0.0, 1.0].
        margin: Candidate score margin in [0.0, 1.0].
        base_threshold: Base threshold (default 0.80).
        uncertainty_weight: Weight for uncertainty penalty.
        confidence_weight: Weight for similarity confidence bonus.
        margin_weight: Weight for margin confidence bonus.
        min_threshold: Minimum allowable threshold (lower clamp).
        max_threshold: Maximum allowable threshold (upper clamp).
        triple_sim: Optional triple-level similarity in [0.0, 1.0].
        triple_weight: Weight for triple agreement.
        semantic_type_similarity: Semantic category similarity from Improvement 1.
        semantic_type_match: Whether candidate types are compatible.
        semantic_type_bonus: Bonus value computed by Improvement 1.
        use_adaptive: Boolean flag. If False, returns base_threshold.
        use_semantic_type: Boolean flag for Improvement 1 integration.

    Returns:
        Tuple of (adaptive_threshold, uncertainty_penalty, confidence_bonus).
    """
    if not use_adaptive:
        return float(base_threshold), 0.0, 0.0

    penalty = calculate_uncertainty_penalty(margin, uncertainty_weight)

    # If types are conflicting, impose incompatibility penalty to prevent lowering threshold
    if use_semantic_type and not semantic_type_match and semantic_type_similarity < 0.25:
        penalty += float(uncertainty_weight) * 0.5

    sem_b = float(semantic_type_bonus) if (use_semantic_type and semantic_type_match) else 0.0

    bonus = calculate_confidence_bonus(
        similarity=similarity,
        margin=margin,
        confidence_weight=confidence_weight,
        margin_weight=margin_weight,
        triple_sim=triple_sim,
        triple_weight=triple_weight,
        semantic_bonus=sem_b
    )

    raw_threshold = float(base_threshold) + penalty - bonus

    # Clamping between configured bounds
    clamped_threshold = max(float(min_threshold), min(float(max_threshold), raw_threshold))

    return clamped_threshold, penalty, bonus


def decide_match(
    similarity: float,
    threshold: float,
    semantic_match: bool = True,
    enforce_semantic_filter: bool = False
) -> bool:
    """
    Determine matching decision by comparing similarity against the threshold
    and checking semantic category compatibility when required.

    Decision:
        MATCH (True)    if similarity >= threshold and (semantic_match if enforce_semantic_filter)
        NO MATCH (False) otherwise
    """
    if similarity is None or math.isnan(similarity):
        return False
    passed_threshold = float(similarity) >= float(threshold)
    if enforce_semantic_filter:
        return passed_threshold and bool(semantic_match)
    return passed_threshold


# ==============================================================================
# PIPELINE INTEGRATION & EXPLAINABLE OUTPUT GENERATION
# ==============================================================================

def apply_adaptive_threshold(
    candidates_df: pd.DataFrame,
    use_adaptive: bool = USE_ADAPTIVE_THRESHOLD,
    use_semantic_type: bool = USE_SEMANTIC_TYPE_MATCHING,
    base_threshold: float = BASE_THRESHOLD,
    min_threshold: float = MIN_ADAPTIVE_THRESHOLD,
    max_threshold: float = MAX_ADAPTIVE_THRESHOLD,
    uncertainty_weight: float = UNCERTAINTY_WEIGHT,
    confidence_weight: float = CONFIDENCE_WEIGHT,
    margin_weight: float = MARGIN_WEIGHT,
    triple_weight: float = TRIPLE_WEIGHT,
    semantic_weight: float = SEMANTIC_BONUS_WEIGHT,
    graph1: Optional[Any] = None,
    graph2: Optional[Any] = None,
    entity_sem_prob_df: Optional[pd.DataFrame] = None,
    entity_col1: str = 'e1',
    entity_col2: str = 'e2',
    sim_col: str = 'sim',
    triple_sim_col: Optional[str] = None,
    subject_sim_col: Optional[str] = None,
    predicate_sim_col: Optional[str] = None,
    object_sim_col: Optional[str] = None,
    use_gpu: bool = False,
    use_semantic_type_matching: Optional[bool] = None,
    semantic_bonus_weight: Optional[float] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Apply Adaptive Threshold Optimization (or Baseline thresholding) to a candidate DataFrame.


    Produces an explainable output containing every decision component:
        - entity_kg1
        - entity_kg2
        - similarity
        - best_candidate_score
        - second_candidate_score
        - candidate_margin
        - base_threshold
        - uncertainty_penalty
        - confidence_bonus
        - adaptive_threshold
        - decision
        (and triple-level similarities if provided)

    Parameters:
        candidates_df: DataFrame containing candidate pairs.
        use_adaptive: If False, runs original baseline thresholding. If True, applies Improvement 2.
        base_threshold: Base threshold value.
        min_threshold: Lower clamping bound.
        max_threshold: Upper clamping bound.
        uncertainty_weight: Uncertainty penalty weight.
        confidence_weight: Similarity confidence bonus weight.
        margin_weight: Margin confidence bonus weight.
        triple_weight: Triple agreement bonus weight.
        entity_col1: Column name for KG1 entity in candidates_df.
        entity_col2: Column name for KG2 entity in candidates_df.
        sim_col: Column name for similarity score in candidates_df.
        triple_sim_col: Optional column name for triple similarity.
        subject_sim_col: Optional column name for subject similarity.
        predicate_sim_col: Optional column name for predicate similarity.
        object_sim_col: Optional column name for object similarity.

    Returns:
        Explainable DataFrame with all decision metrics.
    """
    if candidates_df.empty:
        cols = [
            'entity_kg1', 'entity_kg2', 'similarity', 'best_candidate_score',
            'second_candidate_score', 'candidate_margin',
            'semantic_type_kg1', 'semantic_type_kg2', 'semantic_type_similarity',
            'semantic_type_match', 'semantic_type_bonus',
            'base_threshold', 'uncertainty_penalty', 'confidence_bonus',
            'adaptive_threshold', 'decision'
        ]
        return pd.DataFrame(columns=cols)

    df = candidates_df.copy()

    if use_semantic_type_matching is not None:
        use_semantic_type = bool(use_semantic_type_matching)
    if semantic_bonus_weight is not None:
        semantic_weight = float(semantic_bonus_weight)

    # Step 1: Integrate Improvement 1 Semantic Type Matching
    if use_semantic_type:
        if 'semantic_type_similarity' not in df.columns and stm is not None:
            df = stm.apply_semantic_type_matching(
                candidates_df=df,
                graph1=graph1,
                graph2=graph2,
                entity_sem_prob_df=entity_sem_prob_df,
                entity_col1=entity_col1,
                entity_col2=entity_col2,
                sim_col=sim_col,
                semantic_weight=semantic_weight,
                use_semantic_type=True,
                use_gpu=use_gpu
            )
        elif 'semantic_type_similarity' not in df.columns:
            df['semantic_type_kg1'] = 'unspecified'
            df['semantic_type_kg2'] = 'unspecified'
            df['semantic_type_similarity'] = 0.50
            df['semantic_type_match'] = True
            df['semantic_type_bonus'] = 0.0
    else:
        df['semantic_type_kg1'] = 'unspecified'
        df['semantic_type_kg2'] = 'unspecified'
        df['semantic_type_similarity'] = 0.0
        df['semantic_type_match'] = True
        df['semantic_type_bonus'] = 0.0

    # Ensure numeric similarity
    df[sim_col] = pd.to_numeric(df[sim_col], errors='coerce').fillna(0.0)

    # Sort descending by similarity within each e1
    df = df.sort_values([entity_col1, sim_col], ascending=[True, False]).reset_index(drop=True)

    semantic_bonus_series = pd.to_numeric(df['semantic_type_bonus'], errors='coerce').fillna(0.0) if use_semantic_type else pd.Series(0.0, index=df.index)
    semantic_match_series = df['semantic_type_match'] if use_semantic_type else pd.Series(True, index=df.index)
    semantic_sim_series = pd.to_numeric(df['semantic_type_similarity'], errors='coerce').fillna(0.0) if use_semantic_type else pd.Series(0.0, index=df.index)

    # Calculate best and second-best candidate scores and margins per entity
    if gpu_compute_candidate_margins is not None:
        e1_codes, _ = pd.factorize(df[entity_col1])
        dev_arg = None  # Uses cuda:0 by default if CUDA available
        margins, ranks, best_exp, second_exp = gpu_compute_candidate_margins(
            entity_codes=e1_codes,
            similarities=df[sim_col].values.astype(np.float32),
            device=dev_arg
        )
        df['_rank'] = ranks
        df['candidate_margin'] = margins
        df['_best_score'] = best_exp
        df['_second_score'] = second_exp
    else:
        df['_rank'] = df.groupby(entity_col1).cumcount() + 1
        best_df = df[df['_rank'] == 1][[entity_col1, sim_col]].rename(columns={sim_col: '_best_score'})
        second_df = df[df['_rank'] == 2][[entity_col1, sim_col]].rename(columns={sim_col: '_second_score'})
        df = df.merge(best_df, on=entity_col1, how='left')
        df = df.merge(second_df, on=entity_col1, how='left')
        df['_second_score'] = df['_second_score'].fillna(0.0)
        df['candidate_margin'] = np.where(
            df['_rank'] == 1,
            np.clip(df['_best_score'] - df['_second_score'], 0.0, 1.0),
            0.0
        )

    # Extract optional triple similarity
    if triple_sim_col is None:
        if 'triple_sim' in df.columns:
            triple_sim_col = 'triple_sim'
        elif 'triple_similarity' in df.columns:
            triple_sim_col = 'triple_similarity'

    if triple_sim_col and triple_sim_col in df.columns:
        triple_sim_series = pd.to_numeric(df[triple_sim_col], errors='coerce').fillna(0.0)
    else:
        triple_sim_series = pd.Series(0.0, index=df.index)

    # Compute adaptive thresholds, penalties, and bonuses
    if not use_adaptive:
        # Baseline / Fixed threshold mode
        df['base_threshold'] = float(base_threshold)
        df['uncertainty_penalty'] = 0.0
        df['confidence_bonus'] = 0.0
        df['adaptive_threshold'] = float(base_threshold)

        if use_semantic_type:
            # Mode 2 (Improvement 1 standalone): fixed threshold WITH semantic type verification
            df['decision'] = (df[sim_col] >= df['adaptive_threshold']) & (semantic_match_series == True)
        else:
            # Mode 1 (Baseline FTM): fixed threshold WITHOUT semantic type filtering
            df['decision'] = df[sim_col] >= df['adaptive_threshold']

    elif use_gpu and is_cuda_available() and gpu_accelerate_adaptive_threshold is not None:
        # GPU Accelerated Improvement 2 mode (CUDA / NVIDIA GPU)
        p, b, th, dec = gpu_accelerate_adaptive_threshold(
            similarities=df[sim_col].values,
            margins=df['candidate_margin'].values,
            triple_sims=triple_sim_series.values,
            semantic_bonuses=semantic_bonus_series.values,
            base_threshold=base_threshold,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            uncertainty_weight=uncertainty_weight,
            confidence_weight=confidence_weight,
            margin_weight=margin_weight,
            triple_weight=triple_weight
        )
        df['base_threshold'] = float(base_threshold)
        df['uncertainty_penalty'] = p
        df['confidence_bonus'] = b
        df['adaptive_threshold'] = th
        df['decision'] = dec
    else:
        # Vectorized CPU / Improvement 2 mode integrating Improvement 1
        penalties = [
            calculate_uncertainty_penalty(m, uncertainty_weight) +
            (0.05 if (use_semantic_type and not match and s_sim < 0.25) else 0.0)
            for m, match, s_sim in zip(df['candidate_margin'], semantic_match_series, semantic_sim_series)
        ]
        bonuses = [
            calculate_confidence_bonus(s, m, confidence_weight, margin_weight, ts, triple_weight, sb)
            for s, m, ts, sb in zip(df[sim_col], df['candidate_margin'], triple_sim_series, semantic_bonus_series)
        ]
        thresholds = [
            max(float(min_threshold), min(float(max_threshold), float(base_threshold) + p - b))
            for p, b in zip(penalties, bonuses)
        ]

        df['base_threshold'] = float(base_threshold)
        df['uncertainty_penalty'] = penalties
        df['confidence_bonus'] = bonuses
        df['adaptive_threshold'] = thresholds
        df['decision'] = df[sim_col] >= df['adaptive_threshold']

    # Rename to clean explainable columns
    rename_map = {
        entity_col1: 'entity_kg1',
        entity_col2: 'entity_kg2',
        sim_col: 'similarity',
        '_best_score': 'best_candidate_score',
        '_second_score': 'second_candidate_score'
    }
    df = df.rename(columns=rename_map)

    # Standard columns ordered cleanly
    output_cols = [
        'entity_kg1', 'entity_kg2', 'similarity',
        'best_candidate_score', 'second_candidate_score', 'candidate_margin',
        'semantic_type_kg1', 'semantic_type_kg2', 'semantic_type_similarity',
        'semantic_type_match', 'semantic_type_bonus',
        'base_threshold', 'uncertainty_penalty', 'confidence_bonus',
        'adaptive_threshold', 'decision'
    ]

    # Include optional triple columns if present
    if subject_sim_col and subject_sim_col in df.columns:
        df['subject_similarity'] = pd.to_numeric(df[subject_sim_col], errors='coerce').fillna(0.0)
        output_cols.append('subject_similarity')
    if predicate_sim_col and predicate_sim_col in df.columns:
        df['predicate_similarity'] = pd.to_numeric(df[predicate_sim_col], errors='coerce').fillna(0.0)
        output_cols.append('predicate_similarity')
    if object_sim_col and object_sim_col in df.columns:
        df['object_similarity'] = pd.to_numeric(df[object_sim_col], errors='coerce').fillna(0.0)
        output_cols.append('object_similarity')
    if triple_sim_col and triple_sim_col in df.columns:
        df['triple_similarity'] = triple_sim_series
        output_cols.append('triple_similarity')

    return df[output_cols]


# ==============================================================================
# EVALUATION METRICS (PRECISION, RECALL, F1, HIT@1, HIT@10)
# ==============================================================================

def evaluate_matches(
    predictions_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    gold_col1: str = 'e1',
    gold_col2: str = 'e2',
    entity_col1: str = 'entity_kg1',
    entity_col2: str = 'entity_kg2',
    decision_col: str = 'decision',
    sim_col: str = 'similarity',
    one_to_one: bool = True
) -> Dict[str, Union[int, float]]:
    """
    Evaluate predicted entity matches against ground truth gold mapping.

    Calculates:
        - Total predicted matches
        - True Positives (TP)
        - False Positives (FP)
        - False Negatives (FN)
        - Precision: TP / (TP + FP)
        - Recall: TP / Total Gold
        - F1 Score: 2 * (P * R) / (P + R)
        - Hit@1: Proportion of gold e1 entities whose true e2 is ranked #1 in candidate list
        - Hit@10: Proportion of gold e1 entities whose true e2 is in top-10 candidates

    Parameters:
        predictions_df: DataFrame with predictions and decision column.
        gold_df: DataFrame containing true matching pairs.
        gold_col1: Column name for KG1 entity in gold_df.
        gold_col2: Column name for KG2 entity in gold_df.
        entity_col1: Column name for KG1 entity in predictions_df.
        entity_col2: Column name for KG2 entity in predictions_df.
        decision_col: Column containing True/False decision.
        sim_col: Column containing similarity score.
        one_to_one: If True, enforce 1-to-1 matching constraint on accepted matches.

    Returns:
        Dictionary of evaluation metrics.
    """
    gold_set = set(zip(gold_df[gold_col1].astype(str), gold_df[gold_col2].astype(str)))
    total_gold = len(gold_set)

    if predictions_df.empty:
        return {
            "total_gold": total_gold,
            "predicted_matches": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": total_gold,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "hit_at_1": 0.0,
            "hit_at_10": 0.0
        }

    # Filter accepted matches (decision == True)
    accepted = predictions_df[predictions_df[decision_col] == True].copy()

    # Enforce 1-to-1 matching if requested (standard FTM post-processing)
    rank_metric = 'resolution_score' if ('resolution_score' in accepted.columns and accepted['resolution_score'].abs().sum() > 0) else sim_col
    if one_to_one and not accepted.empty:
        accepted = accepted.sort_values(rank_metric, ascending=False).drop_duplicates(entity_col1).drop_duplicates(entity_col2)

    predicted_pairs = set(zip(accepted[entity_col1].astype(str), accepted[entity_col2].astype(str)))
    tp = len(predicted_pairs.intersection(gold_set))
    fp = len(predicted_pairs - gold_set)
    fn = total_gold - tp

    precision = tp / len(predicted_pairs) if len(predicted_pairs) > 0 else 0.0
    recall = tp / total_gold if total_gold > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Hit@1 and Hit@10 calculation based on ranking (resolution_score if available, else sim_col)
    # Group all candidates by entity_col1 sorted by rank_metric descending
    sorted_cands = predictions_df.sort_values([entity_col1, rank_metric], ascending=[True, False])
    grouped = sorted_cands.groupby(entity_col1)

    gold_e1_to_e2 = {str(r[gold_col1]): str(r[gold_col2]) for _, r in gold_df.iterrows()}

    hit_1_count = 0
    hit_10_count = 0
    evaluated_gold_entities = 0

    for e1, true_e2 in gold_e1_to_e2.items():
        if e1 in grouped.groups:
            evaluated_gold_entities += 1
            group = grouped.get_group(e1)
            top_candidates = group[entity_col2].astype(str).tolist()

            if len(top_candidates) > 0 and top_candidates[0] == true_e2:
                hit_1_count += 1
            if true_e2 in top_candidates[:10]:
                hit_10_count += 1

    hit_at_1 = hit_1_count / total_gold if total_gold > 0 else 0.0
    hit_at_10 = hit_10_count / total_gold if total_gold > 0 else 0.0

    return {
        "total_gold": total_gold,
        "predicted_matches": len(predicted_pairs),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "hit_at_1": round(float(hit_at_1), 6),
        "hit_at_10": round(float(hit_at_10), 6)
    }
