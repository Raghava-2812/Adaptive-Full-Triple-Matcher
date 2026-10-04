"""
Automated Conflict Resolution for Full-Triple-Matcher (FTM)
Improvement 3: Automated identification, analysis, and resolution of matching conflicts
building on Improvement 2 (Adaptive Threshold Optimization).

This module implements:
1. Conflict Detection:
   - Conflict Type 1: One-to-Many Conflict (Source entity mapped to multiple target candidates)
   - Conflict Type 2: Many-to-One Conflict (Multiple source entities mapped to same target candidate)
   - Conflict Type 3: Competing Candidate Conflict (Top candidates with close scores / margin ambiguity)
   - Conflict Type 4: Structural Conflict (Mappings with structural/relational inconsistency or low triple support)
2. Conflict Analysis:
   - Evaluation of multi-source evidence: entity similarity, triple similarity,
     candidate margin, structural neighborhood consistency, and conflict penalties.
3. Automated Conflict Resolution:
   - Multi-stage resolution assigning the most consistently supported match,
     rejecting weaker conflicting matches with audit logs, and flagging genuine ambiguities as UNRESOLVED.

Author: CSE Project - Full-Triple-Matcher Enhancements (Improvement 3)
"""

import math
import os
import sys
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import numpy as np
import pandas as pd
from rdflib import URIRef

# Load GPU utilities if available
try:
    import gpu_utils
except ImportError:
    utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils"))
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    try:
        import gpu_utils
    except ImportError:
        gpu_utils = None


# ==============================================================================
# CONFIGURATION SECTION
# ==============================================================================
# Setting USE_SEMANTIC_TYPE_MATCHING = True activates Improvement 1 integration
USE_SEMANTIC_TYPE_MATCHING: bool = True

# Setting USE_CONFLICT_RESOLUTION = False preserves Improvement 2 / baseline behavior.
# Setting USE_CONFLICT_RESOLUTION = True activates Improvement 3.
USE_CONFLICT_RESOLUTION: bool = False

# Conflict Resolution Weights with Semantic Evidence (Sum = 1.0)
CONFLICT_ENTITY_WEIGHT: float = 0.35      # Weight for entity similarity evidence
CONFLICT_TRIPLE_WEIGHT: float = 0.25      # Weight for triple similarity evidence
CONFLICT_STRUCTURE_WEIGHT: float = 0.15   # Weight for graph structural neighborhood consistency
CONFLICT_SEMANTIC_WEIGHT: float = 0.15    # Weight for semantic type compatibility evidence (Improvement 1)
CONFLICT_MARGIN_WEIGHT: float = 0.10      # Weight for candidate score margin bonus

# Legacy Fallback Weights (when semantic matching is OFF)
LEGACY_ENTITY_WEIGHT: float = 0.40
LEGACY_TRIPLE_WEIGHT: float = 0.30
LEGACY_STRUCTURE_WEIGHT: float = 0.20
LEGACY_MARGIN_WEIGHT: float = 0.10

# Penalties and Thresholds
CONFLICT_PENALTY: float = 0.05            # Penalty deducted if a candidate is involved in a conflict
AMBIGUITY_THRESHOLD: float = 0.02         # If score difference between top 2 competing candidates < this, mark UNRESOLVED
STRUCTURAL_DIVERGENCE_CUTOFF: float = 0.20 # If (entity_sim - triple_sim) > this, flag structural conflict


class ConflictType:
    NONE = "none"
    ONE_TO_MANY = "one_to_many"             # Type 1: Same e1 mapped to multiple e2 candidates
    MANY_TO_ONE = "many_to_one"             # Type 2: Multiple e1 mapped to same e2 candidate
    COMPETING_CANDIDATE = "competing_candidate" # Type 3: Top candidate has close runner-up
    STRUCTURAL_CONFLICT = "structural_conflict" # Type 4: Low structural support / neighborhood divergence
    SEMANTIC_TYPE_CONFLICT = "semantic_type_conflict" # Type 5: Disjoint semantic category mismatch (Improvement 1)
    MULTIPLE = "multiple_conflicts"         # Combination of multiple conflict types


class ResolutionStatus:
    ACCEPTED = "ACCEPTED"
    REJECTED_THRESHOLD = "REJECTED_THRESHOLD"
    REJECTED_CONFLICT = "REJECTED_CONFLICT"
    UNRESOLVED = "UNRESOLVED"


class ConflictResolutionConfig:
    """
    Configuration holder for Conflict Resolution parameters.
    """
    def __init__(
        self,
        use_conflict_resolution: bool = USE_CONFLICT_RESOLUTION,
        use_semantic_type_matching: bool = USE_SEMANTIC_TYPE_MATCHING,
        entity_weight: float = CONFLICT_ENTITY_WEIGHT,
        triple_weight: float = CONFLICT_TRIPLE_WEIGHT,
        structure_weight: float = CONFLICT_STRUCTURE_WEIGHT,
        semantic_weight: float = CONFLICT_SEMANTIC_WEIGHT,
        margin_weight: float = CONFLICT_MARGIN_WEIGHT,
        conflict_penalty: float = CONFLICT_PENALTY,
        ambiguity_threshold: float = AMBIGUITY_THRESHOLD,
        structural_cutoff: float = STRUCTURAL_DIVERGENCE_CUTOFF
    ):
        self.use_conflict_resolution = bool(use_conflict_resolution)
        self.use_semantic_type_matching = bool(use_semantic_type_matching)
        self.entity_weight = float(entity_weight)
        self.triple_weight = float(triple_weight)
        self.structure_weight = float(structure_weight)
        self.semantic_weight = float(semantic_weight)
        self.margin_weight = float(margin_weight)
        self.conflict_penalty = float(conflict_penalty)
        self.ambiguity_threshold = float(ambiguity_threshold)
        self.structural_cutoff = float(structural_cutoff)

    def to_dict(self) -> Dict[str, Union[bool, float]]:
        return {
            "USE_CONFLICT_RESOLUTION": self.use_conflict_resolution,
            "USE_SEMANTIC_TYPE_MATCHING": self.use_semantic_type_matching,
            "CONFLICT_ENTITY_WEIGHT": self.entity_weight,
            "CONFLICT_TRIPLE_WEIGHT": self.triple_weight,
            "CONFLICT_STRUCTURE_WEIGHT": self.structure_weight,
            "CONFLICT_SEMANTIC_WEIGHT": self.semantic_weight,
            "CONFLICT_MARGIN_WEIGHT": self.margin_weight,
            "CONFLICT_PENALTY": self.conflict_penalty,
            "AMBIGUITY_THRESHOLD": self.ambiguity_threshold,
            "STRUCTURAL_DIVERGENCE_CUTOFF": self.structural_cutoff,
        }


# ==============================================================================
# EVIDENCE & RESOLUTION SCORE FUNCTIONS
# ==============================================================================

def calculate_resolution_score(
    entity_sim: float,
    triple_sim: float,
    structural_consistency: float,
    candidate_margin: float,
    has_conflict: bool,
    semantic_type_sim: Optional[float] = None,
    semantic_type_match: Optional[bool] = None,
    entity_weight: float = CONFLICT_ENTITY_WEIGHT,
    triple_weight: float = CONFLICT_TRIPLE_WEIGHT,
    structure_weight: float = CONFLICT_STRUCTURE_WEIGHT,
    semantic_weight: float = CONFLICT_SEMANTIC_WEIGHT,
    margin_weight: float = CONFLICT_MARGIN_WEIGHT,
    conflict_penalty: float = CONFLICT_PENALTY
) -> float:
    """
    Calculate explainable composite resolution score.

    Mathematical formulation (Semantic ON):
        Resolution Score = (w_entity * entity_sim) +
                           (w_triple * triple_sim) +
                           (w_struct * structural_consistency) +
                           (w_sem * semantic_type_sim) +
                           (w_margin * candidate_margin) -
                           (penalty if has_conflict else 0.0) -
                           (semantic_mismatch_penalty if semantic_type_match is False else 0.0)

    Range: Clamped to [0.0, 1.0].
    """
    e_sim = max(0.0, min(1.0, float(entity_sim))) if entity_sim is not None else 0.0
    t_sim = max(0.0, min(1.0, float(triple_sim))) if triple_sim is not None else 0.0
    s_sim = max(0.0, min(1.0, float(structural_consistency))) if structural_consistency is not None else 0.0
    m_val = max(0.0, min(1.0, float(candidate_margin))) if candidate_margin is not None else 0.0

    if semantic_type_sim is not None and semantic_weight > 0.0:
        sem_sim = max(0.0, min(1.0, float(semantic_type_sim)))
        raw_score = (
            float(entity_weight) * e_sim +
            float(triple_weight) * t_sim +
            float(structure_weight) * s_sim +
            float(semantic_weight) * sem_sim +
            float(margin_weight) * m_val
        )
        if semantic_type_match is False:
            raw_score -= float(conflict_penalty) * 2.0  # Strong penalty for semantic category mismatch
    else:
        # Legacy weights (summing to 1.0 without semantic term)
        raw_score = (
            float(LEGACY_ENTITY_WEIGHT) * e_sim +
            float(LEGACY_TRIPLE_WEIGHT) * t_sim +
            float(LEGACY_STRUCTURE_WEIGHT) * s_sim +
            float(LEGACY_MARGIN_WEIGHT) * m_val
        )

    if has_conflict:
        raw_score -= float(conflict_penalty)

    return max(0.0, min(1.0, raw_score))


def compute_graph_neighborhood_consistency(
    e1_uri: str,
    e2_uri: str,
    graph1: Optional[Any] = None,
    graph2: Optional[Any] = None,
    default_consistency: float = 0.5
) -> float:
    """
    Compute structural consistency of candidate pair (e1, e2) using RDF graphs if available.
    Measures the alignment ratio of outgoing (predicate-object) and incoming relations.
    If graphs are not provided, returns default_consistency.
    """
    if graph1 is None or graph2 is None:
        return float(default_consistency)

    try:
        try:
            import triple_matching as tm
            if hasattr(tm, 'calculate_structural_similarity'):
                return float(tm.calculate_structural_similarity(e1_uri, e2_uri, graph1, graph2))
        except (ImportError, Exception):
            pass

        e1_ref = URIRef(str(e1_uri))
        e2_ref = URIRef(str(e2_uri))

        out1 = list(graph1.predicate_objects(e1_ref))
        out2 = list(graph2.predicate_objects(e2_ref))
        if not out1 or not out2:
            return float(default_consistency)

        p1_names = {str(p).split('/')[-1].split('#')[-1].lower() for p, _ in out1}
        p2_names = {str(p).split('/')[-1].split('#')[-1].lower() for p, _ in out2}
        common = p1_names.intersection(p2_names)
        consistency = len(common) / max(len(p1_names), len(p2_names))
        return max(0.0, min(1.0, float(consistency)))
    except Exception:
        return float(default_consistency)


# ==============================================================================
# CONFLICT DETECTION & RESOLUTION PIPELINE
# ==============================================================================

def detect_and_resolve_conflicts(
    adaptive_decisions_df: pd.DataFrame,
    use_conflict_resolution: bool = USE_CONFLICT_RESOLUTION,
    use_semantic_type_matching: bool = USE_SEMANTIC_TYPE_MATCHING,
    entity_weight: Optional[float] = None,
    triple_weight: Optional[float] = None,
    structure_weight: Optional[float] = None,
    semantic_weight: Optional[float] = None,
    margin_weight: Optional[float] = None,
    conflict_penalty: float = CONFLICT_PENALTY,
    ambiguity_threshold: float = AMBIGUITY_THRESHOLD,
    structural_cutoff: float = STRUCTURAL_DIVERGENCE_CUTOFF,
    graph1: Optional[Any] = None,
    graph2: Optional[Any] = None,
    entity_col1: str = 'entity_kg1',
    entity_col2: str = 'entity_kg2',
    sim_col: str = 'similarity',
    triple_sim_col: str = 'triple_similarity',
    semantic_sim_col: str = 'semantic_type_similarity',
    semantic_match_col: str = 'semantic_type_match',
    decision_col: str = 'decision',
    use_gpu: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Multi-stage Automated Conflict Resolution on accepted candidates from Improvement 2,
    integrating Improvement 1 (Semantic Type Matching) as an independent evidence dimension.

    Parameters:
        adaptive_decisions_df: DataFrame output from Improvement 2 (apply_adaptive_threshold).
        use_conflict_resolution: Boolean flag. If False, retains Improvement 2 output unchanged.
        use_semantic_type_matching: Boolean flag. If True, incorporates semantic type evidence.
        entity_weight: Weight for entity similarity.
        triple_weight: Weight for triple similarity.
        structure_weight: Weight for graph neighborhood consistency.
        semantic_weight: Weight for semantic type compatibility (Improvement 1).
        margin_weight: Weight for score margin.
        conflict_penalty: Penalty deducted if involved in a conflict.
        ambiguity_threshold: Minimum score delta to break ties. Below this, marks UNRESOLVED.
        structural_cutoff: Threshold for flagging structural divergence.
        graph1: Optional rdflib.Graph for KG1.
        graph2: Optional rdflib.Graph for KG2.

    Returns:
        Tuple of:
            1. resolved_df: Full DataFrame with all explainable audit columns and final decisions.
            2. conflict_log_df: DataFrame logging every detected conflict, semantic compatibility, and resolution rationale.
            3. stats: Dictionary containing summary conflict resolution statistics.
    """
    df = adaptive_decisions_df.copy()

    # Resolve evidence weights depending on whether semantic matching is enabled
    if entity_weight is None:
        entity_weight = CONFLICT_ENTITY_WEIGHT if use_semantic_type_matching else LEGACY_ENTITY_WEIGHT
    if triple_weight is None:
        triple_weight = CONFLICT_TRIPLE_WEIGHT if use_semantic_type_matching else LEGACY_TRIPLE_WEIGHT
    if structure_weight is None:
        structure_weight = CONFLICT_STRUCTURE_WEIGHT if use_semantic_type_matching else LEGACY_STRUCTURE_WEIGHT
    if semantic_weight is None:
        semantic_weight = CONFLICT_SEMANTIC_WEIGHT if use_semantic_type_matching else 0.0
    if margin_weight is None:
        margin_weight = CONFLICT_MARGIN_WEIGHT if use_semantic_type_matching else LEGACY_MARGIN_WEIGHT

    empty_log_columns = [
        'conflict_id', 'source_entity', 'candidate_target_1', 'candidate_target_2',
        'competing_targets', 'conflict_type',
        'candidate_scores', 'adaptive_thresholds', 'semantic_type_compatibility',
        'resolution_score', 'resolution_scores', 'selected_candidate',
        'rejected_candidates', 'resolution_status', 'reason'
    ]

    # If conflict resolution is disabled, strictly preserve Improvement 2 output
    if not use_conflict_resolution or df.empty:
        df['conflict_detected'] = False
        df['conflict_type'] = ConflictType.NONE
        df['competing_candidates_count'] = 1
        df['resolution_score'] = df[sim_col] if sim_col in df.columns else 0.0
        df['resolution_status'] = np.where(df[decision_col] == True, ResolutionStatus.ACCEPTED, ResolutionStatus.REJECTED_THRESHOLD)
        df['resolution_reason'] = np.where(df[decision_col] == True, "Accepted by threshold (Conflict resolution OFF)", "Rejected below threshold")
        
        empty_log = pd.DataFrame(columns=empty_log_columns)
        stats = {
            "total_candidates": len(df),
            "accepted_by_threshold": int((df[decision_col] == True).sum()),
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "conflicts_unresolved": 0,
            "final_accepted_matches": int((df[decision_col] == True).sum())
        }
        return df, empty_log, stats

    # Ensure required columns exist
    if triple_sim_col is None or triple_sim_col not in df.columns:
        if 'triple_similarity' in df.columns:
            triple_sim_col = 'triple_similarity'
        elif 'triple_sim' in df.columns:
            triple_sim_col = 'triple_sim'
        else:
            triple_sim_col = 'triple_similarity'
            df[triple_sim_col] = 0.0
    if 'candidate_margin' not in df.columns:
        df['candidate_margin'] = 0.0
    if 'adaptive_threshold' not in df.columns:
        df['adaptive_threshold'] = 0.80

    # Ensure numeric
    df[sim_col] = pd.to_numeric(df[sim_col], errors='coerce').fillna(0.0)
    df[triple_sim_col] = pd.to_numeric(df[triple_sim_col], errors='coerce').fillna(0.0)
    df['candidate_margin'] = pd.to_numeric(df['candidate_margin'], errors='coerce').fillna(0.0)

    # Record Improvement 2 acceptance prior to conflict resolution
    df['passed_adaptive_threshold'] = (df[decision_col] == True)

    # STEP 1: Filter candidates accepted by Improvement 2
    # Candidates with decision == False are already rejected by threshold
    accepted_mask = (df[decision_col] == True)
    accepted_df = df[accepted_mask].copy()

    # Initialize audit columns for all candidates
    df['conflict_detected'] = False
    df['conflict_type'] = ConflictType.NONE
    df['competing_candidates_count'] = 1
    df['structural_consistency'] = 0.0
    df['resolution_score'] = 0.0
    df['resolution_status'] = ResolutionStatus.REJECTED_THRESHOLD
    df['resolution_reason'] = "Rejected: score did not pass adaptive threshold"

    if accepted_df.empty:
        empty_log = pd.DataFrame(columns=empty_log_columns)
        stats = {
            "total_candidates": len(df),
            "accepted_by_threshold": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "conflicts_unresolved": 0,
            "final_accepted_matches": 0
        }
        return df, empty_log, stats

    # STEP 2: Index accepted candidates by source (e1) and target (e2)
    e1_counts = accepted_df.groupby(entity_col1).size()
    e2_counts = accepted_df.groupby(entity_col2).size()

    e1_with_conflicts = set(e1_counts[e1_counts > 1].index)
    e2_with_conflicts = set(e2_counts[e2_counts > 1].index)

    # STEP 3: Compute Structural Consistency & Conflict Detection for accepted pairs
    structural_consistencies = []
    conflict_detected_list = []
    conflict_types_list = []
    competing_counts_list = []

    if graph1 is None and graph2 is None and len(accepted_df) > 1000:
        # Fast vectorized path for large benchmark datasets
        e_sim_arr = accepted_df[sim_col].values.astype(np.float32)
        t_sim_arr = accepted_df[triple_sim_col].values.astype(np.float32) if triple_sim_col in accepted_df.columns else np.zeros_like(e_sim_arr)
        margin_arr = accepted_df['candidate_margin'].values.astype(np.float32)
        sem_match_arr = accepted_df[semantic_match_col].values.astype(bool) if (use_semantic_type_matching and semantic_match_col in accepted_df.columns) else np.ones(len(accepted_df), dtype=bool)

        structural_consistencies = np.where(t_sim_arr > 0.0, t_sim_arr, e_sim_arr).tolist()
        is_o2m = accepted_df[entity_col1].isin(e1_with_conflicts).values
        is_m2o = accepted_df[entity_col2].isin(e2_with_conflicts).values
        is_comp = (margin_arr < 0.05) & (is_o2m | is_m2o)
        is_struct = ((e_sim_arr - t_sim_arr) > structural_cutoff) & (t_sim_arr > 0.0)
        is_sem_mismatch = (~sem_match_arr) if use_semantic_type_matching else np.zeros(len(accepted_df), dtype=bool)

        has_conf_arr = is_o2m | is_m2o | is_comp | is_struct | is_sem_mismatch
        conflict_detected_list = has_conf_arr.tolist()

        c1 = accepted_df[entity_col1].map(e1_counts).fillna(1).astype(int)
        c2 = accepted_df[entity_col2].map(e2_counts).fillna(1).astype(int)
        competing_counts_list = np.maximum(c1, c2).tolist()

        conflict_types_list = []
        for o2m, m2o, comp, st, sem in zip(is_o2m, is_m2o, is_comp, is_struct, is_sem_mismatch):
            dt = []
            if o2m: dt.append(ConflictType.ONE_TO_MANY)
            if m2o: dt.append(ConflictType.MANY_TO_ONE)
            if comp: dt.append(ConflictType.COMPETING_CANDIDATE)
            if st: dt.append(ConflictType.STRUCTURAL_CONFLICT)
            if sem: dt.append(ConflictType.SEMANTIC_TYPE_CONFLICT)
            conflict_types_list.append("+".join(dt) if dt else ConflictType.NONE)
    else:
        for idx, row in accepted_df.iterrows():
            e1 = str(row[entity_col1])
            e2 = str(row[entity_col2])
            e_sim = float(row[sim_col])
            t_sim = float(row[triple_sim_col])
            margin = float(row['candidate_margin'])

            # Semantic check
            has_sem = use_semantic_type_matching and (semantic_match_col in row)
            sem_match = bool(row[semantic_match_col]) if has_sem and pd.notna(row[semantic_match_col]) else True

            # Compute graph neighborhood consistency if graphs provided, otherwise use triple agreement
            if graph1 is not None and graph2 is not None:
                struct_cons = compute_graph_neighborhood_consistency(e1, e2, graph1, graph2)
            else:
                struct_cons = t_sim if t_sim > 0.0 else e_sim

            structural_consistencies.append(struct_cons)

            # Conflict type identification
            detected_types = []
            is_one_to_many = e1 in e1_with_conflicts
            is_many_to_one = e2 in e2_with_conflicts
            is_competing = margin < 0.05 and (is_one_to_many or is_many_to_one)
            is_structural = (e_sim - t_sim) > structural_cutoff and t_sim > 0.0
            is_semantic_mismatch = use_semantic_type_matching and (sem_match is False)

            if is_one_to_many:
                detected_types.append(ConflictType.ONE_TO_MANY)
            if is_many_to_one:
                detected_types.append(ConflictType.MANY_TO_ONE)
            if is_competing:
                detected_types.append(ConflictType.COMPETING_CANDIDATE)
            if is_structural:
                detected_types.append(ConflictType.STRUCTURAL_CONFLICT)
            if is_semantic_mismatch:
                detected_types.append(ConflictType.SEMANTIC_TYPE_CONFLICT)

            has_conf = len(detected_types) > 0
            conflict_detected_list.append(has_conf)

            if not has_conf:
                conf_type_str = ConflictType.NONE
            elif len(detected_types) == 1:
                conf_type_str = detected_types[0]
            else:
                conf_type_str = "+".join(detected_types)
            conflict_types_list.append(conf_type_str)

            # Count competitors
            comp_count = max(int(e1_counts.get(e1, 1)), int(e2_counts.get(e2, 1)))
            competing_counts_list.append(comp_count)

    accepted_df['structural_consistency'] = structural_consistencies
    accepted_df['conflict_detected'] = conflict_detected_list
    accepted_df['conflict_type'] = conflict_types_list
    accepted_df['competing_candidates_count'] = competing_counts_list

    # STEP 4: Compute Resolution Score for every accepted candidate
    if use_semantic_type_matching and semantic_sim_col in accepted_df.columns:
        sem_sims = pd.to_numeric(accepted_df[semantic_sim_col], errors='coerce').fillna(1.0).tolist()
        sem_matches = [
            bool(m) if pd.notna(m) else True
            for m in accepted_df[semantic_match_col].tolist()
        ] if semantic_match_col in accepted_df.columns else [True] * len(accepted_df)
    else:
        sem_sims = [1.0] * len(accepted_df)
        sem_matches = [True] * len(accepted_df)

    if use_gpu and gpu_utils is not None and gpu_utils.is_cuda_available():
        sim_arr = accepted_df[sim_col].to_numpy(dtype=np.float32)
        triple_arr = accepted_df[triple_sim_col].to_numpy(dtype=np.float32)
        struct_arr = np.array(structural_consistencies, dtype=np.float32)
        margin_arr = accepted_df['candidate_margin'].to_numpy(dtype=np.float32)
        conflict_arr = np.array(conflict_detected_list, dtype=bool)
        sem_arr = np.array(sem_sims, dtype=np.float32) if use_semantic_type_matching else None

        resolution_scores = gpu_utils.gpu_accelerate_composite_scores(
            sim_array=sim_arr,
            triple_sim_array=triple_arr,
            struct_array=struct_arr,
            margin_array=margin_arr,
            in_conflict_mask=conflict_arr,
            semantic_sim_array=sem_arr,
            w_entity=entity_weight,
            w_triple=triple_weight,
            w_struct=structure_weight,
            w_semantic=semantic_weight if use_semantic_type_matching else 0.0,
            w_margin=margin_weight,
            conflict_penalty=conflict_penalty
        )
    else:
        resolution_scores = [
            calculate_resolution_score(
                entity_sim=e_sim,
                triple_sim=t_sim,
                structural_consistency=s_cons,
                candidate_margin=margin,
                has_conflict=has_conf,
                semantic_type_sim=s_sem if use_semantic_type_matching else None,
                semantic_type_match=s_match if use_semantic_type_matching else None,
                entity_weight=entity_weight,
                triple_weight=triple_weight,
                structure_weight=structure_weight,
                semantic_weight=semantic_weight if use_semantic_type_matching else 0.0,
                margin_weight=margin_weight,
                conflict_penalty=conflict_penalty
            )
            for e_sim, t_sim, s_cons, margin, has_conf, s_sem, s_match in zip(
                accepted_df[sim_col],
                accepted_df[triple_sim_col],
                accepted_df['structural_consistency'],
                accepted_df['candidate_margin'],
                accepted_df['conflict_detected'],
                sem_sims,
                sem_matches
            )
        ]
    accepted_df['resolution_score'] = resolution_scores

    # Helper for formatting semantic type compatibility string
    def _format_sem_compatibility(row1: pd.Series, row2: Optional[pd.Series] = None) -> str:
        t1_1 = str(row1.get('semantic_type_kg1', 'Unknown'))
        t1_2 = str(row1.get('semantic_type_kg2', 'Unknown'))
        m1 = row1.get(semantic_match_col, True)
        c1 = "Compatible" if (m1 is True or m1 == 1) else "Incompatible"
        res = f"T1: {t1_1} vs {t1_2} ({c1})"
        if row2 is not None:
            t2_1 = str(row2.get('semantic_type_kg1', 'Unknown'))
            t2_2 = str(row2.get('semantic_type_kg2', 'Unknown'))
            m2 = row2.get(semantic_match_col, True)
            c2 = "Compatible" if (m2 is True or m2 == 1) else "Incompatible"
            res += f"; T2: {t2_1} vs {t2_2} ({c2})"
        return res

    # STEP 5: Multi-stage Global Conflict Resolution & Ambiguity Handling
    # Sort all accepted candidates by resolution_score descending
    accepted_df = accepted_df.sort_values(by='resolution_score', ascending=False).reset_index()

    assigned_e1: Set[str] = set()
    assigned_e2: Set[str] = set()

    conflict_logs = []
    conflict_counter = 0
    conflicts_resolved_count = 0
    conflicts_unresolved_count = 0

    final_decisions = {}  # index -> (bool decision, status, reason)

    # First pass: analyze conflicts by source entity e1
    e1_groups = accepted_df.groupby(entity_col1)
    ambiguous_e1: Set[str] = set()

    for e1, grp in e1_groups:
        if len(grp) > 1:
            conflict_counter += 1
            sorted_grp = grp.sort_values('resolution_score', ascending=False)
            top1 = sorted_grp.iloc[0]
            top2 = sorted_grp.iloc[1]
            score_diff = top1['resolution_score'] - top2['resolution_score']

            competing_targets = sorted_grp[entity_col2].tolist()
            cand_scores = [round(float(s), 4) for s in sorted_grp[sim_col].tolist()]
            cand_thresholds = [round(float(t), 4) for t in sorted_grp['adaptive_threshold'].tolist()]
            res_scores = [round(float(s), 4) for s in sorted_grp['resolution_score'].tolist()]

            target_1 = str(competing_targets[0]) if len(competing_targets) > 0 else ""
            target_2 = str(competing_targets[1]) if len(competing_targets) > 1 else ""

            sem_compat = _format_sem_compatibility(top1, top2) if use_semantic_type_matching else "N/A"

            if score_diff < ambiguity_threshold:
                # Genuine ambiguity: difference is too close to call reliably
                ambiguous_e1.add(e1)
                conflicts_unresolved_count += 1
                conflict_logs.append({
                    'conflict_id': f"CONF_{conflict_counter:04d}",
                    'source_entity': e1,
                    'candidate_target_1': target_1,
                    'candidate_target_2': target_2,
                    'competing_targets': "; ".join([str(t) for t in competing_targets]),
                    'conflict_type': top1['conflict_type'],
                    'candidate_scores': "; ".join([str(s) for s in cand_scores]),
                    'adaptive_thresholds': "; ".join([str(t) for t in cand_thresholds]),
                    'semantic_type_compatibility': sem_compat,
                    'resolution_score': round(float(top1['resolution_score']), 4),
                    'resolution_scores': "; ".join([str(s) for s in res_scores]),
                    'selected_candidate': "NONE (UNRESOLVED)",
                    'rejected_candidates': "; ".join([str(t) for t in competing_targets]),
                    'resolution_status': ResolutionStatus.UNRESOLVED,
                    'reason': f"Genuine ambiguity: resolution score diff ({score_diff:.4f}) < ambiguity threshold ({ambiguity_threshold})"
                })
            else:
                conflicts_resolved_count += 1
                conflict_logs.append({
                    'conflict_id': f"CONF_{conflict_counter:04d}",
                    'source_entity': e1,
                    'candidate_target_1': target_1,
                    'candidate_target_2': target_2,
                    'competing_targets': "; ".join([str(t) for t in competing_targets]),
                    'conflict_type': top1['conflict_type'],
                    'candidate_scores': "; ".join([str(s) for s in cand_scores]),
                    'adaptive_thresholds': "; ".join([str(t) for t in cand_thresholds]),
                    'semantic_type_compatibility': sem_compat,
                    'resolution_score': round(float(top1['resolution_score']), 4),
                    'resolution_scores': "; ".join([str(s) for s in res_scores]),
                    'selected_candidate': str(top1[entity_col2]),
                    'rejected_candidates': "; ".join([str(t) for t in competing_targets[1:]]),
                    'resolution_status': "RESOLVED",
                    'reason': f"Selected top candidate by composite resolution evidence (score={top1['resolution_score']:.4f} vs runner-up={top2['resolution_score']:.4f}, diff={score_diff:.4f})"
                })

    # Also log target conflicts (many-to-one) if target e2 claimed by multiple e1
    e2_groups = accepted_df.groupby(entity_col2)
    for e2, grp in e2_groups:
        if len(grp) > 1 and e2 not in e1_with_conflicts:
            conflict_counter += 1
            sorted_grp = grp.sort_values('resolution_score', ascending=False)
            top1 = sorted_grp.iloc[0]
            top2 = sorted_grp.iloc[1]
            score_diff = top1['resolution_score'] - top2['resolution_score']

            competing_sources = sorted_grp[entity_col1].tolist()
            cand_scores = [round(float(s), 4) for s in sorted_grp[sim_col].tolist()]
            cand_thresholds = [round(float(t), 4) for t in sorted_grp['adaptive_threshold'].tolist()]
            res_scores = [round(float(s), 4) for s in sorted_grp['resolution_score'].tolist()]

            source_1 = str(competing_sources[0]) if len(competing_sources) > 0 else ""
            source_2 = str(competing_sources[1]) if len(competing_sources) > 1 else ""

            sem_compat = _format_sem_compatibility(top1, top2) if use_semantic_type_matching else "N/A"

            if score_diff < ambiguity_threshold:
                conflicts_unresolved_count += 1
                conflict_logs.append({
                    'conflict_id': f"CONF_{conflict_counter:04d}",
                    'source_entity': f"[Target] {e2}",
                    'candidate_target_1': source_1,
                    'candidate_target_2': source_2,
                    'competing_targets': "; ".join([str(s) for s in competing_sources]),
                    'conflict_type': ConflictType.MANY_TO_ONE,
                    'candidate_scores': "; ".join([str(s) for s in cand_scores]),
                    'adaptive_thresholds': "; ".join([str(t) for t in cand_thresholds]),
                    'semantic_type_compatibility': sem_compat,
                    'resolution_score': round(float(top1['resolution_score']), 4),
                    'resolution_scores': "; ".join([str(s) for s in res_scores]),
                    'selected_candidate': "NONE (UNRESOLVED)",
                    'rejected_candidates': "; ".join([str(s) for s in competing_sources]),
                    'resolution_status': ResolutionStatus.UNRESOLVED,
                    'reason': f"Many-to-one ambiguity: score diff ({score_diff:.4f}) < threshold ({ambiguity_threshold})"
                })
            else:
                conflicts_resolved_count += 1
                conflict_logs.append({
                    'conflict_id': f"CONF_{conflict_counter:04d}",
                    'source_entity': f"[Target] {e2}",
                    'candidate_target_1': source_1,
                    'candidate_target_2': source_2,
                    'competing_targets': "; ".join([str(s) for s in competing_sources]),
                    'conflict_type': ConflictType.MANY_TO_ONE,
                    'candidate_scores': "; ".join([str(s) for s in cand_scores]),
                    'adaptive_thresholds': "; ".join([str(t) for t in cand_thresholds]),
                    'semantic_type_compatibility': sem_compat,
                    'resolution_score': round(float(top1['resolution_score']), 4),
                    'resolution_scores': "; ".join([str(s) for s in res_scores]),
                    'selected_candidate': str(top1[entity_col1]),
                    'rejected_candidates': "; ".join([str(s) for s in competing_sources[1:]]),
                    'resolution_status': "RESOLVED",
                    'reason': f"Many-to-one resolved in favor of source {top1[entity_col1]} (score={top1['resolution_score']:.4f} vs {top2['resolution_score']:.4f})"
                })

    # Second pass: Greedy assignment in descending order of resolution score
    for _, row in accepted_df.iterrows():
        orig_idx = row['index']
        e1 = str(row[entity_col1])
        e2 = str(row[entity_col2])
        score = float(row['resolution_score'])
        has_conf = bool(row['conflict_detected'])

        if e1 in ambiguous_e1:
            # Candidate is part of an unresolved ambiguous conflict
            final_decisions[orig_idx] = (
                False,
                ResolutionStatus.UNRESOLVED,
                f"Flagged UNRESOLVED: competing candidate scores within ambiguity threshold ({ambiguity_threshold})"
            )
            continue

        sem_match = row.get(semantic_match_col, True) if use_semantic_type_matching else True
        if use_semantic_type_matching and sem_match is False:
            final_decisions[orig_idx] = (
                False,
                ResolutionStatus.REJECTED_CONFLICT,
                f"Rejected: Semantic type mismatch ({row.get('semantic_type_kg1')} vs {row.get('semantic_type_kg2')})"
            )
            continue

        if e1 in assigned_e1:
            final_decisions[orig_idx] = (
                False,
                ResolutionStatus.REJECTED_CONFLICT,
                f"Rejected: source entity {e1} already resolved to a higher-scoring candidate"
            )
            continue

        if e2 in assigned_e2:
            final_decisions[orig_idx] = (
                False,
                ResolutionStatus.REJECTED_CONFLICT,
                f"Rejected: target entity {e2} already claimed by a higher-scoring source entity"
            )
            continue

        # Valid conflict-free assignment
        assigned_e1.add(e1)
        assigned_e2.add(e2)
        if has_conf:
            sem_note = f", sem_sim={row.get(semantic_sim_col, 1.0):.2f}" if use_semantic_type_matching else ""
            reason_msg = f"Accepted: Won conflict resolution with composite score {score:.4f} (entity={row[sim_col]:.4f}, triple={row[triple_sim_col]:.4f}, struct={row['structural_consistency']:.4f}{sem_note})"
        else:
            reason_msg = f"Accepted: Passed threshold with no competing conflict (score={score:.4f})"

        final_decisions[orig_idx] = (True, ResolutionStatus.ACCEPTED, reason_msg)

    # STEP 6: Merge decisions back into main DataFrame
    for orig_idx, (dec, status, reason) in final_decisions.items():
        df.loc[orig_idx, decision_col] = dec
        df.loc[orig_idx, 'resolution_status'] = status
        df.loc[orig_idx, 'resolution_reason'] = reason

    # Update metadata columns from accepted_df
    for _, row in accepted_df.iterrows():
        orig_idx = row['index']
        df.loc[orig_idx, 'conflict_detected'] = row['conflict_detected']
        df.loc[orig_idx, 'conflict_type'] = row['conflict_type']
        df.loc[orig_idx, 'competing_candidates_count'] = row['competing_candidates_count']
        df.loc[orig_idx, 'structural_consistency'] = row['structural_consistency']
        df.loc[orig_idx, 'resolution_score'] = row['resolution_score']

    conflict_log_df = pd.DataFrame(conflict_logs)

    final_accepted_count = int((df[decision_col] == True).sum())
    stats = {
        "total_candidates": len(df),
        "accepted_by_threshold": len(accepted_df),
        "conflicts_detected": conflict_counter,
        "conflicts_resolved": conflicts_resolved_count,
        "conflicts_unresolved": conflicts_unresolved_count,
        "final_accepted_matches": final_accepted_count
    }

    # Structured logging as per FTM project specifications
    print(f"\n[Conflict Resolution]")
    print(f"Accepted candidates: {len(accepted_df)}")
    print(f"\n[Conflict Detection]")
    print(f"Conflicts found: {conflict_counter}")
    print(f"\n[Conflict Resolution]")
    print(f"Resolved automatically: {conflicts_resolved_count}")
    print(f"Unresolved: {conflicts_unresolved_count}")
    print(f"\n[Final Matching]")
    print(f"Final accepted matches: {final_accepted_count}")

    return df, conflict_log_df, stats
