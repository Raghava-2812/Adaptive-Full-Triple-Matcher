"""
Entity Matching Pipeline Orchestrator for Full-Triple-Matcher (FTM).
Dataset-Agnostic Final Implementation supporting both named benchmarks
(Movies, newMovies, DW-NB) and user-specified explicit Knowledge Graph pairs.

Supports:
1. Baseline matching (fixed global threshold, e.g. 0.80, Semantic OFF, Adaptive OFF, Conflict OFF).
2. Improvement 1: Semantic Type-Aware Matching (ontology & graph evidence, compatibility bonus).
3. Improvement 2: Adaptive Threshold Optimization (uncertainty penalty + confidence bonus).
4. Improvement 3 / Final: Automated Conflict Resolution (1-to-many, many-to-1, competing candidate arbitration).
5. Complete 4-way evaluation with structured export to results/ directory.
"""

import os
import sys
import time
import datetime
import argparse
from typing import Dict, Any, Optional, Tuple, List, Union
import pandas as pd
from rdflib import Graph

# Dynamic path resolution to ensure all internal modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
utils_dir = os.path.abspath(os.path.join(current_dir, "..", "utils"))
config_dir = os.path.abspath(os.path.join(current_dir, "..", "config"))

for p in [current_dir, utils_dir, config_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from file_utils import load_csv, load_rdf_graph, save_csv, save_json, ensure_dir
import gpu_utils
from gpu_utils import is_cuda_available, get_gpu_info, get_device
import semantic_type_matching as stm
import adaptive_threshold as at
import conflict_resolution as cr
import candidate_generation as cg
from evaluation import evaluate_predictions, build_comparison_dataframe
from dataset_config import (
    DATASETS,
    get_project_root,
    resolve_dataset_config,
    validate_dataset_config,
    list_available_datasets,
    normalize_dataset_name
)


class EntityMatcherPipeline:
    """
    High-level execution pipeline for Full-Triple-Matcher Entity Matching
    supporting 4 modes:
      1. Baseline: Fixed global threshold (0.80), Semantic OFF, Adaptive OFF, Conflict OFF.
      2. Improvement 1: Semantic ON, Adaptive OFF, Conflict OFF (Fixed theta = 0.80 + Semantic Verification).
      3. Improvement 2: Semantic ON, Adaptive ON, Conflict OFF (Threshold = theta_base + Uncertainty - Confidence - SemanticBonus).
      4. Improvement 3 / Final: Semantic ON, Adaptive ON, Conflict ON (Holistic arbitration incorporating semantic type compatibility).
    """
    def __init__(
        self,
        base_threshold: float = at.BASE_THRESHOLD,
        min_threshold: float = at.MIN_ADAPTIVE_THRESHOLD,
        max_threshold: float = at.MAX_ADAPTIVE_THRESHOLD,
        uncertainty_weight: float = at.UNCERTAINTY_WEIGHT,
        confidence_weight: float = at.CONFIDENCE_WEIGHT,
        margin_weight: float = at.MARGIN_WEIGHT,
        triple_weight: float = at.TRIPLE_WEIGHT,
        semantic_bonus_weight: float = at.SEMANTIC_BONUS_WEIGHT,
        conflict_entity_weight: float = cr.CONFLICT_ENTITY_WEIGHT,
        conflict_triple_weight: float = cr.CONFLICT_TRIPLE_WEIGHT,
        conflict_structure_weight: float = cr.CONFLICT_STRUCTURE_WEIGHT,
        conflict_semantic_weight: float = cr.CONFLICT_SEMANTIC_WEIGHT,
        conflict_margin_weight: float = cr.CONFLICT_MARGIN_WEIGHT,
        conflict_penalty: float = cr.CONFLICT_PENALTY,
        ambiguity_threshold: float = cr.AMBIGUITY_THRESHOLD
    ):
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.uncertainty_weight = uncertainty_weight
        self.confidence_weight = confidence_weight
        self.margin_weight = margin_weight
        self.triple_weight = triple_weight
        self.semantic_bonus_weight = semantic_bonus_weight
        self.conflict_entity_weight = conflict_entity_weight
        self.conflict_triple_weight = conflict_triple_weight
        self.conflict_structure_weight = conflict_structure_weight
        self.conflict_semantic_weight = conflict_semantic_weight
        self.conflict_margin_weight = conflict_margin_weight
        self.conflict_penalty = conflict_penalty
        self.ambiguity_threshold = ambiguity_threshold

    def match(
        self,
        candidates_df: pd.DataFrame,
        mode: str = "baseline",
        graph1: Optional[Graph] = None,
        graph2: Optional[Graph] = None,
        entity_col1: str = "e1",
        entity_col2: str = "e2",
        sim_col: str = "sim",
        triple_sim_col: Optional[str] = None,
        entity_sem_prob_df: Optional[pd.DataFrame] = None,
        use_gpu: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Executes matching on a candidate DataFrame according to selected mode:
          - 'baseline': Fixed threshold (0.80), Semantic OFF, Adaptive OFF, Conflict OFF.
          - 'improvement-1' or 'semantic': Semantic ON, Adaptive OFF, Conflict OFF.
          - 'improvement-2' or 'adaptive': Semantic ON, Adaptive ON, Conflict OFF.
          - 'improvement-3', 'conflict', or 'final': Semantic ON, Adaptive ON, Conflict ON.
        """
        mode_clean = mode.lower().replace("_", "-")
        use_semantic = mode_clean in ["improvement-1", "semantic", "improvement-2", "adaptive", "improvement-3", "conflict", "final"]
        use_adaptive = mode_clean in ["improvement-2", "adaptive", "improvement-3", "conflict", "final"]
        use_conflict = mode_clean in ["improvement-3", "conflict", "final"]

        # Step 1: Threshold application (Baseline, Improvement 1, or Improvement 2 Adaptive)
        decisions_df = at.apply_adaptive_threshold(
            candidates_df=candidates_df,
            use_adaptive=use_adaptive,
            use_semantic_type_matching=use_semantic,
            base_threshold=self.base_threshold,
            min_threshold=self.min_threshold,
            max_threshold=self.max_threshold,
            uncertainty_weight=self.uncertainty_weight,
            confidence_weight=self.confidence_weight,
            margin_weight=self.margin_weight,
            triple_weight=self.triple_weight,
            semantic_bonus_weight=self.semantic_bonus_weight,
            graph1=graph1,
            graph2=graph2,
            entity_sem_prob_df=entity_sem_prob_df,
            entity_col1=entity_col1,
            entity_col2=entity_col2,
            sim_col=sim_col,
            triple_sim_col=triple_sim_col,
            use_gpu=use_gpu
        )

        conflict_log_df = pd.DataFrame()
        conflict_stats = {}

        # Step 2: Conflict resolution if enabled (Improvement 3 / Final)
        if use_conflict:
            decisions_df, conflict_log_df, conflict_stats = cr.detect_and_resolve_conflicts(
                adaptive_decisions_df=decisions_df,
                use_conflict_resolution=True,
                use_semantic_type_matching=use_semantic,
                entity_weight=self.conflict_entity_weight,
                triple_weight=self.conflict_triple_weight,
                structure_weight=self.conflict_structure_weight,
                semantic_weight=self.conflict_semantic_weight,
                margin_weight=self.conflict_margin_weight,
                conflict_penalty=self.conflict_penalty,
                ambiguity_threshold=self.ambiguity_threshold,
                graph1=graph1,
                graph2=graph2,
                entity_col1="entity_kg1",
                entity_col2="entity_kg2",
                sim_col="similarity",
                triple_sim_col="triple_similarity" if "triple_similarity" in decisions_df.columns else None,
                semantic_sim_col="semantic_type_similarity",
                semantic_match_col="semantic_type_match",
                decision_col="decision",
                use_gpu=use_gpu
            )

        return decisions_df, conflict_log_df, conflict_stats


def save_run_config(
    config: Dict[str, Any],
    run_dir: str,
    mode: str,
    use_gpu: bool
) -> str:
    """
    Saves run_config.json in the experiment root directory.
    """
    cfg_data = {
        "dataset": config.get("dataset_name", "custom"),
        "display_name": config.get("display_name", ""),
        "kg1": config.get("kg1"),
        "kg2": config.get("kg2"),
        "candidates": config.get("candidates"),
        "gold": config.get("gold"),
        "mode": mode,
        "gpu": use_gpu,
        "timestamp": datetime.datetime.now().isoformat()
    }
    cfg_path = os.path.join(run_dir, "run_config.json")
    save_json(cfg_data, cfg_path)
    return cfg_path


def save_experiment_results(
    config: Dict[str, Any],
    mode_name: str,
    decisions_df: pd.DataFrame,
    metrics: Dict[str, Any],
    conflict_log_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
    project_root_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Saves comprehensive mode results into structured directories:
    Primary: <output_dir>/<mode_name>/
    Backward-compatibility: results/full-triple-matcher/<dataset>/<mode_name>/
    """
    root = project_root_dir or get_project_root()
    dataset_name = config.get("dataset_name", "custom").lower()

    folder_map = {
        "baseline": "baseline",
        "improvement-1": "improvement-1",
        "semantic": "improvement-1",
        "improvement-2": "improvement-2",
        "adaptive": "improvement-2",
        "improvement-3": "improvement-3",
        "conflict": "improvement-3",
        "final": "final"
    }
    norm_folder = folder_map.get(mode_name.lower(), mode_name.lower())

    # Build primary target directory
    if output_dir:
        primary_dir = os.path.join(output_dir, norm_folder)
    else:
        primary_dir = os.path.join(root, "results", dataset_name, "latest", norm_folder)

    # Legacy compatibility directory
    legacy_dir = os.path.join(root, "results", "full-triple-matcher", dataset_name, norm_folder)

    saved_paths: Dict[str, str] = {}

    target_dirs = [primary_dir]
    if primary_dir != legacy_dir:
        target_dirs.append(legacy_dir)

    for out_d in target_dirs:
        ensure_dir(out_d)

        # 1. Matching decisions / predictions
        pred_path = os.path.join(out_d, "predictions.csv")
        save_csv(decisions_df, pred_path)
        # Legacy name
        save_csv(decisions_df, os.path.join(out_d, "matching_decisions.csv"))

        # 2. Accepted matches
        if "decision" in decisions_df.columns:
            accepted_df = decisions_df[decisions_df["decision"] == True].copy()
        else:
            accepted_df = decisions_df.copy()
        save_csv(accepted_df, os.path.join(out_d, "accepted_matches.csv"))

        # 3. Metrics JSON
        save_json(metrics, os.path.join(out_d, "metrics.json"))

        # 4. Metrics CSV (summary.csv and metrics.csv)
        sum_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
        save_csv(sum_df, os.path.join(out_d, "metrics.csv"))
        save_csv(sum_df, os.path.join(out_d, "summary.csv"))

        # 5. Runtime JSON
        runtime_data = {
            "mode": mode_name,
            "runtime_seconds": metrics.get("runtime_seconds", 0.0),
            "timestamp": datetime.datetime.now().isoformat()
        }
        save_json(runtime_data, os.path.join(out_d, "runtime.json"))

        # 6. Mode-specific artifact exports
        if norm_folder == "improvement-1":
            sem_cols = [c for c in ['entity_kg1', 'entity_kg2', 'similarity',
                                    'semantic_type_kg1', 'semantic_type_kg2',
                                    'semantic_type_similarity', 'semantic_type_match',
                                    'semantic_type_bonus', 'semantic_evidence_source',
                                    'decision'] if c in decisions_df.columns]
            if sem_cols:
                save_csv(decisions_df[sem_cols], os.path.join(out_d, "semantic_types.csv"))

        elif norm_folder == "improvement-2":
            cols = [c for c in ['entity_kg1', 'entity_kg2', 'similarity', 'candidate_margin',
                                'base_threshold', 'uncertainty_penalty', 'confidence_bonus',
                                'semantic_type_bonus', 'adaptive_threshold', 'decision'] if c in decisions_df.columns]
            if cols:
                save_csv(decisions_df[cols], os.path.join(out_d, "adaptive_thresholds.csv"))

        elif norm_folder in ["improvement-3", "final"]:
            if conflict_log_df is not None and not conflict_log_df.empty:
                save_csv(conflict_log_df, os.path.join(out_d, "conflict_resolution.csv"))

        if out_d == primary_dir:
            saved_paths["directory"] = primary_dir
            saved_paths["predictions"] = pred_path

    return saved_paths


def save_comparison_results(
    comp_df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
    project_root_dir: Optional[str] = None
):
    """
    Saves final 4-way comparison table in both CSV and JSON formats:
    - <output_dir>/comparison/four_way_comparison.csv
    - <output_dir>/comparison/four_way_comparison.json
    - results/<dataset>/comparison/four_way_comparison.csv
    - results/full-triple-matcher/<dataset>/comparison.csv (backward compatibility)
    """
    root = project_root_dir or get_project_root()
    dataset_name = config.get("dataset_name", "custom").lower()

    target_locations = []
    if output_dir:
        target_locations.append(os.path.join(output_dir, "comparison"))
    
    # Dataset-level persistent comparison
    target_locations.append(os.path.join(root, "results", dataset_name, "comparison"))
    # Legacy location
    target_locations.append(os.path.join(root, "results", "full-triple-matcher", dataset_name))

    for loc in target_locations:
        ensure_dir(loc)
        save_csv(comp_df, os.path.join(loc, "four_way_comparison.csv"))
        # Save JSON
        comp_records = comp_df.to_dict(orient="records")
        save_json(comp_records, os.path.join(loc, "four_way_comparison.json"))
        # Also save legacy comparison.csv name
        save_csv(comp_df, os.path.join(loc, "comparison.csv"))


def run_pipeline(
    dataset_name: Optional[str] = None,
    kg1_path: Optional[str] = None,
    kg2_path: Optional[str] = None,
    candidates_path: Optional[str] = None,
    gold_path: Optional[str] = None,
    mode: str = "final",
    output_dir: Optional[str] = None,
    project_root_dir: Optional[str] = None,
    use_gpu: bool = False,
    save_results: bool = True,
    run_id: Optional[str] = None,
    load_graphs: Optional[bool] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Central API to execute the Full-Triple-Matcher pipeline.
    Fully dataset-agnostic: supports registered benchmarks (movies, newmovies, dw-nb)
    or user-specified explicit KG1 / KG2 pairs.

    Parameters:
      dataset_name: Name of benchmark dataset or None if explicit KG pair.
      kg1_path: Path to KG1 RDF file (optional override).
      kg2_path: Path to KG2 RDF file (optional override).
      candidates_path: Path to candidate pairs CSV (optional, generated if omitted).
      gold_path: Path to gold standard alignments CSV (optional).
      mode: 'baseline', 'improvement-1', 'improvement-2', 'improvement-3', 'final', or 'all'.
      output_dir: Custom output directory for experiment artifacts.
      project_root_dir: Project root directory.
      use_gpu: Enable CUDA GPU acceleration if available.
      save_results: Whether to save artifacts to results/ directory.
      run_id: Unique run ID / timestamp string.
      load_graphs: Explicitly override graph parsing in memory.
      verbose: Print progress information.

    Returns:
      Dictionary containing metrics, decisions DataFrame, conflict stats, and run configuration.
    """
    root = project_root_dir or get_project_root()

    # Step 1: Centrally resolve configuration
    config = resolve_dataset_config(
        dataset_name=dataset_name,
        kg1=kg1_path,
        kg2=kg2_path,
        candidates=candidates_path,
        gold=gold_path,
        project_root_dir=root,
        load_graphs=load_graphs,
        output_dir=output_dir
    )

    # Validate dataset configuration
    has_gold = config.get("gold") is not None and os.path.exists(config.get("gold"))
    validate_dataset_config(config, require_candidates=False, require_gold=False, verbose=verbose)

    # Establish run ID and output directory
    active_run_id = run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_key = config["dataset_name"].lower()

    if output_dir:
        run_output_dir = output_dir
    else:
        run_output_dir = os.path.join(root, "results", ds_key, active_run_id)

    if verbose:
        gpu_info = gpu_utils.get_gpu_info()
        print("\n" + "=" * 65)
        print("FULL-TRIPLE-MATCHER (FTM) EXECUTION PIPELINE")
        print("=" * 65)
        print(f"Selected Dataset: {config.get('display_name')}")
        print(f"Dataset Name:     {config.get('dataset_name')}")
        print(f"Mode:             {mode.upper()}")
        print(f"KG1:              {config.get('kg1')}")
        print(f"KG2:              {config.get('kg2')}")
        print(f"Candidates:       {config.get('candidates') or '(Will be generated dynamically)'}")
        print(f"Gold Standard:    {config.get('gold') or '(None provided - matching only)'}")
        print(f"Output Directory: {run_output_dir}")
        print(f"GPU Requested:    {use_gpu} (CUDA Available: {gpu_info['cuda_available']})")
        print("=" * 65 + "\n")

    # Step 2: Load RDF Graphs if requested or needed for candidate generation
    g1, g2 = None, None
    needs_graphs = config.get("load_graphs", True) or (config.get("candidates") is None)
    
    if needs_graphs and config.get("kg1") and os.path.exists(config.get("kg1")):
        if verbose:
            print(f"[Loading KG1] Parsing {config['kg1']}...")
        g1 = load_rdf_graph(config["kg1"])
    if needs_graphs and config.get("kg2") and os.path.exists(config.get("kg2")):
        if verbose:
            print(f"[Loading KG2] Parsing {config['kg2']}...")
        g2 = load_rdf_graph(config["kg2"])

    # Step 3: Load or Generate Candidates
    cand_path = config.get("candidates")
    if cand_path and os.path.exists(cand_path):
        if verbose:
            print(f"[Candidates] Loading candidate pairs from {cand_path}...")
        cand_df = load_csv(cand_path)
    else:
        if g1 is not None and g2 is not None:
            if verbose:
                print("[Candidates] No candidate CSV provided. Generating candidate pairs from KG1 and KG2...")
            cand_df = cg.generate_candidates(graph1=g1, graph2=g2, min_similarity=0.30, top_k=5)
            if verbose:
                print(f"[Candidates] Generated {len(cand_df)} candidate pairs from graphs.")
        else:
            raise FileNotFoundError(
                f"Candidates file not found at '{cand_path}' and RDF graphs are not available to generate candidates."
            )

    # Step 4: Load Gold Data if available
    gold_df = None
    gold_p = config.get("gold")
    if gold_p and os.path.exists(gold_p):
        gold_df = load_csv(gold_p)

    # Step 5: Check auxiliary files (e.g. entity_sem_prob for large graphs)
    sem_prob_df = None
    if config.get("entity_sem_prob") and os.path.exists(config["entity_sem_prob"]):
        if verbose:
            print(f"[Semantic Probabilities] Loading {config['entity_sem_prob']}...")
        sem_prob_df = load_csv(config["entity_sem_prob"])

    pipeline = EntityMatcherPipeline()
    triple_col = 'triple_sim' if 'triple_sim' in cand_df.columns else None

    # Step 6: Execute Pipeline
    mode_clean = mode.lower().replace("_", "-")

    if mode_clean == "all":
        modes_to_run = ["baseline", "improvement-1", "improvement-2", "final"]
        all_metrics = {}
        all_results = {}

        if save_results:
            ensure_dir(run_output_dir)
            save_run_config(config, run_output_dir, mode="all", use_gpu=use_gpu)

        for m in modes_to_run:
            t0 = time.time()
            dec_df, conf_df, conf_stats = pipeline.match(
                candidates_df=cand_df,
                mode=m,
                graph1=g1,
                graph2=g2,
                triple_sim_col=triple_col,
                entity_sem_prob_df=sem_prob_df,
                use_gpu=use_gpu
            )
            m_runtime = time.time() - t0

            if gold_df is not None:
                m_metrics = evaluate_predictions(
                    predictions_df=dec_df,
                    gold_df=gold_df,
                    pred_col1="entity_kg1",
                    pred_col2="entity_kg2",
                    decision_col="decision",
                    sim_col="similarity"
                )
            else:
                accepted_count = len(dec_df[dec_df["decision"] == True]) if "decision" in dec_df.columns else len(dec_df)
                m_metrics = {
                    "total_gold": 0,
                    "predicted_matches": int(accepted_count),
                    "true_positives": None,
                    "false_positives": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "hit_at_1": None,
                    "hit_at_10": None,
                    "note": "Evaluation skipped: no gold standard supplied"
                }

            m_metrics["runtime_seconds"] = round(m_runtime, 4)
            m_metrics["dataset"] = config.get("dataset_name")
            m_metrics["mode"] = m

            all_metrics[m] = m_metrics
            all_results[m] = {
                "decisions": dec_df,
                "conflicts": conf_df,
                "conflict_stats": conf_stats,
                "metrics": m_metrics
            }

            if save_results:
                save_experiment_results(
                    config=config,
                    mode_name=m,
                    decisions_df=dec_df,
                    metrics=m_metrics,
                    conflict_log_df=conf_df,
                    output_dir=run_output_dir,
                    project_root_dir=root
                )

            if verbose:
                prec_str = f"{m_metrics['precision']:.4f}" if m_metrics['precision'] is not None else "N/A"
                rec_str = f"{m_metrics['recall']:.4f}" if m_metrics['recall'] is not None else "N/A"
                f1_str = f"{m_metrics['f1']:.4f}" if m_metrics['f1'] is not None else "N/A"
                print(f"Mode [{m:15}]: Precision={prec_str} | Recall={rec_str} | F1={f1_str} | Runtime={m_runtime:.3f}s")

        # Improvement 3 is alias of final
        all_metrics["improvement-3"] = all_metrics["final"]

        # Build 4-way comparison DataFrame
        comp_df = build_comparison_dataframe(
            baseline_metrics=all_metrics["baseline"],
            imp1_metrics=all_metrics["improvement-1"],
            imp2_metrics=all_metrics["improvement-2"],
            imp3_metrics=all_metrics["final"]
        )

        if save_results:
            save_comparison_results(comp_df, config, output_dir=run_output_dir, project_root_dir=root)

        if verbose:
            print("\n" + "=" * 65)
            print("FTM EXPERIMENT COMPLETE")
            print("=" * 65)
            print(f"Dataset:       {config.get('dataset_name', 'CUSTOM').upper()}")
            print(f"KG1:           {config.get('kg1')}")
            print(f"KG2:           {config.get('kg2')}")
            if gold_df is not None:
                print(f"Baseline F1:      {all_metrics['baseline']['f1']:.4f}")
                print(f"Improvement 1 F1: {all_metrics['improvement-1']['f1']:.4f}")
                print(f"Improvement 2 F1: {all_metrics['improvement-2']['f1']:.4f}")
                print(f"Final F1:         {all_metrics['final']['f1']:.4f}")
            print(f"\nResults saved to:\n{run_output_dir}")
            print("=" * 65)

        return {
            "dataset": config.get("dataset_name"),
            "run_id": active_run_id,
            "output_dir": run_output_dir,
            "config": config,
            "comparison_df": comp_df,
            "all_metrics": all_metrics,
            "results_by_mode": all_results,
            "metrics": all_metrics["final"]
        }

    else:
        # Single mode execution
        t0 = time.time()
        dec_df, conf_df, conf_stats = pipeline.match(
            candidates_df=cand_df,
            mode=mode_clean,
            graph1=g1,
            graph2=g2,
            triple_sim_col=triple_col,
            entity_sem_prob_df=sem_prob_df,
            use_gpu=use_gpu
        )
        runtime = time.time() - t0

        if gold_df is not None:
            metrics = evaluate_predictions(
                predictions_df=dec_df,
                gold_df=gold_df,
                pred_col1="entity_kg1",
                pred_col2="entity_kg2",
                decision_col="decision",
                sim_col="similarity"
            )
        else:
            accepted_count = len(dec_df[dec_df["decision"] == True]) if "decision" in dec_df.columns else len(dec_df)
            metrics = {
                "total_gold": 0,
                "predicted_matches": int(accepted_count),
                "true_positives": None,
                "false_positives": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "hit_at_1": None,
                "hit_at_10": None,
                "note": "Evaluation skipped: no gold standard supplied"
            }

        metrics["runtime_seconds"] = round(runtime, 4)
        metrics["dataset"] = config.get("dataset_name")
        metrics["mode"] = mode_clean

        accepted_df = dec_df[dec_df["decision"] == True].copy() if "decision" in dec_df.columns else dec_df.copy()

        if save_results:
            ensure_dir(run_output_dir)
            save_run_config(config, run_output_dir, mode=mode_clean, use_gpu=use_gpu)
            save_experiment_results(
                config=config,
                mode_name=mode_clean,
                decisions_df=dec_df,
                metrics=metrics,
                conflict_log_df=conf_df,
                output_dir=run_output_dir,
                project_root_dir=root
            )

        if verbose:
            print("\n" + "=" * 65)
            print("FTM EXPERIMENT COMPLETE")
            print("=" * 65)
            print(f"Dataset:       {config.get('dataset_name', 'CUSTOM').upper()}")
            print(f"Mode:          {mode_clean.upper()}")
            print(f"KG1:           {config.get('kg1')}")
            print(f"KG2:           {config.get('kg2')}")
            if gold_df is not None:
                print(f"Precision:     {metrics['precision']:.4f}")
                print(f"Recall:        {metrics['recall']:.4f}")
                print(f"F1-Score:      {metrics['f1']:.4f}")
            print(f"Matches:       {len(accepted_df)}")
            print(f"Runtime:       {runtime:.3f}s")
            print(f"\nResults saved to:\n{run_output_dir}")
            print("=" * 65)

        return {
            "dataset": config.get("dataset_name"),
            "mode": mode_clean,
            "run_id": active_run_id,
            "output_dir": run_output_dir,
            "config": config,
            "metrics": metrics,
            "decisions": dec_df,
            "accepted_matches": accepted_df,
            "conflicts": conf_df,
            "conflict_stats": conf_stats,
            "runtime_seconds": round(runtime, 4)
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full-Triple-Matcher (FTM) Dataset/KG-Agnostic Execution Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run named benchmark on all 4 modes:
  python codes/entity-matching/entity_matching.py --dataset movies --mode all

  # Run on GPU:
  python codes/entity-matching/entity_matching.py --dataset movies --mode all --gpu

  # Run with explicit KG files:
  python codes/entity-matching/entity_matching.py \\
      --kg1 dataset/movies/small_movie_kg1.ttl \\
      --kg2 dataset/movies/small_movie_kg2.ttl \\
      --candidates dataset/movies/movie_candidates.csv \\
      --gold dataset/movies/small_movie_gold.csv \\
      --mode all

  # List available datasets:
  python codes/entity-matching/entity_matching.py --list-datasets
"""
    )

    parser.add_argument("--dataset", type=str, default=None,
                        help="Name of dataset (e.g. 'movies', 'newmovies', 'dw-nb')")
    parser.add_argument("--kg1", type=str, default=None,
                        help="Path to KG1 RDF file (overrides dataset default or for explicit pair)")
    parser.add_argument("--kg2", type=str, default=None,
                        help="Path to KG2 RDF file (overrides dataset default or for explicit pair)")
    parser.add_argument("--candidates", type=str, default=None,
                        help="Path to candidates CSV (optional override)")
    parser.add_argument("--gold", type=str, default=None,
                        help="Path to gold standard alignments CSV (optional override)")
    parser.add_argument("--mode", choices=["baseline", "improvement-1", "improvement-2", "improvement-3", "final", "all"],
                        default="all", help="Execution mode (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output directory for experiment results")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable NVIDIA GPU acceleration (CUDA)")
    parser.add_argument("--list-datasets", action="store_true",
                        help="List all registered and discovered datasets and exit")

    args = parser.parse_args()

    if args.list_datasets:
        list_available_datasets(verbose=True)
        sys.exit(0)

    # Validate argument combination
    if not args.dataset and not (args.kg1 and args.kg2):
        parser.print_help()
        print("\n[ERROR] Either --dataset <name> OR (--kg1 <path> AND --kg2 <path>) must be specified.")
        sys.exit(1)

    # Execute pipeline
    result = run_pipeline(
        dataset_name=args.dataset,
        kg1_path=args.kg1,
        kg2_path=args.kg2,
        candidates_path=args.candidates,
        gold_path=args.gold,
        mode=args.mode,
        output_dir=args.output,
        use_gpu=args.gpu
    )

    if args.mode == "all" and "comparison_df" in result:
        print("\n--- Final 4-Way Comparison Table ---")
        print(result["comparison_df"].to_string(index=False))
