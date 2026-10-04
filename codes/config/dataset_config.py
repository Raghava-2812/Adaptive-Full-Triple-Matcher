"""
Central Dataset and Knowledge Graph Configuration System for Full-Triple-Matcher (FTM).

Provides:
1. Registered benchmark datasets (Movies, newMovies, DW-NB).
2. Dynamic discovery of datasets in the dataset/ directory.
3. Central path resolution for both named datasets and explicit user-specified KG pairs.
4. Dataset configuration validation and sanity checking.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd


def get_project_root() -> str:
    """
    Locates the project root directory dynamically by searching upward
    for markers ('codes' and 'dataset').
    """
    # Start from current working directory or this file's location
    candidates = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    ]
    for start in candidates:
        curr = os.path.abspath(start)
        while curr and os.path.dirname(curr) != curr:
            if os.path.isdir(os.path.join(curr, "codes")) and os.path.isdir(os.path.join(curr, "dataset")):
                return curr
            curr = os.path.dirname(curr)
    
    # Fallback to 2 levels up from codes/config
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


PROJECT_ROOT = get_project_root()


# Central registry of benchmark datasets
DATASETS: Dict[str, Dict[str, Any]] = {
    "movies": {
        "name": "movies",
        "display_name": "Movies Knowledge Graph (Small)",
        "kg1": "dataset/movies/small_movie_kg1.ttl",
        "kg2": "dataset/movies/small_movie_kg2.ttl",
        "candidates": "dataset/movies/movie_candidates.csv",
        "gold": "dataset/movies/small_movie_gold.csv",
        "kg1_inv": "dataset/movies/kg1_inv.csv",
        "kg2_inv": "dataset/movies/kg2_inv.csv",
        "load_graphs": True,
        "description": "Standard domain-specific movie knowledge graph alignment benchmark."
    },
    "newmovies": {
        "name": "newmovies",
        "display_name": "newMovies Adversarial Benchmark",
        "kg1": "dataset/newMovies/kg1.ttl",
        "kg2": "dataset/newMovies/kg2.ttl",
        "candidates": "dataset/newMovies/candidates.csv",
        "gold": "dataset/newMovies/gold.csv",
        "kg1_inv": "dataset/newMovies/kg1_inv.csv",
        "kg2_inv": "dataset/newMovies/kg2_inv.csv",
        "load_graphs": True,
        "description": "Adversarial benchmark featuring distractor entities, polysemy, and 1-to-many variants."
    },
    "dw-nb": {
        "name": "dw-nb",
        "display_name": "DW-NB Benchmark (DBpedia - Wikidata)",
        "kg1": "dataset/dw-nb/kg1.ttl",
        "kg2": "dataset/dw-nb/kg2.ttl",
        "candidates": "results/full-triple-matcher/dw-nb/entity_sim_0.csv",
        "gold": "dataset/dw-nb/gold.csv",
        "kg1_inv": "dataset/dw-nb/kg1_inv.csv",
        "kg2_inv": "dataset/dw-nb/kg2_inv.csv",
        "entity_sem_prob": "results/full-triple-matcher/dw-nb/entity_sem_prob.csv",
        "load_graphs": False,  # Large graph: uses precomputed similarity table & URI heuristics
        "description": "Large-scale DBpedia to Wikidata alignment benchmark (666k candidate pairs, 25k gold alignments)."
    }
}

DATASET_ALIASES: Dict[str, str] = {
    "movie": "movies",
    "small_movie": "movies",
    "small_movies": "movies",
    "new_movies": "newmovies",
    "new-movies": "newmovies",
    "newmovie": "newmovies",
    "dwnb": "dw-nb",
    "dw_nb": "dw-nb",
    "dbpedia-wikidata": "dw-nb",
    "dbpedia_wikidata": "dw-nb"
}


def normalize_dataset_name(name: Optional[str]) -> Optional[str]:
    """
    Normalizes dataset name using standard casing and aliases.
    """
    if not name:
        return None
    cleaned = str(name).strip().lower().replace("_", "-")
    alias_clean = DATASET_ALIASES.get(cleaned, cleaned)
    if alias_clean in DATASETS:
        return alias_clean
    # Try with underscore replacement
    alt = cleaned.replace("-", "")
    for k in DATASETS:
        if k.replace("-", "") == alt:
            return k
    return cleaned


def list_available_datasets(
    dataset_dir: Optional[str] = None,
    project_root_dir: Optional[str] = None,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Lists all available datasets by combining registered benchmarks with
    auto-discovered datasets inside the dataset/ directory.
    """
    root = project_root_dir or get_project_root()
    base_ds_dir = dataset_dir or os.path.join(root, "dataset")

    available: Dict[str, Dict[str, Any]] = {}

    # 1. Add registered datasets
    for key, cfg in DATASETS.items():
        entry = dict(cfg)
        entry["registered"] = True
        entry["source"] = "predefined"
        available[key] = entry

    # 2. Discover subdirectories in dataset/
    if os.path.isdir(base_ds_dir):
        for entry in os.listdir(base_ds_dir):
            sub_path = os.path.join(base_ds_dir, entry)
            if not os.path.isdir(sub_path):
                continue

            norm_name = normalize_dataset_name(entry)
            if norm_name in available:
                continue

            # Auto-discover files in new dataset folder
            discovered_kg1 = None
            discovered_kg2 = None
            discovered_cand = None
            discovered_gold = None

            files = os.listdir(sub_path)
            for f in files:
                f_lower = f.lower()
                rel_path = os.path.relpath(os.path.join(sub_path, f), root)
                if ("kg1" in f_lower or "source" in f_lower) and f_lower.endswith((".ttl", ".nt", ".rdf", ".xml")):
                    discovered_kg1 = rel_path
                elif ("kg2" in f_lower or "target" in f_lower) and f_lower.endswith((".ttl", ".nt", ".rdf", ".xml")):
                    discovered_kg2 = rel_path
                elif ("candidate" in f_lower or "cand" in f_lower or "sim" in f_lower) and f_lower.endswith(".csv"):
                    discovered_cand = rel_path
                elif ("gold" in f_lower or "ground" in f_lower or "truth" in f_lower or "ref" in f_lower) and f_lower.endswith(".csv"):
                    discovered_gold = rel_path

            available[norm_name] = {
                "name": norm_name,
                "display_name": f"{entry} (Auto-Discovered)",
                "kg1": discovered_kg1,
                "kg2": discovered_kg2,
                "candidates": discovered_cand,
                "gold": discovered_gold,
                "load_graphs": True,
                "registered": False,
                "source": "auto-discovered",
                "description": f"Auto-discovered dataset in {sub_path}"
            }

    result_list = list(available.values())

    if verbose:
        print("=" * 65)
        print("AVAILABLE DATASETS FOR FULL-TRIPLE-MATCHER:")
        print("=" * 65)
        for i, d in enumerate(result_list, 1):
            reg = "[Registered]" if d.get("registered") else "[Discovered]"
            print(f"{i}. {d['name']:<15} {reg:<14} -> {d['display_name']}")
            print(f"   KG1: {d.get('kg1')}")
            print(f"   KG2: {d.get('kg2')}")
            print(f"   Candidates: {d.get('candidates')}")
            print(f"   Gold: {d.get('gold')}\n")

    return result_list


def resolve_path(path: Optional[str], project_root_dir: str) -> Optional[str]:
    """
    Resolves relative path to absolute path relative to project root.
    """
    if not path:
        return None
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(project_root_dir, path))


def resolve_dataset_config(
    dataset_name: Optional[str] = None,
    kg1: Optional[str] = None,
    kg2: Optional[str] = None,
    candidates: Optional[str] = None,
    gold: Optional[str] = None,
    project_root_dir: Optional[str] = None,
    load_graphs: Optional[bool] = None,
    output_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Centrally resolves and prepares the configuration dictionary for an FTM run.
    Supports two major input paradigms:
      1. Named benchmark dataset (e.g. 'movies', 'newmovies', 'dw-nb').
      2. Explicit Knowledge Graph pair (user supplies kg1 and kg2 directly).

    Parameters:
      dataset_name: Name of registered or discovered dataset.
      kg1: Explicit path to KG1 RDF file (overrides dataset default).
      kg2: Explicit path to KG2 RDF file (overrides dataset default).
      candidates: Explicit path to candidates CSV (overrides dataset default).
      gold: Explicit path to gold standard CSV (overrides dataset default).
      project_root_dir: Custom project root directory.
      load_graphs: Whether to parse RDF graphs into memory (defaults per dataset).
      output_dir: Custom output directory for experiment artifacts.

    Returns:
      A fully resolved configuration dictionary with absolute paths and metadata.
    """
    root = project_root_dir or get_project_root()
    norm_name = normalize_dataset_name(dataset_name)

    is_explicit = (norm_name is None) and (kg1 is not None and kg2 is not None)

    if not norm_name and not is_explicit:
        raise ValueError(
            "Neither a valid dataset_name nor explicit kg1/kg2 paths were provided.\n"
            f"Available registered datasets: {list(DATASETS.keys())}\n"
            "Or specify: kg1_path='path/to/kg1.ttl', kg2_path='path/to/kg2.ttl'"
        )

    # Base configuration from registered or discovered datasets
    base_cfg: Dict[str, Any] = {}
    if norm_name:
        if norm_name in DATASETS:
            base_cfg = dict(DATASETS[norm_name])
        else:
            # Check auto-discovered datasets
            discovered = {d["name"]: d for d in list_available_datasets(project_root_dir=root)}
            if norm_name in discovered:
                base_cfg = dict(discovered[norm_name])
            elif not (kg1 and kg2):
                raise ValueError(
                    f"Unknown dataset '{dataset_name}'. "
                    f"Available datasets: {list(DATASETS.keys()) + [d for d in discovered if d not in DATASETS]}"
                )
            else:
                base_cfg = {
                    "name": norm_name,
                    "display_name": f"Custom Dataset ({dataset_name})",
                    "load_graphs": True
                }
    else:
        # Explicit KG mode without dataset name
        kg1_base = os.path.splitext(os.path.basename(kg1))[0]
        kg2_base = os.path.splitext(os.path.basename(kg2))[0]
        base_cfg = {
            "name": f"custom_{kg1_base}_{kg2_base}",
            "display_name": f"Explicit KG Pair: {kg1_base} vs {kg2_base}",
            "load_graphs": True
        }

    # Apply explicit overrides if provided
    final_kg1 = kg1 or base_cfg.get("kg1")
    final_kg2 = kg2 or base_cfg.get("kg2")
    final_cand = candidates or base_cfg.get("candidates")
    final_gold = gold or base_cfg.get("gold")
    final_load_graphs = load_graphs if load_graphs is not None else base_cfg.get("load_graphs", True)

    resolved_config: Dict[str, Any] = {
        "dataset_name": base_cfg.get("name", "custom"),
        "display_name": base_cfg.get("display_name", "Custom KG Pair"),
        "is_explicit_pair": is_explicit,
        "project_root": root,
        "kg1": resolve_path(final_kg1, root),
        "kg2": resolve_path(final_kg2, root),
        "candidates": resolve_path(final_cand, root),
        "gold": resolve_path(final_gold, root),
        "load_graphs": final_load_graphs,
        "kg1_inv": resolve_path(base_cfg.get("kg1_inv"), root),
        "kg2_inv": resolve_path(base_cfg.get("kg2_inv"), root),
        "entity_sem_prob": resolve_path(base_cfg.get("entity_sem_prob"), root),
        "output_dir": resolve_path(output_dir, root) if output_dir else None,
        "raw_inputs": {
            "dataset_name": dataset_name,
            "kg1": kg1,
            "kg2": kg2,
            "candidates": candidates,
            "gold": gold
        }
    }

    return resolved_config


def validate_dataset_config(
    config: Dict[str, Any],
    require_candidates: bool = False,
    require_gold: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validates that files in the configuration exist, are readable, and conform to expected formats.
    Returns a dictionary summarizing validation status, warnings, and file statistics.
    """
    errors: List[str] = []
    warnings: List[str] = []
    info: Dict[str, Any] = {}

    # Check KG1
    kg1_path = config.get("kg1")
    if not kg1_path:
        errors.append("KG1 path is not specified in configuration.")
    elif not os.path.exists(kg1_path):
        errors.append(f"KG1 file not found at: {kg1_path}")
    elif not os.path.isfile(kg1_path):
        errors.append(f"KG1 path is not a file: {kg1_path}")
    else:
        info["kg1_size_bytes"] = os.path.getsize(kg1_path)

    # Check KG2
    kg2_path = config.get("kg2")
    if not kg2_path:
        errors.append("KG2 path is not specified in configuration.")
    elif not os.path.exists(kg2_path):
        errors.append(f"KG2 file not found at: {kg2_path}")
    elif not os.path.isfile(kg2_path):
        errors.append(f"KG2 path is not a file: {kg2_path}")
    else:
        info["kg2_size_bytes"] = os.path.getsize(kg2_path)

    # Check Candidates
    cand_path = config.get("candidates")
    if cand_path:
        if not os.path.exists(cand_path):
            if require_candidates:
                errors.append(f"Candidates file required but not found at: {cand_path}")
            else:
                warnings.append(f"Candidates file not found at {cand_path}; candidates will be generated from KGs.")
        else:
            try:
                # Preview top lines
                sample = pd.read_csv(cand_path, nrows=5)
                cols = list(sample.columns)
                info["candidates_columns"] = cols
                has_entity_cols = any(c in cols for c in ['e1', 'entity_kg1', 'src', 'source']) and \
                                  any(c in cols for c in ['e2', 'entity_kg2', 'tgt', 'target'])
                if not has_entity_cols:
                    warnings.append(f"Candidates file {cand_path} missing standard entity columns ('e1'/'e2'). Columns: {cols}")
            except Exception as e:
                errors.append(f"Candidates file {cand_path} could not be read as CSV: {e}")
    elif require_candidates:
        errors.append("Candidates file is required for this experiment but none was specified.")
    else:
        info["candidates_source"] = "will_be_generated_from_graphs"

    # Check Gold
    gold_path = config.get("gold")
    if gold_path:
        if not os.path.exists(gold_path):
            if require_gold:
                errors.append(f"Gold standard file required but not found at: {gold_path}")
            else:
                warnings.append(f"Gold standard file not found at {gold_path}; evaluation metrics cannot be computed.")
        else:
            try:
                sample_gold = pd.read_csv(gold_path, nrows=5)
                cols_gold = list(sample_gold.columns)
                info["gold_columns"] = cols_gold
                has_gold_cols = any(c in cols_gold for c in ['e1', 'entity_kg1', 'src']) and \
                                any(c in cols_gold for c in ['e2', 'entity_kg2', 'tgt'])
                if not has_gold_cols:
                    warnings.append(f"Gold file {gold_path} missing expected entity columns ('e1'/'e2'). Columns: {cols_gold}")
            except Exception as e:
                errors.append(f"Gold standard file {gold_path} could not be read as CSV: {e}")
    elif require_gold:
        errors.append("Gold standard alignment file is required for evaluation but none was specified.")

    is_valid = len(errors) == 0

    if verbose:
        print(f"Validation Status: {'PASSED' if is_valid else 'FAILED'}")
        if errors:
            print("Errors:")
            for err in errors:
                print(f"  - [ERROR] {err}")
        if warnings:
            print("Warnings:")
            for w in warnings:
                print(f"  - [WARNING] {w}")

    if not is_valid:
        raise ValueError(f"Dataset configuration validation failed:\n" + "\n".join(f"- {e}" for e in errors))

    return {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "file_info": info
    }
