"""
Full-Triple-Matcher (FTM) — Master Demonstration & Evaluation Script.

Final-Year Research Project:
"Full Triple Matcher with Semantic Type Matching, Adaptive Threshold Optimization,
and Conflict Resolution"

Usage:
    # Run full 4-way evaluation on Movies benchmark:
    python demo_master_evaluation.py --dataset movies --mode all

    # Run on newMovies benchmark:
    python demo_master_evaluation.py --dataset newmovies --mode all

    # Run on DW-NB benchmark:
    python demo_master_evaluation.py --dataset dw-nb --mode all

    # Run with explicit Knowledge Graph files:
    python demo_master_evaluation.py --kg1 dataset/movies/small_movie_kg1.ttl --kg2 dataset/movies/small_movie_kg2.ttl

    # List all available datasets:
    python demo_master_evaluation.py --list-datasets
"""

import os
import sys
import argparse

# Ensure codes directories are in Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CODES_DIR = os.path.join(PROJECT_ROOT, "codes")
ENTITY_MATCHING_DIR = os.path.join(CODES_DIR, "entity-matching")
CONFIG_DIR = os.path.join(CODES_DIR, "config")
PIPELINE_DIR = os.path.join(CODES_DIR, "pipeline")
UTILS_DIR = os.path.join(CODES_DIR, "utils")

for p in [ENTITY_MATCHING_DIR, CONFIG_DIR, PIPELINE_DIR, UTILS_DIR, CODES_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from entity_matching import run_pipeline
from dataset_config import list_available_datasets, DATASETS


def main():
    parser = argparse.ArgumentParser(
        description="Full-Triple-Matcher (FTM) Final Research Project Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_master_evaluation.py --dataset movies --mode all
  python demo_master_evaluation.py --dataset newmovies --mode all
  python demo_master_evaluation.py --dataset dw-nb --mode baseline
  python demo_master_evaluation.py --list-datasets
  python demo_master_evaluation.py --kg1 dataset/movies/small_movie_kg1.ttl --kg2 dataset/movies/small_movie_kg2.ttl
        """
    )
    parser.add_argument("--dataset", type=str, default="movies",
                        help="Dataset name ('movies', 'newmovies', 'dw-nb', or auto-discovered)")
    parser.add_argument("--kg1", type=str, default=None,
                        help="Path to KG1 RDF file (for explicit KG mode)")
    parser.add_argument("--kg2", type=str, default=None,
                        help="Path to KG2 RDF file (for explicit KG mode)")
    parser.add_argument("--candidates", type=str, default=None,
                        help="Path to candidate pairs CSV (optional)")
    parser.add_argument("--gold", type=str, default=None,
                        help="Path to gold standard alignments CSV (optional)")
    parser.add_argument("--mode", choices=["baseline", "improvement-1", "improvement-2", "improvement-3", "final", "all"],
                        default="all", help="Evaluation mode (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output directory")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable CUDA acceleration if available")
    parser.add_argument("--list-datasets", action="store_true",
                        help="List available datasets and exit")

    args = parser.parse_args()

    if args.list_datasets:
        list_available_datasets(verbose=True)
        return

    # Check if explicit mode is requested
    if args.kg1 or args.kg2:
        if not (args.kg1 and args.kg2):
            print("[ERROR] Both --kg1 and --kg2 must be provided for explicit KG matching.")
            sys.exit(1)
        ds_arg = None
    else:
        ds_arg = args.dataset

    print("\n" + "=" * 70)
    print("DEMO: FULL-TRIPLE-MATCHER (FTM) FINAL MASTER EVALUATION")
    print("=" * 70)

    result = run_pipeline(
        dataset_name=ds_arg,
        kg1_path=args.kg1,
        kg2_path=args.kg2,
        candidates_path=args.candidates,
        gold_path=args.gold,
        mode=args.mode,
        output_dir=args.output,
        use_gpu=args.gpu,
        verbose=True
    )

    if args.mode == "all" and "comparison_df" in result:
        print("\n" + "=" * 70)
        print("MASTER 4-WAY COMPARISON TABLE")
        print("=" * 70)
        print(result["comparison_df"].to_string(index=False))


if __name__ == "__main__":
    main()
