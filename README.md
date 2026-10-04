# Full-Triple-Matcher (FTM)

## Integrating All Triple Elements Between Heterogeneous Knowledge Graphs with Semantic Type Matching, Adaptive Threshold Optimization, and Conflict Resolution

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.22914-b31b1b.svg)](https://arxiv.org/abs/2507.22914)

---

## 1. Overview & Research Context

Knowledge Graph (KG) integration is essential for connecting general-purpose knowledge sources (e.g., DBpedia, Wikidata, YAGO) with domain-specific repositories. While traditional entity matching techniques align entities solely based on surface label similarities or isolated graph neighborhoods, the **Full-Triple-Matcher (FTM)** framework aligns all three triple elements: **subjects, predicates, and objects**.

### Base Paper Citation
> **"Full Triple Matcher: Integrating All Triple Elements between Heterogeneous Knowledge Graphs"**  
> *Victor Eiti Yamamoto and Hideaki Takeda*  
> **ACM Transactions on the Web (TWEB)**, Vol. 19, No. 4, Article 48, 2025.  
> DOI: [10.1145/3754338](https://doi.org/10.1145/3754338) | arXiv: [2507.22914](https://arxiv.org/abs/2507.22914)  
> *A copy of the paper is included in `paper/Full-Triple-Matcher.pdf`.*

---

## 2. Project File Structure

```
Full-Triple-Matcher-2/
│
├── dataset/
│   ├── movies/
│   │   ├── small_movie_kg1.ttl
│   │   ├── small_movie_kg2.ttl
│   │   ├── movie_candidates.csv
│   │   ├── small_movie_gold.csv
│   │   ├── kg1_inv.csv
│   │   └── kg2_inv.csv
│   │
│   ├── newMovies/
│   │   ├── kg1.ttl
│   │   ├── kg2.ttl
│   │   ├── candidates.csv
│   │   ├── gold.csv
│   │   ├── kg1_inv.csv
│   │   └── kg2_inv.csv
│   │
│   └── dw-nb/
│       ├── kg1.ttl
│       ├── kg2.ttl
│       ├── gold.csv
│       ├── kg1_inv.csv
│       └── kg2_inv.csv
│
├── notebooks/
│   ├── FTM_Final_Project.ipynb                   <-- Single Final Research Notebook for Defense & Demo
│   ├── 01_Baseline_FTM.ipynb
│   ├── 02_Improvement_1_Semantic_Type.ipynb
│   ├── 02_Improvement_2_Adaptive_Threshold.ipynb
│   ├── 03_Improvement_3_Conflict_Resolution.ipynb
│   ├── 04_Final_Improved_FTM.ipynb
│   └── 05_Test_newMovies_Pipeline.ipynb
│
├── codes/
│   ├── config/
│   │   ├── __init__.py
│   │   └── dataset_config.py                      <-- Central Dataset & KG Configuration System
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── runner.py                              <-- Clean High-Level Pipeline API
│   │
│   ├── entity-matching/
│   │   ├── entity_matching.py                     <-- Core Pipeline Orchestrator & CLI Runner
│   │   ├── candidate_generation.py                <-- Graph Entity Extraction & Candidate Generation
│   │   ├── semantic_type_matching.py              <-- Improvement 1: Semantic Type-Aware Matching
│   │   ├── adaptive_threshold.py                  <-- Improvement 2: Adaptive Threshold Optimization
│   │   ├── conflict_resolution.py                 <-- Improvement 3: Automated Conflict Resolution
│   │   ├── predicate_matching.py                  <-- Predicate Alignment & Inversability
│   │   ├── triple_matching.py                     <-- Full Triple & Datatype-Aware Similarity
│   │   └── evaluation.py                          <-- Precision, Recall, F1, Hit@k & Comparison Reporting
│   │
│   └── utils/
│       ├── gpu_utils.py                           <-- CUDA GPU Detection & Tensor Acceleration
│       ├── preprocessing.py                       <-- URI Parsing & Entity Extraction
│       ├── file_utils.py                          <-- Safe RDF/CSV/JSON I/O
│       └── normalization.py                       <-- Score Clamping & Normalization
│
├── results/
│   ├── movies/
│   ├── newmovies/
│   └── dw-nb/
│
├── demo_master_evaluation.py                      <-- Top-level demonstration and benchmark CLI
├── paper/
│   └── Full-Triple-Matcher.pdf
├── environment.yml
└── README.md
```

---

## 3. Methodological Pipeline & Architectural Innovations

The system executes as a unified, dataset/KG-agnostic research pipeline:

```
[Selected Benchmark or Explicit KG Pair]
                    │
                    ▼
          [Candidate Generation]
                    │
                    ▼
          [Stage 1: Baseline FTM] ────────── Fixed Global Threshold (θ = 0.80)
                    │
                    ▼
          [Stage 2: Improvement 1] ───────── Semantic Type-Aware Matching
                                             (Explicit RDF Types + Structural Domain Roles + Synonym Clusters)
                    │
                    ▼
          [Stage 3: Improvement 2] ───────── Adaptive Threshold Optimization
                                             (Uncertainty Penalty + Margin/Triple Confidence + Semantic Bonus)
                    │
                    ▼
          [Stage 4: Improvement 3 / Final] ── Automated Conflict Resolution
                                             (1-to-Many, Many-to-1, Competing Candidate Arbitration)
                    │
                    ▼
          [Stage 5: Final Evaluation & Saving] ── Reproducible Results in results/<dataset>/<run_id>/
```

### Stage 1: Baseline FTM
- Evaluates candidate entity pairs using string similarity and structural triple evidence.
- Predicate functionality and inverse functionality weight relational importance:
  $$\text{functionality}(p) = \frac{|\{s \mid (s, p, o) \in G\}|}{|\{(s, p, o) \in G\}|}$$
  $$\text{inverse\_functionality}(p) = \frac{|\{o \mid (s, p, o) \in G\}|}{|\{(s, p, o) \in G\}|}$$
  $$\text{inversability}(p) = \text{functionality}(p) \times \text{inverse\_functionality}(p)$$
- Matches are determined by a static global threshold ($\theta_{base} = 0.80$).

### Stage 2: Improvement 1 — Semantic Type-Aware Matching
- Extracts explicit types (`rdf:type`, `wdt:P31`, `schema:additionalType`, etc.), relational role signatures, and URI disambiguation patterns.
- Projects categories into cross-ontology clusters (e.g. Movie $\leftrightarrow$ Film, Director $\leftrightarrow$ DirectedBy).
- Computes normalized semantic compatibility and awards a bounded bonus $\text{Bonus}_{sem} \in [0.0, 0.10]$ to compatible alignments while blocking cross-category false positives (e.g., Actor vs. Director or Movie vs. City).

### Stage 3: Improvement 2 — Adaptive Threshold Optimization
Replaces the static cutoff with a candidate-dependent dynamic threshold:
$$\theta_{adaptive} = \text{clamp}\Big(\theta_{base} + \text{Uncertainty Penalty} - \text{Confidence Bonus} - \text{Semantic Bonus}, \theta_{min}, \theta_{max}\Big)$$
Where:
- $\text{Uncertainty Penalty} = w_{unc} \times (1.0 - \text{Margin})$
- $\text{Confidence Bonus} = (w_{conf} \times \text{sim}) + (w_{margin} \times \text{Margin}) + (w_{triple} \times \text{TripleSim})$
- $\text{Semantic Bonus} = w_{sem} \times \text{SemanticBonus}$
- *Effect:* Recovers valid true positives lying just below 0.80 without inflating false positives.

### Stage 4: Improvement 3 / Final — Automated Conflict Resolution
Detects and resolves 4 conflict types:
1. **One-to-Many Conflicts:** $|\text{targets}(e_1)| > 1$
2. **Many-to-One Conflicts:** $|\text{sources}(e_2)| > 1$
3. **Competing Candidate Conflicts:** $\text{Margin}(e_1) < \epsilon_{margin}$
4. **Structural Conflicts:** High string similarity with divergent graph neighborhoods.

Matches are adjudicated using a **Composite Resolution Score**:
$$\text{Resolution Score} = \Big(0.35 \cdot \text{sim} + 0.25 \cdot \text{triple\_sim} + 0.15 \cdot \text{struct} + 0.15 \cdot \text{sem\_compat} + 0.10 \cdot \text{Margin}\Big) - \text{Penalty}$$
- Weaker alternatives are systematically pruned with complete audit logging.
- Genuinely ambiguous candidates ($|\Delta| < 0.02$) are marked `UNRESOLVED` rather than forcing incorrect matches.

---

## 4. Running the Final FTM Pipeline

The refactored pipeline supports both **named benchmark datasets** and **explicit user-provided KG files**.

### A. Command-Line Interface (CLI)

Run via `entity_matching.py` or the top-level `demo_master_evaluation.py`:

```bash
# 1. Run full 4-way evaluation on Movies benchmark (baseline, imp-1, imp-2, final):
python codes/entity-matching/entity_matching.py --dataset movies --mode all

# 2. Run with GPU acceleration (CUDA):
python codes/entity-matching/entity_matching.py --dataset movies --mode all --gpu

# 3. Run on newMovies adversarial benchmark:
python codes/entity-matching/entity_matching.py --dataset newmovies --mode all

# 4. Run on large-scale DW-NB benchmark (DBpedia - Wikidata):
python codes/entity-matching/entity_matching.py --dataset dw-nb --mode all

# 5. Run using EXPLICIT Knowledge Graph files (dataset-agnostic mode):
python codes/entity-matching/entity_matching.py \
    --kg1 dataset/movies/small_movie_kg1.ttl \
    --kg2 dataset/movies/small_movie_kg2.ttl \
    --candidates dataset/movies/movie_candidates.csv \
    --gold dataset/movies/small_movie_gold.csv \
    --mode all

# 6. Run with explicit KGs without candidate file (dynamic graph candidate generation):
python codes/entity-matching/entity_matching.py \
    --kg1 dataset/movies/small_movie_kg1.ttl \
    --kg2 dataset/movies/small_movie_kg2.ttl \
    --mode final

# 7. List all available registered and auto-discovered datasets:
python codes/entity-matching/entity_matching.py --list-datasets
```

Or using `demo_master_evaluation.py` at the project root:

```bash
python demo_master_evaluation.py --dataset movies --mode all
python demo_master_evaluation.py --dataset newmovies --mode all
python demo_master_evaluation.py --list-datasets
```

---

### B. Python API

Use the clean, centralized API in your scripts or external tools:

```python
from codes.pipeline import run_pipeline

# 1. Named benchmark run:
results = run_pipeline(
    dataset_name="movies",
    mode="all",        # 'baseline', 'improvement-1', 'improvement-2', 'final', or 'all'
    use_gpu=True
)
print(results["comparison_df"])

# 2. Explicit Knowledge Graph pair run:
custom_results = run_pipeline(
    kg1_path="dataset/movies/small_movie_kg1.ttl",
    kg2_path="dataset/movies/small_movie_kg2.ttl",
    candidates_path="dataset/movies/movie_candidates.csv",
    gold_path="dataset/movies/small_movie_gold.csv",
    mode="final",
    output_dir="results/custom_run"
)
print("F1-Score:", custom_results["metrics"]["f1"])
```

---

### C. Final Research Notebook

Open and run the master demonstration notebook:
```text
notebooks/FTM_Final_Project.ipynb
```
This notebook contains:
1. System title and architecture walkthrough.
2. Central configuration cell (`DATASET_NAME = "movies"`, `RUN_ALL_MODES = True`, `USE_GPU = True`).
3. Dataset validation and input file verification.
4. Generic data loading.
5. End-to-end 4-way evaluation.
6. 4-way comparison table and research findings.
7. Explainability deep-dive for each improvement.
8. Explicit KG pair demonstration without requiring a dataset name.
9. Audit log and artifact verification.

---

## 5. Experiment Results & Reproducibility

Every experiment automatically writes structured, reproducible outputs:

```text
results/
    <dataset_name>/
        <timestamp_or_run_id>/
            run_config.json              <-- Complete configuration and metadata
            baseline/
                metrics.json             <-- Precision, Recall, F1, Hit@k
                metrics.csv              <-- Summary table
                predictions.csv          <-- All candidates with scores and decisions
                accepted_matches.csv     <-- Final accepted 1-to-1 alignments
                runtime.json             <-- Execution timing
            improvement-1/
                ... + semantic_types.csv <-- Ontological types, similarity, bonuses
            improvement-2/
                ... + adaptive_thresholds.csv <-- Dynamic thresholds & margins
            final/ (improvement-3)
                ... + conflict_resolution.csv <-- Audit log of resolved conflicts
            comparison/
                four_way_comparison.csv  <-- Master 4-way table
                four_way_comparison.json <-- JSON serialization of comparative metrics
```

### Verified Benchmark Metrics

#### Movie Knowledge Graph Benchmark
| Metric | Baseline | Improvement 1 (Semantic) | Improvement 2 (Adaptive) | Improvement 3 / Final |
|---|---|---|---|---|
| **Predicted Matches** | 18 | 18 | 21 | 17 |
| **True Positives** | 18 | 18 | 20 | 16 |
| **False Positives** | 0 | 0 | 1 | 1 |
| **Precision** | 1.0000 | 1.0000 | 0.9524 | 0.9412 |
| **Recall** | 0.8182 | 0.8182 | **0.9091** | 0.7273 |
| **F1-Score** | 0.9000 | 0.9000 | **0.9302** | 0.8205 |
| **Hit@1** | 0.9500 | 0.9500 | 0.9500 | 0.9500 |
| **Hit@10** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

#### newMovies Adversarial Benchmark
| Metric | Baseline | Improvement 1 (Semantic) | Improvement 2 (Adaptive) | Improvement 3 / Final |
|---|---|---|---|---|
| **Predicted Matches** | 27 | 28 | 29 | 28 |
| **True Positives** | 26 | 28 | 29 | 28 |
| **False Positives** | 1 | 0 | 0 | 0 |
| **Precision** | 0.9630 | 1.0000 | 1.0000 | 1.0000 |
| **Recall** | 0.6667 | 0.7179 | **0.7436** | 0.7179 |
| **F1-Score** | 0.7879 | 0.8358 | **0.8529** | 0.8358 |
| **Hit@1** | 0.8438 | 0.8438 | 0.8438 | 0.8438 |
| **Hit@10** | 0.9688 | 0.9688 | 0.9688 | 0.9688 |

---

## 6. GPU Acceleration (CUDA / PyTorch)

For large Knowledge Graphs, the framework includes a vectorized tensor acceleration engine (`codes/utils/gpu_utils.py`):
- **Automatic Device Discovery**: Detects CUDA availability, device count, compute capability, and memory budget.
- **Batched Tensor Cosine Similarity**: Uses chunked matrix multiplication on CUDA (`torch.mm`) with automatic cache clearing.
- **Tensor Adaptive Thresholds**: Vectorizes uncertainty penalties, confidence bonuses, and margin clamping.
- **Graceful Fallback**: If CUDA is unavailable, the pipeline transparently falls back to vectorized NumPy/SciPy CPU computation.

---

## 7. Environment Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate full-triple-matcher
```

---

## 8. License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See [GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.en.html) for full details.