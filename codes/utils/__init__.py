"""
Utility package for Full-Triple-Matcher (FTM)
"""
from .preprocessing import clean_text, extract_local_name, tokenize, parse_date_value, extract_entity_label_from_graph
from .file_utils import ensure_dir, load_rdf_graph, load_csv, save_csv, load_json, save_json
from .normalization import clamp_score, min_max_scale, calculate_score_margin, compute_weighted_sum
from .gpu_utils import (
    is_cuda_available,
    get_gpu_info,
    get_device,
    clear_gpu_cache,
    gpu_batch_similarity,
    gpu_accelerate_adaptive_threshold,
    gpu_accelerate_composite_scores
)
