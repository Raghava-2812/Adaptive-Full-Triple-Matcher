"""
Normalization Utilities for Full-Triple-Matcher (FTM)
Provides numerical clamping, feature scaling, and score composition.
"""

from typing import Iterable, List, Optional, Union
import numpy as np
import pandas as pd


def clamp_score(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Clamps a numeric score strictly within [min_val, max_val].
    """
    try:
        val_f = float(val)
        if np.isnan(val_f):
            return min_val
        return max(min_val, min(max_val, val_f))
    except (ValueError, TypeError):
        return min_val


def min_max_scale(series: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
    """
    Scales an array/series of values to the [0.0, 1.0] range.
    Handles uniform/degenerate distributions gracefully by returning zeros.
    """
    arr = np.asarray(series, dtype=float)
    if len(arr) == 0:
        return arr
    min_v = np.nanmin(arr)
    max_v = np.nanmax(arr)
    if np.isclose(max_v, min_v):
        return np.zeros_like(arr)
    scaled = (arr - min_v) / (max_v - min_v)
    return np.nan_to_num(scaled, nan=0.0)


def calculate_score_margin(best_score: float, second_best_score: float) -> float:
    """
    Calculates the margin between best and second best candidate similarity scores.
    Margin = max(0.0, best_score - second_best_score).
    """
    best = clamp_score(best_score)
    second = clamp_score(second_best_score)
    return max(0.0, min(1.0, best - second))


def compute_weighted_sum(
    values: List[float],
    weights: List[float],
    penalty: float = 0.0
) -> float:
    """
    Calculates a linear weighted combination of scores minus an optional penalty.
    Clamps final result to [0.0, 1.0].
    """
    total = sum(v * w for v, w in zip(values, weights)) - penalty
    return clamp_score(total, 0.0, 1.0)
