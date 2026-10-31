"""
Selection metrics: each turns the validation folds' scores of one hyperparameter
combination into one number, higher is better; the best combination is selected.

A metric is built from quantities, each a value per fold: joint_avg from the four
recovery |r| (EEG and fNIRS, per source). Each quantity is averaged over the folds on
its own (through Fisher's z for correlations, which is how repeated estimates of one
correlation are averaged), and the metric is the plain mean of those averages, since
different quantities are not estimates of one correlation. A combination is only
compared when it was judged on every fold: a failed fold or a missing value excludes it.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.types import Scores


@dataclass(frozen=True)
class Metric:
    """
    Attributes
    ----------
    quantities : Callable[[Scores], dict[str, float]]
        The metric's quantities in one fold.
    fisher : bool
        Average each quantity over the folds through Fisher's z (correlations).
    needs_ground_truth : bool
        Synthetic data only.
    """
    quantities: Callable[[Scores], dict[str, float]]
    fisher: bool
    needs_ground_truth: bool


METRICS = {
    # |r| of each labelled pair's EEG and fNIRS component with its source.
    "joint_avg": Metric(
        lambda s: {**{f"eeg_{k}": abs(v) for k, v in s.r_eeg.items()},
                   **{f"fnirs_{k}": abs(v) for k, v in s.r_fnirs.items()}},
        fisher=True, needs_ground_truth=True),
    # 1 - Amari ISI per modality (1 = perfect separation); NaN with one source.
    "isi_avg": Metric(lambda s: {"eeg": 1.0 - s.isi_eeg, "fnirs": 1.0 - s.isi_fnirs},
                      fisher=False, needs_ground_truth=True),
    # Signed canonical correlation of each labelled pair; no ground truth in the value.
    "canonical_correlation_avg": Metric(lambda s: dict(s.canonical_r), fisher=True,
                                        needs_ground_truth=False),
}


def fold_value(name: str, scores: Scores | None) -> float:
    """
    One fold's value of a metric: the plain mean of its quantities in that fold.
    For display; selection uses `fold_mean`. NaN for a failed fit.
    """
    if scores is None:
        return np.nan
    values = np.array(list(METRICS[name].quantities(scores).values()), dtype=float)
    return float(np.mean(values)) if np.isfinite(values).all() else np.nan


def fold_mean(name: str, folds: list[Scores | None]) -> float:
    """
    The value selection compares: each quantity averaged over the folds (Fisher's z
    for correlations, plain otherwise), then the plain mean of the quantities.

    Returns
    -------
    float
        NaN, so the combination cannot be selected, if a fold failed (None, e.g. NaN
        filters) or a quantity is missing (NaN) in a fold: a combination is only
        compared when it could be judged on every fold. A labelled pair with too few
        samples stops the run before this (TooFewSamples), so a missing value here is
        one that does not exist, such as the ISI with one source.
    """
    metric = METRICS[name]
    if not folds or any(s is None for s in folds):
        return np.nan
    per_fold = [metric.quantities(s) for s in folds]
    means = []
    for key in per_fold[0]:
        values = np.array([q[key] for q in per_fold], dtype=float)
        if not np.isfinite(values).all():
            return np.nan
        if metric.fisher:
            means.append(np.tanh(np.mean(np.arctanh(np.clip(values, -0.999999, 0.999999)))))
        else:
            means.append(np.mean(values))
    return float(np.mean(means))


def check_metrics(names: tuple[str, ...]):
    """Raise on a metric name that is not in METRICS."""
    unknown = set(names) - set(METRICS)
    if unknown:
        raise ValueError(f"Unknown metric(s) {sorted(unknown)}; known: {sorted(METRICS)}.")
