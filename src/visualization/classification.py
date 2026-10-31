"""
Classification accuracy of the selected models, compared across methods.
"""
import matplotlib.pyplot as plt
import numpy as np

from src.types import RunResult
from src.visualization.style import BAR, BOUNDARY, MODALITY

FEATURE_SET_COLOUR = {"eeg_fnirs": BAR, "eeg": MODALITY["eeg"], "fnirs": MODALITY["fnirs"]}
FEATURE_SET_NAME = {"eeg_fnirs": "EEG + fNIRS", "eeg": "EEG", "fnirs": "fNIRS"}


def plot_classification(result: RunResult, metric: str):
    """
    Test accuracy of the classifier on every method's model selected by `metric`:
    one group of bars per method, one bar per feature set, the chance level (1 /
    number of classes) as a dashed line. With nested evaluation, each held-out
    group's accuracy is a dot on its bar.

    Parameters
    ----------
    result : RunResult
        With classification.
    metric : str
        A selection metric of the run.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if result.classification is None:
        raise ValueError("The run has no classification (no classification section).")
    scores = [s for s in result.classification.scores if s.metric == metric]
    if not scores:
        raise ValueError(f"No classification of models selected by {metric!r}.")
    methods = list(dict.fromkeys(s.method for s in scores))
    feature_sets = list(dict.fromkeys(s.feature_set for s in scores))
    classes = scores[0].classes
    nested = [s for group in (result.classification.nested or {}).values() for s in group
              if s.metric == metric]

    width = 0.8 / len(feature_sets)
    fig, ax = plt.subplots(figsize=(max(4.5, 1.3 * len(methods) * len(feature_sets)), 3.8))
    for k, feature_set in enumerate(feature_sets):
        x = np.arange(len(methods)) + (k - (len(feature_sets) - 1) / 2) * width
        accuracy = [next(s.accuracy for s in scores
                         if s.method == m and s.feature_set == feature_set) for m in methods]
        bars = ax.bar(x, accuracy, width=width * 0.95, color=FEATURE_SET_COLOUR[feature_set],
                      label=FEATURE_SET_NAME[feature_set])
        for bar, value in zip(bars, accuracy):
            ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
        for i, method in enumerate(methods):
            spread = [s.accuracy for s in nested
                      if s.method == method and s.feature_set == feature_set]
            ax.scatter(np.full(len(spread), x[i]), spread, s=10, color="black", zorder=3)
    ax.axhline(1 / len(classes), color=BOUNDARY, ls="--", lw=0.8, label="chance")
    ax.set_xticks(range(len(methods)), methods, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("test accuracy")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    title = (f"rLDA {' / '.join(classes)} · selected by {metric} · test group "
             f"{result.classification.test_group}")
    if nested:
        title += "\ndots: each held-out group of nested evaluation"
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig
