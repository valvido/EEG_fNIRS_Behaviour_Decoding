"""
Which hyperparameters were selected, and how the selection metric looked over the grid.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from src.types import CombinationScore, RunResult, Selection
from src.visualization.style import SELECTED, SEQUENTIAL

# Hyperparameters given as [EEG, fNIRS].
PER_MODALITY = ("l1_reg", "l2_reg")


def plot_selection_landscape(result: RunResult, method: str, metric: str):
    """
    One panel per tuned hyperparameter of `method`: the metric's mean over the
    validation folds for each value, with the other hyperparameters held at the
    selected ones (a slice through the winner). [EEG, fNIRS] pairs such as l1_reg are
    a heatmap with EEG on x and fNIRS on y; other hyperparameters (time_shifts) are
    bars, with each fold's value as a dot. A thick outline marks the selected value.
    With nested evaluation, "nested k×" counts how many held-out groups picked a
    value (only picks that lie on the slice). A method with nothing tuned (CCA) has
    no landscape.

    Parameters
    ----------
    result : RunResult
    method : str
    metric : str

    Returns
    -------
    matplotlib.figure.Figure | None
        None when `method` has nothing tuned.
    """
    test_group = result.unmixing.test_group
    combos = [c for c in result.unmixing.combinations
              if c.method == method and c.test_group == test_group]
    if not combos:
        raise ValueError(f"No combinations of {method!r} in the run.")
    selected = [s for s in result.unmixing.selections if s.method == method and s.metric == metric]
    if not selected:
        raise ValueError(f"{metric!r} selected no {method} model.")
    winner = selected[0].params
    tuned = [key for key in winner if len({_label(c.params[key]) for c in combos}) > 1]
    nested = [s.params for sels in (result.unmixing.nested or {}).values() for s in sels
              if s.method == method and s.metric == metric]

    if not tuned:
        return None
    fig, axes = plt.subplots(1, len(tuned), figsize=(4.6 * len(tuned), 4.0), squeeze=False)
    for ax, key in zip(axes[0], tuned):
        others = [k for k in winner if k != key]
        on_slice = [c for c in combos if all(_same(c.params[k], winner[k]) for k in others)]
        picks = [p for p in nested if all(_same(p[k], winner[k]) for k in others)]
        if key in PER_MODALITY:
            _heatmap(ax, key, on_slice, winner, picks, metric)
        else:
            _bars(ax, key, on_slice, winner, picks, metric, result, method, test_group)
    held = ", ".join(f"{k}={_label(winner[k])}" for k in winner if k not in tuned) or "none"
    fig.suptitle(f"{method} · mean {metric} over the validation folds · other "
                 f"hyperparameters at the selected values · fixed: {held}", fontsize=9)
    fig.tight_layout()
    return fig


def selected_params(result: RunResult) -> pd.DataFrame:
    """
    The selected hyperparameters: one row per method, one column per metric. A value
    on the edge of its grid is marked "(edge)": the grid may need to reach further.

    Returns
    -------
    pd.DataFrame
    """
    table = {}
    for sel in result.unmixing.selections:
        combos = [c for c in result.unmixing.combinations
                  if c.method == sel.method and c.test_group == sel.test_group]
        table.setdefault(sel.method, {})[sel.metric] = _describe(sel.params, combos) or "(none)"
    return pd.DataFrame(table).T


def _heatmap(ax, key: str, combos: list[CombinationScore], winner: dict, picks: list,
             metric: str):
    xs = sorted({c.params[key][0] for c in combos})
    ys = sorted({c.params[key][1] for c in combos})
    grid = np.full((len(ys), len(xs)), np.nan)
    for c in combos:
        grid[ys.index(c.params[key][1]), xs.index(c.params[key][0])] = c.means[metric]
    image = ax.imshow(grid, origin="lower", cmap=SEQUENTIAL, aspect="auto")
    counts = {}
    for p in picks:
        cell = (ys.index(p[key][1]), xs.index(p[key][0]))
        counts[cell] = counts.get(cell, 0) + 1
    for i in range(len(ys)):
        for j in range(len(xs)):
            if np.isfinite(grid[i, j]):
                text = f"{grid[i, j]:.2f}" + (f"\nnested {counts[i, j]}×" if (i, j) in counts else "")
                ax.text(j, i, text, ha="center", va="center", fontsize=7,
                        color="white" if grid[i, j] > np.nanmean(grid) else "black")
    i, j = ys.index(winner[key][1]), xs.index(winner[key][0])
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=SELECTED, lw=3,
                           label="selected"))
    ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs], fontsize=8)
    ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys], fontsize=8)
    ax.set_xlabel(f"{key} EEG")
    ax.set_ylabel(f"{key} fNIRS")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=f"mean {metric}")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.15))


def _bars(ax, key: str, combos: list[CombinationScore], winner: dict, picks: list, metric: str,
          result: RunResult, method: str, test_group: str):
    labels = sorted({_label(c.params[key]) for c in combos})
    means = {_label(c.params[key]): c.means[metric] for c in combos}
    w = labels.index(_label(winner[key]))
    bars = ax.bar(range(len(labels)), [means[l] for l in labels], color="#c3c2b7", width=0.6)
    bars[w].set_edgecolor(SELECTED)
    bars[w].set_linewidth(3)
    bars[w].set_label("selected")
    for i, label in enumerate(labels):
        params = next(c.params for c in combos if _label(c.params[key]) == label)
        folds = [p.metrics[metric] for p in result.unmixing.landscape
                 if p.method == method and p.test_group == test_group and _same(p.params, params)]
        ax.scatter(np.full(len(folds), i) + np.linspace(-0.12, 0.12, len(folds)), folds, s=16,
                   color="#2a78d6", zorder=3, label="validation fold" if i == 0 else None)
        n = sum(_label(p[key]) == label for p in picks)
        if n:
            ax.annotate(f"nested {n}×", (i, 0), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel(key)
    ax.set_ylabel(f"{metric}: mean (bar) and per fold (dots)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def _describe(params: dict, combos: list[CombinationScore]) -> str:
    parts = []
    for key, value in params.items():
        values = {_label(c.params[key]) for c in combos}
        text = _label(value)
        if len(values) > 1 and key in PER_MODALITY:
            edges = []
            for i, name in enumerate(("EEG", "fNIRS")):
                grid = sorted({c.params[key][i] for c in combos})
                if len(grid) > 1 and value[i] in (grid[0], grid[-1]):
                    edges.append(name)
            if edges:
                text += f" (edge: {', '.join(edges)})"
        parts.append(f"{key} {text}")
    return " · ".join(parts)


def _label(value) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return "[" + ", ".join(f"{v:g}" for v in np.asarray(value, dtype=float)) + "]"
    return f"{value:g}" if isinstance(value, float) else str(value)


def _same(a, b) -> bool:
    plain = lambda v: json.dumps(v, sort_keys=True, default=lambda o: np.asarray(o).tolist())
    return plain(a) == plain(b)
