"""
Scores of the selected models, compared across methods.
"""
import matplotlib.pyplot as plt
import numpy as np

from src.types import RunResult, Selection
from src.visualization.style import BAR, MODALITY, MODALITY_NAME, hand


def plot_source_recovery(result: RunResult, metric: str):
    """
    Test-group recovery of every method's model selected by `metric`, one bar per
    method.

    Top row: |r| of each source's labelled pair with that source, one panel per
    modality and source. Bottom row (two sources): |r| of the pair labelled as the
    other source with this source, i.e. how much of each source leaks into the other
    source's component. Bars and the numbers on them are |r|, on 0 to 1.

    Parameters
    ----------
    result : RunResult
    metric : str
        A selection metric of the run.

    Returns
    -------
    matplotlib.figure.Figure
    """
    selections = _by_metric(result.unmixing.selections, metric)
    sources = list(selections[0].scores.labels.pairs)
    panels = [(m, s) for m in ("eeg", "fnirs") for s in sources]
    rows = [("recovery", lambda sel, m, s: _r(sel.scores, m)[s])]
    if len(sources) == 2:
        rows.append(("cross", lambda sel, m, s: getattr(sel.scores, f"leak_{m}")[s]))

    fig, axes = plt.subplots(len(rows), len(panels), figsize=(3.6 * len(panels), 3.4 * len(rows)),
                             sharey=True, squeeze=False)
    methods = [sel.method for sel in selections]
    for (kind, value), row_axes in zip(rows, axes):
        for ax, (modality, source) in zip(row_axes, panels):
            _method_bars(ax, methods, np.abs([value(sel, modality, source) for sel in selections]))
            other = hand(sources[1 - sources.index(source)]) if kind == "cross" else None
            ax.set_title(f"{MODALITY_NAME[modality]} {hand(source)}" if kind == "recovery" else
                         f"{MODALITY_NAME[modality]} {hand(source)}\nvs pair labelled {other}",
                         fontsize=9)
        row_axes[0].set_ylabel("test correlation |r|" if kind == "recovery" else
                               "leakage |r|")
    axes[0, 0].set_ylim(0, 1)
    fig.suptitle(f"Source recovery by method · selected by {metric} · test group "
                 f"{result.unmixing.test_group}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_pattern_recovery(result: RunResult, metric: str):
    """
    Test-group pattern recovery of every method's model selected by `metric`, one bar
    per method: |r| of each source's labelled pair's Haufe pattern (on the test group)
    with that source's ground-truth mixing, one panel per modality and source. A
    diagnostic, never used for selection.

    Parameters
    ----------
    result : RunResult
    metric : str
        A selection metric of the run.

    Returns
    -------
    matplotlib.figure.Figure
    """
    selections = _by_metric(result.unmixing.selections, metric)
    sources = list(selections[0].scores.labels.pairs)
    panels = [(m, s) for m in ("eeg", "fnirs") for s in sources]
    methods = [sel.method for sel in selections]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.4), sharey=True,
                             squeeze=False)
    for ax, (modality, source) in zip(axes[0], panels):
        _method_bars(ax, methods, [getattr(sel.scores, f"pattern_r_{modality}")[source]
                                   for sel in selections])
        ax.set_title(f"{MODALITY_NAME[modality]} {hand(source)}", fontsize=9)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("pattern correlation |r|")
    fig.suptitle(f"Pattern recovery (Haufe pattern vs ground-truth mixing) by method · "
                 f"selected by {metric} · test group {result.unmixing.test_group}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_isi(result: RunResult, metric: str):
    """
    Test-group Amari ISI of every method's model selected by `metric`: the labelled
    pairs' filters against the ground-truth mixing, one bar per method, one panel per
    modality. 0 is perfect separation, 1 none. Not defined for one source.

    Parameters
    ----------
    result : RunResult
    metric : str
        A selection metric of the run.

    Returns
    -------
    matplotlib.figure.Figure
    """
    selections = _by_metric(result.unmixing.selections, metric)
    methods = [sel.method for sel in selections]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, modality in zip(axes, ("eeg", "fnirs")):
        _method_bars(ax, methods, [getattr(sel.scores, f"isi_{modality}") for sel in selections])
        ax.set_title(MODALITY_NAME[modality], fontsize=9)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("ISI (lower is better)")
    fig.suptitle(f"Separation (ISI) by method · selected by {metric} · test group "
                 f"{result.unmixing.test_group}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_nested_spread(result: RunResult, metric: str, kind: str = "recovery"):
    """
    Nested evaluation: the spread of each method's test scores over the held-out
    groups, one box per method, one panel per modality and source.

    Parameters
    ----------
    result : RunResult
        With nested evaluation.
    metric : str
    kind : str
        "recovery": |r| of the labelled pair with its source. "pattern": |r| of the
        pair's pattern with the ground-truth mixing (a diagnostic, never used for
        selection).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not result.unmixing.nested:
        raise ValueError("The run has no nested evaluation (unmixing.nested: false).")
    per_group = [_by_metric(sels, metric) for sels in result.unmixing.nested.values()]
    methods = [sel.method for sel in per_group[0]]
    sources = list(per_group[0][0].scores.labels.pairs)
    panels = [(m, s) for m in ("eeg", "fnirs") for s in sources]
    get = (lambda sc, m: _r(sc, m)) if kind == "recovery" else \
        (lambda sc, m: getattr(sc, f"pattern_r_{m}"))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6), sharey=True)
    for ax, (modality, source) in zip(np.atleast_1d(axes), panels):
        data = [[get(sel.scores, modality)[source] for sels in per_group for sel in sels
                 if sel.method == method] for method in methods]
        ax.boxplot(data, widths=0.5, showfliers=False, medianprops={"color": "black"},
                   boxprops={"color": MODALITY[modality]}, whiskerprops={"color": MODALITY[modality]},
                   capprops={"color": MODALITY[modality]})
        for i, values in enumerate(data, start=1):
            ax.scatter(np.full(len(values), i), values, s=14, color=MODALITY[modality], zorder=3)
        ax.set_xticks(range(1, len(methods) + 1), methods, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{MODALITY_NAME[modality]} {hand(source)}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    first = np.atleast_1d(axes)[0]
    first.set_ylim(0, 1)
    first.set_ylabel("|r|" if kind == "recovery" else "pattern |r|")
    fig.suptitle(f"Nested evaluation: {'recovery' if kind == 'recovery' else 'pattern correlation'}"
                 f" over {len(per_group)} held-out groups · selected by {metric}", fontsize=10)
    fig.tight_layout()
    return fig


def _method_bars(ax, methods: list[str], values):
    """One bar per method with its value written on top; NaN bars are left empty."""
    values = np.asarray(values, dtype=float)
    bars = ax.bar(range(len(methods)), values, color=BAR, width=0.7)
    for bar, v in zip(bars, values):
        if np.isfinite(v):
            ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(range(len(methods)), methods, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def _by_metric(selections: list[Selection], metric: str) -> list[Selection]:
    chosen = [sel for sel in selections if sel.metric == metric]
    if not chosen:
        raise ValueError(f"No selection by metric {metric!r}; the run has "
                         f"{sorted({sel.metric for sel in selections})}.")
    return chosen


def _r(scores, modality: str) -> dict:
    return scores.r_eeg if modality == "eeg" else scores.r_fnirs

