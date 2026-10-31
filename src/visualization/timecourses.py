"""
Recovered components over time, next to the ground truth where there is one.
"""
import matplotlib.pyplot as plt

from src.types import Selection
from src.visualization.orient import oriented
from src.visualization.style import (GROUND_TRUTH, MODALITY_NAME, RECOVERED, hand,
                                     mark_runs, shade_stimuli, stimulus_legend, zscore)


def plot_time_courses(selection: Selection, val_group: str | None = None):
    """
    Per modality and hand: the labelled pair's component (left) and the ground
    truth (right), both z-scored, with run boundaries and the movement stimuli
    (onset line and duration band). The component is flipped for display as its
    pattern is (`oriented`). Without ground truth, the components alone, each with
    its pair's canonical r.

    Without `val_group`, the selected model on the test group. With it, the selected
    hyperparameters fit in that validation fold, on the validation group.

    Parameters
    ----------
    selection : Selection
    val_group : str | None
        One of the groups in `selection.folds`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if val_group is None:
        scores, recovery = selection.scores, selection.recovery
        where = f"test group {selection.test_group}"
    else:
        folds = {f.val_group: f for f in selection.folds}
        if val_group not in folds:
            raise ValueError(f"No fold {val_group!r}; the selection has {list(folds)}.")
        scores, recovery = folds[val_group].scores, folds[val_group].recovery
        where = f"validation fold {val_group}"

    with_truth = recovery.s_eeg is not None
    rows = [(m, s) for m in ("eeg", "fnirs") for s in scores.labels.pairs]
    fig, axes = plt.subplots(len(rows), 2 if with_truth else 1,
                             figsize=(15, 2.2 * len(rows)), sharex=True, squeeze=False)
    for (modality, source), row_axes in zip(rows, axes):
        pair = scores.labels.pairs[source]
        comp, _ = oriented(recovery, selection.method, modality, pair)
        label = f"{MODALITY_NAME[modality]} pair {pair}\n-> {hand(source)}"
        if not with_truth:
            label += f"  canonical r={scores.canonical_r[source]:+.2f}"
        row_axes[0].plot(comp.time, zscore(comp.values), color=RECOVERED, lw=0.8)
        row_axes[0].set_ylabel(label, fontsize=8)
        if with_truth:
            truth = recovery.s_eeg if modality == "eeg" else recovery.s_fnirs
            r = (scores.r_eeg if modality == "eeg" else scores.r_fnirs)[source]
            gt = truth.sel(source=source)
            row_axes[1].plot(gt.time, zscore(gt.values), color=GROUND_TRUTH, lw=0.8)
            row_axes[1].set_ylabel(f"truth {MODALITY_NAME[modality]}\n{hand(source)}  "
                                   f"|r|={r:.2f}", fontsize=8)
        for ax in row_axes:
            mark_runs(ax, recovery.run_starts)
            shade_stimuli(ax, recovery.stim)
            ax.grid(alpha=0.3)
    axes[0, 0].set_title("recovered (z-scored)", fontsize=9)
    if with_truth:
        axes[0, 1].set_title("ground truth (z-scored)", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("time on the joined runs (s); dashed = run boundary")
    stimulus_legend(fig, recovery.stim)
    params = selection.params or "no hyperparameters"
    fig.suptitle(f"{selection.method} · selected by {selection.metric} · {params} · {where}",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
