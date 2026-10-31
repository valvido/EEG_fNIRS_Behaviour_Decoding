"""
Which canonical pair each hand was assigned to, in every fold and in the test refit,
next to the assignment that would have been best on the held-out data.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.config import AssignmentConfig
from src.types import Recovery, Selection
from src.unmixing import best_assignment, match_weights
from src.visualization.style import SELECTED, SEQUENTIAL, hand

BEST = "#d62728"               # the assignment that is best on the held-out data
OUTSIDE = "#bdbdbd"            # pairs beyond n_top, never assigned


def plot_pair_assignment(selection: Selection, assignment: AssignmentConfig):
    """
    The match score of every pair with every hand, one row per validation fold and
    one for the test refit.

    The score is the one the assignment maximises, eeg_weight |r_eeg| + fnirs_weight
    |r_fnirs|, from 0 to eeg_weight + fnirs_weight. Top row of panels: on the
    training runs, what the assignment saw. Bottom row: on the held-out runs. A box
    marks the pair the hand was assigned (on the training runs); a star marks the
    pair the best assignment on the held-out runs would have given it. Pairs beyond
    `n_top` are greyed out, as no hand can be assigned to them. A row label ends in
    "(differs)" when the assigned and the held-out best assignment differ.

    Parameters
    ----------
    selection : Selection
        A non-nested selection, with its fold refits.
    assignment : AssignmentConfig
        The run's assignment config.

    Returns
    -------
    matplotlib.figure.Figure
    """
    splits = [(f"fold {f.val_group}", f.recovery, f.scores.labels) for f in selection.folds]
    splits.append((f"test {selection.test_group}", selection.recovery, selection.scores.labels))
    hands = list(selection.scores.labels.pairs)
    n_pairs = selection.recovery.u.sizes["component"]
    n_top = min(assignment.n_top, n_pairs)
    vmax = assignment.eeg_weight + assignment.fnirs_weight

    held_out_best = [best_assignment(rec.r_test_eeg, rec.r_test_fnirs, hands, assignment)
                     for _, rec, _ in splits]
    row_labels = [name if labels.pairs == best.pairs else f"{name} (differs)"
                  for (name, _, labels), best in zip(splits, held_out_best)]

    fig, axes = plt.subplots(2, len(hands), squeeze=False, sharex=True, sharey=True,
                             figsize=(0.55 * n_pairs * len(hands) + 2.5,
                                      2 * (0.45 * len(splits) + 1.2)))
    for row, data in enumerate(("train", "test")):
        for col, source in enumerate(hands):
            ax = axes[row, col]
            weights = np.array([_match_with(rec, data, source, assignment)
                                for _, rec, _ in splits])
            image = ax.imshow(weights, cmap=SEQUENTIAL, vmin=0, vmax=vmax, aspect="auto")
            _write_values(ax, weights, vmax)
            if n_top < n_pairs:
                ax.axvspan(n_top - 0.5, n_pairs - 0.5, color=OUTSIDE, alpha=0.6, lw=0)
            for i, (_, _, labels) in enumerate(splits):
                ax.add_patch(Rectangle((labels.pairs[source] - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor=SELECTED, lw=2.2))
            if data == "test":
                # In the cell's top right corner, so the value stays readable.
                ax.scatter([best.pairs[source] + 0.3 for best in held_out_best],
                           np.arange(len(splits)) - 0.28, marker="*", s=45, color=BEST,
                           zorder=3)
            ax.set_title(f"{hand(source)} · {'training' if data == 'train' else 'held-out'} "
                         f"runs", fontsize=9)
    for ax in axes[-1]:
        ax.set_xticks(range(n_pairs))
        ax.set_xlabel("pair (component)")
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(splits)), row_labels, fontsize=8)

    fig.colorbar(image, ax=axes, shrink=0.8,
                 label=f"{assignment.eeg_weight:g} |r EEG| + {assignment.fnirs_weight:g} |r fNIRS|")
    fig.legend(handles=[
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=SELECTED, lw=2.2,
                  label="assigned (on training runs)"),
        Line2D([], [], marker="*", ls="", color=BEST, markersize=10,
               label="best assignment on held-out runs"),
        Rectangle((0, 0), 1, 1, color=OUTSIDE, alpha=0.6, label=f"beyond n_top = {n_top}")],
        loc="lower center", bbox_to_anchor=(0.45, -0.02), ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"Pair assignment · {selection.method} · selected by {selection.metric}",
                 fontsize=10)
    return fig


def _match_with(recovery: Recovery, data: str, source: str,
                assignment: AssignmentConfig) -> np.ndarray:
    """Every pair's match score with one source, on the training or held-out runs."""
    r_eeg = getattr(recovery, f"r_{data}_eeg").sel(source=[source])
    r_fnirs = getattr(recovery, f"r_{data}_fnirs").sel(source=[source])
    return match_weights(r_eeg, r_fnirs, assignment)[:, 0]


def _write_values(ax, weights: np.ndarray, vmax: float):
    """Each cell's value, in white on dark cells."""
    for i, j in np.ndindex(weights.shape):
        ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center", fontsize=6,
                color="white" if weights[i, j] > 0.6 * vmax else "black")
