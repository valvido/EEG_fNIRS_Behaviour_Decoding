"""
What every unmixing figure shares: colours, source names, run boundaries and stimuli.
"""
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Colours by role, the same in every figure.
RECOVERED = "#2a78d6"          # recovered components
GROUND_TRUTH = "#eb6834"       # ground-truth sources
MODALITY = {"eeg": "#2a78d6", "fnirs": "#eb6834"}
# Movement trials by trial type, a darker shade of the side's colour for the arm.
# Trial types not listed here (e.g. rest) are not drawn.
STIMULUS = {"right": "#1baf7a", "right_arm": "#0b6e4c", "left": "#6250d6",
            "left_arm": "#3a2d8f"}
STIMULUS_NAME = {"right": "right hand", "right_arm": "right arm", "left": "left hand",
                 "left_arm": "left arm"}
BOUNDARY = "#52514e"           # run boundaries
SELECTED = "#eda100"           # the selected hyperparameters
BAR = "#4f81b8"                # one bar per method, as in the legacy comparison
DIVERGING = "RdBu_r"           # signed values around 0: patterns, correlations
SEQUENTIAL = "Blues"           # magnitudes: metric values over a grid

MODALITY_NAME = {"eeg": "EEG", "fnirs": "fNIRS"}


def hand(source: str) -> str:
    """The display name of a source label, e.g. "right_hand" -> "right hand"."""
    return str(source).replace("_", " ")


def zscore(values: np.ndarray) -> np.ndarray:
    """Standardized over the finite entries; NaNs stay NaN."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 2 or finite.std() == 0:
        return values
    return (values - finite.mean()) / finite.std()


def mark_runs(ax, run_starts):
    """Dashed lines where each run after the first starts."""
    for start in list(run_starts)[1:]:
        ax.axvline(start, color=BOUNDARY, ls="--", lw=0.8, alpha=0.7)


def shade_stimuli(ax, stim: pd.DataFrame):
    """
    Each movement stimulus (a trial type in STIMULUS) as a line at its onset and a
    band over its duration, coloured by trial type.
    """
    for row in stim[stim["trial_type"].isin(list(STIMULUS))].itertuples():
        color = STIMULUS[row.trial_type]
        ax.axvline(row.onset, color=color, lw=0.8, alpha=0.8)
        ax.axvspan(row.onset, row.onset + row.duration, color=color, alpha=0.15, lw=0)


def stimulus_legend(fig, stim: pd.DataFrame):
    """A legend for the stimulus bands, if there are any."""
    types = [t for t in STIMULUS if t in set(stim["trial_type"])]
    if types:
        fig.legend(handles=[Patch(color=STIMULUS[t], alpha=0.3, label=STIMULUS_NAME[t])
                            for t in types],
                   loc="upper right", ncol=len(types), frameon=False, fontsize=8)
