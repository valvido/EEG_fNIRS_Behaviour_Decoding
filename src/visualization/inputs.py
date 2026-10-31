"""
The preprocessed signals the models were fit on.
"""
import matplotlib.pyplot as plt
import numpy as np

from src.types import PreprocessedRecording
from src.visualization.style import (BOUNDARY, MODALITY, MODALITY_NAME, shade_stimuli,
                                     stimulus_legend)


def plot_inputs(recordings: list[PreprocessedRecording], groups: list[str] | None = None,
                n_channels: int = 5):
    """
    Example channels of the EEG feature and the HbO, run after run, with the
    stimuli. The runs are drawn end to end on one time axis, each starting where the
    previous one ended; the gaps between runs are not shown.

    Parameters
    ----------
    recordings : list[PreprocessedRecording]
        As `load_inputs` returns them.
    groups : list[str] | None
        Only the runs of these groups, e.g. one fold's training groups; None = all.
    n_channels : int
        Channels per modality, spread evenly over the channel list.

    Returns
    -------
    matplotlib.figure.Figure
    """
    runs = [r for r in recordings if groups is None or r.group in groups]
    if not runs:
        raise ValueError(f"No runs of groups {groups}.")
    rows = [("eeg", ch) for ch in _spread(runs[0].eeg.channel.values, n_channels)] + \
           [("fnirs", ch) for ch in _spread(runs[0].fnirs.channel.values, n_channels)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(15, 1.3 * len(rows)), sharex=True)
    offset, starts = 0.0, []
    for run in runs:
        t0 = float(run.fnirs.time[0])
        starts.append(offset)
        for ax, (modality, channel) in zip(axes, rows):
            signal = (run.eeg if modality == "eeg" else run.fnirs).sel(channel=channel)
            ax.plot(signal.time.values - t0 + offset, signal.values, color=MODALITY[modality],
                    lw=0.5)
            stim = run.stim[(run.stim["onset"] >= t0) & (run.stim["onset"] <= float(run.fnirs.time[-1]))]
            shade_stimuli(ax, stim.assign(onset=stim["onset"] - t0 + offset))
        offset += float(run.fnirs.time[-1]) - t0 + float(np.median(np.diff(run.fnirs.time)))
    for ax, (modality, channel) in zip(axes, rows):
        for start in starts[1:]:
            ax.axvline(start, color=BOUNDARY, ls="--", lw=0.8)
        ax.set_ylabel(f"{MODALITY_NAME[modality]}\n{channel}", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time over the runs, end to end (s); dashed = run boundary")
    stimulus_legend(fig, runs[0].stim)
    names = ", ".join(sorted({r.group for r in runs}))
    fig.suptitle(f"Model inputs: {len(runs)} runs of {names}", fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _spread(channels, n: int) -> list:
    """`n` channels spread evenly over the list."""
    idx = np.linspace(0, len(channels) - 1, min(n, len(channels))).round().astype(int)
    return [channels[i] for i in idx]
