"""
Maps over the head: patterns, against the ground-truth mixing where there is one, and
the pruned channels.
EEG as MNE topomaps on the subject's electrode positions, fNIRS with cedalion's scalp
plot on the subject's optode positions.
"""
import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
import xarray as xr
import cedalion.geometry.registration as registration
from cedalion.vis.anatomy.scalp_plot import scalp_plot
from matplotlib.colors import ListedColormap

from src.positions import electrode_montage
from src.types import Geometry, RunResult, Selection
from src.unmixing import ground_truth_mixing
from src.unmixing.models import METHODS
from src.visualization.orient import oriented
from src.visualization.style import BOUNDARY, DIVERGING, hand

# fNIRS maps, as in the legacy figures.
FNIRS_MAP = "YlOrRd"
# The circles marking channels on the EEG maps, as on the fNIRS ones.
MARKED_EEG = {"marker": "o", "markerfacecolor": "none", "markeredgecolor": "black",
              "markeredgewidth": 1.2, "markersize": 9}


def plot_patterns(selection: Selection, geometry: Geometry, val_group: str | None = None,
                  motor_channels: dict[str, dict[str, tuple[str, ...]]] | None = None):
    """
    Per hand (columns): the ground-truth mixing and the labelled pair's Haufe
    pattern, for EEG and fNIRS (rows); without ground truth, the patterns alone. The
    patterns are computed on the held-out runs and flipped for display to be
    positive at their largest channel (`oriented`). The mixing is taken as in the
    scores: EEG band power as the square of the EEG field, fNIRS as HbO at the first
    wavelength, both scaled like the z-scored data the patterns were computed on.

    Without `val_group`, the selected model's patterns on the test group. With it,
    the patterns of the selected hyperparameters fit in that validation fold, on the
    validation group.

    Colours as in the legacy figures: each EEG map spans its own minimum to maximum;
    fNIRS maps share one scale per row (truth, patterns).

    Parameters
    ----------
    selection : Selection
    geometry : Geometry
        The run's geometry; its ground-truth mixing, if any, is drawn.
    val_group : str | None
        One of the groups in `selection.folds`.
    motor_channels : dict[str, dict[str, tuple[str, ...]]] | None
        Modality to hand to the channels the "motor_channels" assignment looked at
        (`AssignmentConfig.motor_channels`); they are circled on that hand's
        patterns, where the data has them. None marks nothing.

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
    with_truth = geometry.A_eeg is not None
    motor_channels = motor_channels or {}
    marked = lambda modality, source: motor_channels.get(modality, {}).get(source, ())

    pairs = scores.labels.pairs
    sources = list(pairs)
    pattern = lambda modality, s: oriented(recovery, selection.method, modality, pairs[s])[1]
    patterns_fnirs = [pattern("fnirs", s) for s in sources]
    if with_truth:
        A_eeg = ground_truth_mixing(geometry, "eeg", recovery.gain_eeg,
                                    METHODS[selection.method].eeg_feature)
        A_fnirs = ground_truth_mixing(geometry, "fnirs", recovery.gain_fnirs)
        truth_fnirs = [A_fnirs.sel(source=s) for s in sources]
    n_rows = 4 if with_truth else 2
    fig, axes = plt.subplots(n_rows, len(sources), figsize=(3.8 * len(sources), 3.25 * n_rows),
                             squeeze=False)
    eeg_row, fnirs_row = (1, 3) if with_truth else (0, 1)
    for col, source in enumerate(sources):
        pair = pairs[source]
        if with_truth:
            _eeg_map(axes[0, col], A_eeg.sel(source=source), geometry, f"truth EEG {hand(source)}")
            _fnirs_map(axes[2, col], truth_fnirs[col], geometry, _limits(truth_fnirs),
                       f"truth fNIRS (HbO) {hand(source)}")
            eeg_scores = (f"|r|={scores.r_eeg[source]:.2f} · "
                          f"pattern |r|={scores.pattern_r_eeg[source]:.2f}")
            fnirs_scores = (f"|r|={scores.r_fnirs[source]:.2f} · "
                            f"pattern |r|={scores.pattern_r_fnirs[source]:.2f}")
        else:
            eeg_scores = fnirs_scores = (f"{hand(source)} · "
                                         f"canonical r={scores.canonical_r[source]:+.2f}")
        _eeg_map(axes[eeg_row, col], pattern("eeg", source), geometry,
                 f"EEG pattern, pair {pair}\n{eeg_scores}", marked("eeg", source))
        _fnirs_map(axes[fnirs_row, col], patterns_fnirs[col], geometry, _limits(patterns_fnirs),
                   f"fNIRS pattern, pair {pair}\n{fnirs_scores}", marked("fnirs", source))
    against = " vs ground-truth mixing" if with_truth else ""
    circled = " · circled: motor channels" if motor_channels else ""
    fig.suptitle(f"{selection.method} · selected by {selection.metric} · patterns on the "
                 f"{where}{against}{circled}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_bad_channels(result: RunResult):
    """
    The channels pruning removed: bad EEG electrodes (interpolated) marked on the
    montage, and dropped fNIRS channels on the probe.

    Returns
    -------
    matplotlib.figure.Figure
    """
    bad = result.bad_channels
    geometry = result.geometry
    fig, (ax_eeg, ax_fnirs) = plt.subplots(1, 2, figsize=(11, 5))

    eeg = sorted(set(geometry.eeg_channels) | set(bad.eeg))
    info = _eeg_info(eeg, geometry)
    info["bads"] = list(bad.eeg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mne.viz.plot_sensors(info, kind="topomap", show_names=True, axes=ax_eeg, show=False)
    ax_eeg.set_title(f"EEG: {len(bad.eeg)} bad, interpolated (red)\n{', '.join(bad.eeg) or 'none'}",
                     fontsize=9)

    fnirs = list(geometry.fnirs_channels) + [c for c in bad.fnirs if c not in geometry.fnirs_channels]
    dropped = xr.DataArray([float(c in bad.fnirs) for c in fnirs], dims="channel",
                           coords={"channel": fnirs})
    scalp_plot(_channels(fnirs, geometry), geometry.geo3d, dropped, ax=ax_fnirs, vmin=0, vmax=1,
               cmap=ListedColormap(["#c3c2b7", "#e34948"]), add_colorbar=False, zorder="ascending")
    ax_fnirs.set_title(f"fNIRS: {len(bad.fnirs)} of {len(fnirs)} channels dropped (red)",
                       fontsize=9)
    fig.suptitle("Pruning, found on the rest recording", fontsize=10)
    fig.tight_layout()
    return fig


def _eeg_map(ax, values: xr.DataArray, geometry: Geometry, title: str,
             marked: tuple[str, ...] = ()):
    """
    A topomap of one value per EEG channel, colours from its own min to max, with
    the `marked` channels circled.
    """
    values = values.dropna("channel")
    mask = np.isin(values.channel.values, list(marked))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image, _ = mne.viz.plot_topomap(values.values, _eeg_info(values.channel.values, geometry),
                                        axes=ax, show=False, cmap=DIVERGING,
                                        vlim=_limits([values]), mask=mask,
                                        mask_params=MARKED_EEG)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8)


def _fnirs_map(ax, values: xr.DataArray, geometry: Geometry, limits: tuple, title: str,
               marked: tuple[str, ...] = ()):
    """
    One value per fNIRS channel on the probe, on the given colour limits, with the
    `marked` channels circled at their midpoints.
    """
    values = values.dropna("channel")
    values = values.drop_vars([c for c in values.coords if c != "channel"])
    scalp_plot(_channels(values.channel.values, geometry), geometry.geo3d, values, ax=ax,
               vmin=limits[0], vmax=limits[1], cmap=FNIRS_MAP, bad_color=BOUNDARY,
               zorder="ascending")
    _circle_fnirs_channels(ax, [c for c in marked if c in values.channel.values], geometry)
    ax.set_title(title, fontsize=8)


def _circle_fnirs_channels(ax, channels: list[str], geometry: Geometry):
    """
    A circle at the midpoint of each channel's source and detector, on the 2D
    projection cedalion's scalp plot draws the probe on.
    """
    if not channels:
        return
    geo2d = registration.simple_scalp_projection(geometry.geo3d)
    midpoints = np.array([(geo2d.loc[source].values + geo2d.loc[detector].values) / 2
                          for source, detector in (geometry.fnirs_optodes[c] for c in channels)])
    ax.scatter(midpoints[:, 0], midpoints[:, 1], s=90, facecolors="none", edgecolors="black",
               linewidths=1.2, zorder=5)


def _limits(maps: list) -> tuple[float, float]:
    """(min, max) over the finite values of the maps; (-1, 1) if empty or flat."""
    values = np.concatenate([np.asarray(m, dtype=float).ravel() for m in maps])
    values = values[np.isfinite(values)]
    if values.size and values.max() > values.min():
        return float(values.min()), float(values.max())
    return -1.0, 1.0


def _eeg_info(channels, geometry: Geometry) -> mne.Info:
    """The MNE info of `channels`, placed by the run's electrode positions."""
    info = mne.create_info([str(c) for c in channels], sfreq=1.0, ch_types="eeg")
    info.set_montage(electrode_montage(geometry.eeg_geo3d))
    return info


def _channels(names, geometry: Geometry) -> xr.DataArray:
    """The channel description cedalion's scalp plot needs: each channel's optodes."""
    names = [str(c) for c in names]
    pairs = [geometry.fnirs_optodes[c] for c in names]
    return xr.DataArray(np.zeros(len(names)), dims="channel",
                        coords={"channel": names, "source": ("channel", [s for s, _ in pairs]),
                                "detector": ("channel", [d for _, d in pairs])})
