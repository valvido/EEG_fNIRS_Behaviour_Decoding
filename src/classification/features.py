"""
Trials, epochs and interval-mean features of the recovered components.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from cedalion import units
from cedalion.sigproc.epochs import to_epochs

from src.config import ClassificationConfig
from src.types import FittedModel, PreprocessedRecording
from src.unmixing import unmix_run


@dataclass(frozen=True)
class TrialFeatures:
    """
    The features of every trial of a set of runs, one row per trial. The EEG and
    the fNIRS features have the same rows.

    Attributes
    ----------
    eeg : np.ndarray
        Dims (trial, interval x pair): the mean of each labelled pair's EEG component
        in each EEG interval.
    fnirs : np.ndarray
        The same for the fNIRS components and the fNIRS intervals.
    labels : np.ndarray
        The class of each trial.
    n_dropped : int
        Trials left out because their baseline window or a whole interval had no
        finite sample.
    """
    eeg: np.ndarray
    fnirs: np.ndarray
    labels: np.ndarray
    n_dropped: int

    def of(self, feature_set: str) -> np.ndarray:
        """The feature matrix of "eeg_fnirs", "eeg" or "fnirs"."""
        if feature_set == "eeg_fnirs":
            return np.hstack([self.eeg, self.fnirs])
        return {"eeg": self.eeg, "fnirs": self.fnirs}[feature_set]


def epoch_span(cfg: ClassificationConfig) -> tuple[float, float]:
    """
    [start, end] seconds relative to onset that every epoch covers: the baseline
    window and every EEG and fNIRS interval.
    """
    windows = [cfg.baseline_window, *cfg.eeg_intervals, *cfg.fnirs_intervals]
    return min(lo for lo, _ in windows), max(hi for _, hi in windows)


def trial_features(fitted: FittedModel, pairs: dict[str, int],
                   recordings: list[PreprocessedRecording],
                   cfg: ClassificationConfig,
                   trial_classes: dict[str, tuple[str, ...]] | None = None) -> TrialFeatures:
    """
    The features of every trial in `recordings`, from the labelled pairs of one
    fitted model.

    Each run is unmixed on its own, so its stimuli apply on its own clock and no
    time shift of a temporal model reaches into another run. Each trial is cut into
    an epoch of `epoch_span`, its baseline mean is subtracted, and the mean of every
    interval is one feature.

    Parameters
    ----------
    fitted : FittedModel
    pairs : dict[str, int]
        Source to pair, as labelled on the model's training runs.
    recordings : list[PreprocessedRecording]
    cfg : ClassificationConfig
    trial_classes : dict[str, tuple[str, ...]] | None
        Side to the stim trial types that make up its class (see `run_trials`).

    Returns
    -------
    TrialFeatures
    """
    span = epoch_span(cfg)
    eeg, fnirs, labels, n_dropped = [], [], [], 0
    for rec in recordings:
        u, v, stim = unmix_run(fitted, rec)
        comps = pair_components(u, v, pairs)
        time = comps.time.values
        trials = run_trials(stim, cfg, time[0], time[-1], trial_classes)
        epochs = subtract_baseline(epoch_components(comps, trials, span), cfg.baseline_window)
        is_eeg = epochs.modality.values == "eeg"
        x_eeg = interval_means(epochs.isel(component=is_eeg), cfg.eeg_intervals)
        x_fnirs = interval_means(epochs.isel(component=~is_eeg), cfg.fnirs_intervals)
        finite = np.isfinite(x_eeg).all(axis=1) & np.isfinite(x_fnirs).all(axis=1)
        eeg.append(x_eeg[finite])
        fnirs.append(x_fnirs[finite])
        labels.append(epochs.trial_type.values[finite])
        n_dropped += len(trials) - int(finite.sum())
    return TrialFeatures(eeg=np.vstack(eeg), fnirs=np.vstack(fnirs),
                         labels=np.concatenate(labels).astype(str), n_dropped=n_dropped)


def pair_components(u: xr.DataArray, v: xr.DataArray, pairs: dict[str, int]) -> xr.DataArray:
    """
    The EEG and the fNIRS component of every labelled pair, on the fNIRS
    components' time points.

    Parameters
    ----------
    u, v : xr.DataArray
        EEG and fNIRS components of one run, dims (time, component).
    pairs : dict[str, int]
        Source to pair.

    Returns
    -------
    xr.DataArray
        Dims (time, component), the EEG components first. The coordinates
        `modality` ("eeg" | "fnirs") and `source` on component name each one. EEG
        time points a temporal model does not reach are NaN.
    """
    u = u.reindex(time=v.time)
    columns, modality, source = [], [], []
    for name, comps in (("eeg", u), ("fnirs", v)):
        for src, pair in pairs.items():
            columns.append(comps.sel(component=pair).values)
            modality.append(name)
            source.append(src)
    return xr.DataArray(np.stack(columns, axis=1), dims=("time", "component"),
                        coords={"time": v.time.values, "modality": ("component", modality),
                                "source": ("component", source)})


def run_trials(stim: pd.DataFrame, cfg: ClassificationConfig, data_start: float,
               data_end: float,
               trial_classes: dict[str, tuple[str, ...]] | None = None) -> pd.DataFrame:
    """
    The trials of `cfg.classes` in one run, by onset. If rest is a class and the
    run's stim table has no rest trials, they are cut from the gaps (`rest_trials`).

    Parameters
    ----------
    stim : pd.DataFrame
    cfg : ClassificationConfig
    data_start, data_end : float
        Seconds the run's components cover.
    trial_classes : dict[str, tuple[str, ...]] | None
        Side to the stim trial types that make up its class
        (`SubjectConfig.trial_classes`). Each trial type is relabelled to its side
        first, so with {"left": ("left", "left_arm")} a "left_arm" trial is a "left"
        trial. Trial types in no class, and all of them if None, keep their label.

    Returns
    -------
    pd.DataFrame
        onset, duration, trial_type.
    """
    if trial_classes:
        side_of = {t: side for side, types in trial_classes.items() for t in types}
        stim = stim.assign(trial_type=stim.trial_type.map(lambda t: side_of.get(t, t)))
    trials = stim[stim.trial_type.isin(cfg.classes)]
    if cfg.rest_class in cfg.classes and not (stim.trial_type == cfg.rest_class).any():
        rest = rest_trials(stim, epoch_span(cfg), data_start, data_end, cfg.rest_class)
        trials = pd.concat([trials, rest], ignore_index=True)
    return trials.sort_values("onset", ignore_index=True)[["onset", "duration", "trial_type"]]


def rest_trials(stim: pd.DataFrame, span: tuple[float, float], data_start: float,
                data_end: float, rest_class: str) -> pd.DataFrame:
    """
    Rest trials in the stretches of a run that no stimulus's epoch covers.

    Every stimulus, of any trial type, covers [onset + start, onset + end] of
    `span`. Each free stretch between `data_start` and `data_end` holds as many
    rest epochs of the same length as fit, back to back and centred in the stretch.

    Returns
    -------
    pd.DataFrame
        onset, duration (0), trial_type (`rest_class`); none if no stretch is as long
        as one epoch.
    """
    start, end = span
    length = end - start
    covered = sorted((onset + start, onset + end) for onset in stim.onset)
    onsets = []
    free_from = data_start
    for lo, hi in covered + [(data_end, data_end)]:
        free = lo - free_from
        n_fit = max(0, int(free // length))
        first_start = free_from + (free - n_fit * length) / 2
        onsets += [first_start + k * length - start for k in range(n_fit)]
        free_from = max(free_from, hi)
    return pd.DataFrame({"onset": onsets, "duration": 0.0, "trial_type": rest_class})


def epoch_components(comps: xr.DataArray, trials: pd.DataFrame,
                     span: tuple[float, float]) -> xr.DataArray:
    """
    One epoch per trial with cedalion's `to_epochs`, covering `span` around onset.
    An epoch that reaches past either end of `comps` is kept, NaN where the run has
    no data.

    Returns
    -------
    xr.DataArray
        Dims (epoch, reltime, component), with `trial_type` on epoch.
    """
    start, end = span
    comps = pad_with_nan(comps, max(0.0, -start), max(0.0, end))
    # cedalion's time series schema: a samples coordinate and the time unit.
    comps = comps.assign_coords(samples=("time", np.arange(comps.sizes["time"])))
    comps.time.attrs["units"] = "s"
    epochs = to_epochs(comps, trials, trial_types=sorted(set(trials.trial_type)),
                       before=max(0.0, -start) * units.s, after=max(0.0, end) * units.s)
    if getattr(epochs.pint, "units", None) is not None:
        epochs = epochs.pint.dequantify()
    return epochs.transpose("epoch", "reltime", "component")


def pad_with_nan(comps: xr.DataArray, before: float, after: float) -> xr.DataArray:
    """
    `comps` with NaN samples added on its own regular time grid: at least `before`
    seconds before its first time point and `after` seconds after its last one.
    """
    dt = float(np.median(np.diff(comps.time.values)))
    n_before = int(np.ceil(before / dt)) + 1
    n_after = int(np.ceil(after / dt)) + 1
    t0, t1 = float(comps.time[0]), float(comps.time[-1])
    time = np.r_[t0 - dt * np.arange(n_before, 0, -1), comps.time.values,
                 t1 + dt * np.arange(1, n_after + 1)]
    return comps.reindex(time=time)


def subtract_baseline(epochs: xr.DataArray, window: tuple[float, float]) -> xr.DataArray:
    """
    Each epoch minus its mean over `window` (seconds relative to onset), over the
    finite samples there. An epoch with no finite sample in the window becomes NaN.
    """
    lo, hi = window
    in_window = (epochs.reltime >= lo) & (epochs.reltime <= hi)
    return epochs - epochs.sel(reltime=in_window).mean("reltime")


def interval_means(epochs: xr.DataArray, intervals: tuple[tuple[float, float], ...]) -> np.ndarray:
    """
    The mean of every component in every [t0, t1] interval (seconds after onset),
    over the finite samples there: an interval cut by the end of the run is the mean
    of the part the run has. NaN where the interval has no finite sample.

    Returns
    -------
    np.ndarray
        Dims (epoch, interval x component), interval-major.
    """
    means = [epochs.sel(reltime=slice(t0, t1)).mean("reltime") for t0, t1 in intervals]
    return np.hstack([m.transpose("epoch", "component").values for m in means])
