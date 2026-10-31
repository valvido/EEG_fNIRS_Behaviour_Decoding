"""
Join preprocessed runs end to end into the one recording a model is fit on.
"""
import numpy as np
import pandas as pd
import xarray as xr

from src.types import JOINED_STIM_COLUMNS, JoinedRuns, PreprocessedRecording


def concat_runs(recordings: list[PreprocessedRecording]) -> JoinedRuns:
    """
    Join runs in the order given.

    Each run's fNIRS is placed on a new regular time axis that continues one fNIRS
    sample after the previous run ends. The EEG and the ground truth move with it:
    band power keeps exactly the fNIRS time points, band-passed EEG keeps its offset
    from the run's first fNIRS sample, so mSPoC's blocks stay on their fNIRS samples.
    The stimuli move by the same offset as the run's time axis; a stimulus whose onset
    lies outside the run's fNIRS samples is left out, since it would land in the
    neighbouring run. The runs' channel stds become one gain per channel
    (`JoinedRuns.eeg_gain`, `fnirs_gain`).

    Parameters
    ----------
    recordings : list[PreprocessedRecording]
        Runs of one subject. All must have the same EEG and fNIRS channels, the same
        sources and the same rates.

    Returns
    -------
    JoinedRuns

    Raises
    ------
    ValueError
        If the runs differ in channels, sources or rates.
    """
    if not recordings:
        raise ValueError("No runs to join.")
    first = recordings[0]
    fnirs_dt = _dt(first.fnirs)
    eeg_dt = _dt(first.eeg)

    eeg, fnirs, s_eeg, s_fnirs, stim, starts = [], [], [], [], [], []
    cursor = 0.0
    for rec in recordings:
        name = f"{rec.group}/{rec.run}"
        _check_same(rec, first, name)
        for label, da, dt in (("fNIRS", rec.fnirs, fnirs_dt), ("EEG", rec.eeg, eeg_dt)):
            if not np.isclose(_dt(da), dt, rtol=1e-6):
                raise ValueError(f"{name}: {label} rate {1 / _dt(da):.6g} Hz differs from "
                                 f"the first run's {1 / dt:.6g} Hz.")

        t0 = float(rec.fnirs.time[0])
        fnirs_time = cursor + np.arange(rec.fnirs.sizes["time"]) * fnirs_dt
        same_axis = np.array_equal(rec.eeg.time.values, rec.fnirs.time.values)
        eeg_time = fnirs_time if same_axis else cursor + (rec.eeg.time.values - t0)

        starts.append(cursor)
        fnirs.append(_on(rec.fnirs, fnirs_time, first.fnirs.channel.values, "channel"))
        eeg.append(_on(rec.eeg, eeg_time, first.eeg.channel.values, "channel"))
        if first.s_eeg is not None:
            s_eeg.append(_on(rec.s_eeg, fnirs_time, first.s_eeg.source.values, "source"))
            s_fnirs.append(_on(rec.s_fnirs, fnirs_time, first.s_fnirs.source.values, "source"))
        stim.append(_stim_on(rec.stim, t0, float(rec.fnirs.time[-1]), cursor, name))
        cursor = float(fnirs_time[-1]) + fnirs_dt

    join = lambda parts: xr.concat(parts, dim="time") if parts else None
    lengths = [rec.fnirs.sizes["time"] for rec in recordings]
    return JoinedRuns(eeg=join(eeg), fnirs=join(fnirs), s_eeg=join(s_eeg),
                      s_fnirs=join(s_fnirs), run_starts=tuple(starts),
                      runs=tuple(f"{r.group}/{r.run}" for r in recordings),
                      stim=pd.concat(stim, ignore_index=True), ground_truth=first.ground_truth,
                      eeg_gain=_gain([r.eeg_std for r in recordings], lengths,
                                     first.eeg.channel.values, "EEG"),
                      fnirs_gain=_gain([r.fnirs_std for r in recordings], lengths,
                                       first.fnirs.channel.values, "fNIRS"))


def _gain(stds: list[xr.DataArray | None], lengths: list[int], channels: np.ndarray,
          label: str) -> xr.DataArray | None:
    """
    1 / std per channel, averaged over the runs weighted by their length: the one
    channel scaling that best describes the joined data, whose runs were each
    z-scored on their own. None if no run was scaled.
    """
    if all(std is None for std in stds):
        return None
    if any(std is None for std in stds):
        raise ValueError(f"Some runs have their {label} std and others do not.")
    inverse = [1.0 / std.sel(channel=channels) for std in stds]
    return sum(n * g for n, g in zip(lengths, inverse)) / sum(lengths)


def _check_same(rec: PreprocessedRecording, first: PreprocessedRecording, name: str):
    """Raise if `rec` has other channels or sources than the first run."""
    pairs = [("EEG channels", rec.eeg.channel, first.eeg.channel),
             ("fNIRS channels", rec.fnirs.channel, first.fnirs.channel)]
    if (rec.s_eeg is None) != (first.s_eeg is None):
        raise ValueError(f"{name}: some runs have ground truth and others do not.")
    if first.s_eeg is not None:
        pairs += [("EEG sources", rec.s_eeg.source, first.s_eeg.source),
                  ("fNIRS sources", rec.s_fnirs.source, first.s_fnirs.source)]
    for label, mine, theirs in pairs:
        if set(mine.values) != set(theirs.values):
            raise ValueError(f"{name}: {label} differ from the first run's: "
                             f"{sorted(set(mine.values) ^ set(theirs.values))}.")


def _on(da: xr.DataArray, time: np.ndarray, labels: np.ndarray, dim: str) -> xr.DataArray:
    """`da` as (time, dim) on the new time axis, in the order of `labels`."""
    da = da.transpose("time", dim).sel({dim: labels})
    da = da.drop_vars([c for c in da.coords if c not in ("time", dim)])
    return da.assign_coords(time=xr.DataArray(time, dims="time", attrs={"units": "s"}))


def _stim_on(stim: pd.DataFrame, t0: float, t1: float, cursor: float,
             run: str) -> pd.DataFrame:
    """The stimuli with onsets in [t0, t1], moved so t0 lands on `cursor`."""
    if stim.empty:
        return pd.DataFrame(columns=JOINED_STIM_COLUMNS)
    stim = stim[(stim["onset"] >= t0) & (stim["onset"] <= t1)].copy()
    stim["onset"] = stim["onset"] - t0 + cursor
    stim["run"] = run
    return stim[JOINED_STIM_COLUMNS]


def _dt(da: xr.DataArray) -> float:
    return float(np.median(np.diff(da.time.values)))
