"""
Cut both modalities to the stimulus window, put them on one time axis, and scale.
"""
import numpy as np
import pandas as pd
import xarray as xr


def stimulus_window(stim: pd.DataFrame, pre_sec: float, post_sec: float) -> tuple[float, float]:
    """
    From `pre_sec` before the first onset until `post_sec` after the last stimulus
    ends. The stim table is on the clock both modalities share, so one window serves
    both.

    Returns
    -------
    start, end : float
        Seconds on the recording's clock.
    """
    if stim.empty:
        raise ValueError("The recording has no stimuli to take the window from.")
    start = float(stim["onset"].min()) - pre_sec
    end = float((stim["onset"] + stim["duration"]).max()) + post_sec
    return start, end


def align_to_window(eeg: xr.DataArray, fnirs: xr.DataArray, sources: dict[str, xr.DataArray],
                    window: tuple[float, float],
                    same_axis: bool) -> tuple[xr.DataArray, xr.DataArray, dict[str, xr.DataArray]]:
    """
    Cut every signal to `window` and put the EEG on the fNIRS time axis.

    The fNIRS and the ground-truth sources share one time axis already; they are cut
    to the window. Then:
    - `same_axis` (band power): the EEG is interpolated onto the fNIRS time points,
      so all signals end up on exactly the same time axis.
    - otherwise (band-passed EEG for mSPoC): the EEG keeps its own rate and is cut
      into one block of k EEG samples per fNIRS sample (`cut_to_blocks`).
    Time points where the fNIRS, the sources or (with `same_axis`) the EEG are not
    finite are dropped.

    Parameters
    ----------
    eeg : xr.DataArray
        EEG feature, dims (time, channel), time in seconds on the recording's clock.
    fnirs : xr.DataArray
        HbO, dims (time, channel), on the same clock.
    sources : dict[str, xr.DataArray]
        Ground-truth sources on the fNIRS time axis; empty for real data.
    window : tuple[float, float]
        Start and end (s), from `stimulus_window`.
    same_axis : bool
        Whether the EEG is put on the fNIRS time points.

    Returns
    -------
    eeg, fnirs, sources
        Cut and aligned.
    """
    start, end = window
    fnirs = fnirs.sel(time=slice(start, end))
    sources = {name: s.sel(time=slice(start, end)) for name, s in sources.items()}

    if same_axis:
        eeg = on_time_axis(eeg, fnirs.time)
        keep = finite_times(fnirs, eeg, *sources.values())
        eeg = eeg.isel(time=keep)
    else:
        keep = finite_times(fnirs, *sources.values())
    fnirs = fnirs.isel(time=keep)
    sources = {name: s.isel(time=keep) for name, s in sources.items()}
    if not same_axis:
        eeg, fnirs, sources = cut_to_blocks(eeg, fnirs, sources)
    return eeg, fnirs, sources


def cut_to_blocks(eeg: xr.DataArray, fnirs: xr.DataArray,
                  sources: dict[str, xr.DataArray]) -> tuple[xr.DataArray, xr.DataArray,
                                                             dict[str, xr.DataArray]]:
    """
    Cut band-passed EEG into exactly k = EEG rate / fNIRS rate samples per fNIRS
    sample: fNIRS sample i gets the EEG from its own time to the next fNIRS sample.

    mSPoC takes the EEG power of block i as the partner of fNIRS sample i, with the
    block length computed as (EEG samples) // (fNIRS samples). Only when the EEG has
    exactly k samples per fNIRS sample do the blocks stay on their fNIRS samples;
    with fewer, every block starts a little early and the error grows along the run.
    The EEG starts at the first fNIRS sample; fNIRS samples at the end without a full
    EEG block are dropped, together with their ground truth.

    Parameters
    ----------
    eeg : xr.DataArray
        Band-passed EEG, dims (time, channel), starting at or before the fNIRS.
    fnirs : xr.DataArray
        HbO, dims (time, channel), at a rate that divides the EEG rate.
    sources : dict[str, xr.DataArray]
        Ground-truth sources on the fNIRS time axis.

    Returns
    -------
    eeg, fnirs, sources
        EEG with k samples per remaining fNIRS sample.
    """
    eeg_rate = 1.0 / float(np.median(np.diff(eeg.time.values)))
    fnirs_rate = 1.0 / float(np.median(np.diff(fnirs.time.values)))
    k = round(eeg_rate / fnirs_rate)
    if k < 2 or abs(eeg_rate / fnirs_rate - k) > 1e-3 * k:
        raise ValueError(f"The fNIRS rate ({fnirs_rate:.4g} Hz) must divide the EEG rate "
                         f"({eeg_rate:.4g} Hz) into whole blocks of at least 2 EEG samples.")
    first = int(np.searchsorted(eeg.time.values, float(fnirs.time[0]) - 0.5 / eeg_rate))
    eeg = eeg.isel(time=slice(first, None))
    n = min(fnirs.sizes["time"], eeg.sizes["time"] // k)
    sources = {name: s.isel(time=slice(0, n)) for name, s in sources.items()}
    return eeg.isel(time=slice(0, n * k)), fnirs.isel(time=slice(0, n)), sources


def on_time_axis(da: xr.DataArray, time: xr.DataArray) -> xr.DataArray:
    """
    `da` linearly interpolated onto `time`, dims (time, ...). Samples outside `da`'s
    span become NaN.
    """
    da = da.drop_vars([c for c in ("samples", "chromo") if c in da.coords])
    da = da.interp(time=np.asarray(time))
    return da.transpose("time", ...)


def finite_times(*arrays: xr.DataArray) -> np.ndarray:
    """
    Mask over the shared time axis: True where every array is finite in every
    channel. All arrays must be on the same time axis.
    """
    keep = np.ones(arrays[0].sizes["time"], dtype=bool)
    for da in arrays:
        other_dims = [dim for dim in da.dims if dim != "time"]
        keep &= np.isfinite(da).all(dim=other_dims).values
    return keep


def finite_span(*arrays: xr.DataArray) -> slice:
    """
    The part of the shared time axis between the first and the last sample where
    every array is finite. Signals of slightly different length leave NaN at the
    edges once they are put on one time axis; a filter run over such a signal spreads
    the NaN through all of it.
    """
    keep = finite_times(*arrays)
    if not keep.any():
        raise ValueError("No time point where all signals are finite.")
    first = int(np.argmax(keep))
    last = len(keep) - int(np.argmax(keep[::-1]))
    return slice(first, last)


def zscore(da: xr.DataArray) -> xr.DataArray:
    """
    Each channel scaled to mean 0 and standard deviation 1 over time.
    """
    return (da - da.mean("time")) / da.std("time")


def as_timeseries(da: xr.DataArray) -> xr.DataArray:
    """
    Dims (time, ...) with a contiguous samples coordinate and the time unit, which
    cedalion's epoching needs.
    """
    da = da.transpose("time", ...)
    da = da.assign_coords(samples=("time", np.arange(da.sizes["time"])))
    return da.assign_coords(time=da.time.assign_attrs(units="s"))
