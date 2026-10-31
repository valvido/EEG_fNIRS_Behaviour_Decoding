"""
EEG cleaning and the two EEG features the models use.

The feature functions take any mne.io.Raw, so the EEG ground-truth sources go through
exactly the same code as the EEG data.
"""
import warnings

import mne
import numpy as np
import xarray as xr

from src.config import EegPreprocessingConfig


def clean_eeg(eeg: mne.io.Raw, n_interpolated: int, cfg: EegPreprocessingConfig,
              allow_removal: bool = True) -> mne.io.Raw:
    """
    Broadband filter, then remove eye components with ICA. Without an EOG channel
    to find the eye components by, ICA is skipped with a warning and the EEG is only
    filtered.

    Parameters
    ----------
    eeg : mne.io.Raw
        Pruned EEG: montage set, bad channels already interpolated. Its EOG channels,
        if any, find the eye components.
    n_interpolated : int
        How many channels pruning interpolated. An interpolated channel is a linear
        combination of the others, so ICA can fit at most one component per channel
        that was not interpolated.
    cfg : EegPreprocessingConfig
        Filter band and ICA settings.
    allow_removal : bool
        False raises instead of removing eye components. Removing k components leaves
        the EEG with k fewer independent dimensions, so the channel covariance of
        band-passed EEG becomes singular, and mSPoC cannot be fit on it.

    Returns
    -------
    mne.io.Raw
        The cleaned EEG (a copy), all channels.

    Raises
    ------
    ValueError
        If `allow_removal` is False and ICA finds eye components.
    """
    raw = eeg.copy().filter(*cfg.bandpass, method="fir", phase="zero", verbose=False)
    if len(mne.pick_types(raw.info, eog=True)) == 0:
        warnings.warn("The EEG has no EOG channel to find eye components by: ICA is "
                      "skipped, eye artefacts stay in the EEG.", stacklevel=2)
        return raw

    n_eeg = len(mne.pick_types(raw.info, eeg=True))
    ica = mne.preprocessing.ICA(
        n_components=min(cfg.ica_n_components, n_eeg - n_interpolated),
        method=cfg.ica_method,
        random_state=cfg.random_state,
    )
    ica.fit(raw, verbose=False)
    ica.exclude, _ = ica.find_bads_eog(raw, verbose=False)
    if ica.exclude and not allow_removal:
        raise ValueError(
            f"ICA found {len(ica.exclude)} eye component(s) {sorted(ica.exclude)} to remove. "
            f"Removing them makes the covariance of band-passed EEG singular (rank "
            f"{n_eeg - n_interpolated - len(ica.exclude)} of {n_eeg - n_interpolated} "
            f"channels that were not interpolated), which mSPoC cannot fit on. "
            f"Reducing the EEG to its rank with PCA would allow it (not implemented).")
    return ica.apply(raw, verbose=False)


def band_power(raw: mne.io.Raw, band: tuple[float, float], rate: float,
               downsample_method: str) -> xr.DataArray:
    """
    Power envelope of the EEG channels in `band`, brought to `rate`: band-pass,
    Hilbert envelope squared (the instantaneous power |z|^2), low-pass at the new
    Nyquist frequency, then downsample.

    Power, not the amplitude |z|: with uncorrelated sources, a channel's power is the
    sum of the source powers weighted by the square of their field, so the ground-truth
    mixing of this feature is exactly A_eeg ** 2 (`ground_truth_mixing`).

    Parameters
    ----------
    raw : mne.io.Raw
        Cleaned EEG, or the EEG ground-truth sources as EEG-type channels.
    band : tuple[float, float]
        Frequency band (Hz).
    rate : float
        Target rate (Hz).
    downsample_method : str
        "resample": mne resampling. "epoch_mean": mean over consecutive windows of
        1 / rate seconds, time stamped at each window's start.

    Returns
    -------
    xr.DataArray
        Dims (time, channel), time in seconds on the recording's clock.
    """
    raw = raw.copy().pick("eeg")
    raw.filter(*band, method="fir", phase="zero", verbose=False)
    raw.apply_hilbert(envelope=True, verbose=False)
    raw.apply_function(np.square, channel_wise=False, verbose=False)
    raw.filter(None, rate / 2.0, method="fir", phase="zero", verbose=False)

    if downsample_method == "resample":
        raw.resample(rate, verbose=False)
        data, time = raw.get_data().T, raw.times
    else:
        epochs = mne.make_fixed_length_epochs(raw, duration=1.0 / rate, preload=True,
                                              verbose=False)
        data = epochs.get_data().mean(axis=-1)
        time = (epochs.events[:, 0] - raw.first_samp) / raw.info["sfreq"]

    return xr.DataArray(data, dims=("time", "channel"),
                        coords={"time": time, "channel": raw.ch_names})


def band_passed(raw: mne.io.Raw, band: tuple[float, float]) -> xr.DataArray:
    """
    The EEG channels filtered to `band`, at their own rate. mSPoC computes band
    power itself, as the variance within each of its epochs.

    Returns
    -------
    xr.DataArray
        Dims (time, channel), time in seconds on the recording's clock.
    """
    raw = raw.copy().pick("eeg")
    raw.filter(*band, method="fir", phase="zero", verbose=False)
    return xr.DataArray(raw.get_data().T, dims=("time", "channel"),
                        coords={"time": raw.times, "channel": raw.ch_names})
