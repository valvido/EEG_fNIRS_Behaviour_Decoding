"""
Build the objects a RawRecording holds from plain arrays, for datasets that are not
stored as FIF and SNIRF.
"""
import cedalion.dataclasses as cdc
import mne
import numpy as np
import xarray as xr

# Spacing between consecutive time stamps may deviate from 1 / rate by this much.
_RATE_TOLERANCE_S = 1e-6


def build_eeg(data_v: np.ndarray, sfreq: float, ch_names: list[str]) -> mne.io.RawArray:
    """
    EEG as an mne Raw, every channel of type "eeg", starting at time 0.

    Parameters
    ----------
    data_v : np.ndarray
        Signals in volts, shape (n_channels, n_times).
    sfreq : float
        Sampling rate (Hz).
    ch_names : list[str]
        Channel names, in row order.
    """
    info = mne.create_info(list(ch_names), sfreq=float(sfreq), ch_types="eeg")
    return mne.io.RawArray(data_v, info, verbose=False)


def build_fnirs_amplitude(intensities: dict[float, np.ndarray], time: np.ndarray,
                          channels: list[str], sources: list[str], detectors: list[str],
                          units: str = "V") -> xr.DataArray:
    """
    fNIRS amplitude in the layout cedalion reads from a SNIRF file.

    Parameters
    ----------
    intensities : dict[float, np.ndarray]
        Wavelength (nm) to raw intensity, shape (n_times, n_channels).
    time : np.ndarray
        Time stamps (s), shape (n_times,).
    channels, sources, detectors : list[str]
        Per channel: its name and the labels of its source and detector optodes,
        which `build_geo3d` places.
    units : str
        Units of the intensities.

    Returns
    -------
    xr.DataArray
        Dims (channel, wavelength, time), with source and detector coordinates on
        the channel axis and units on the values and the time axis.
    """
    wavelengths = list(intensities)
    data = np.stack([intensities[wl].T for wl in wavelengths], axis=1)
    return cdc.build_timeseries(
        data, dims=["channel", "wavelength", "time"], time=time, channel=list(channels),
        value_units=units, time_units="s",
        other_coords={"source": ("channel", list(sources)),
                      "detector": ("channel", list(detectors)),
                      "wavelength": ("wavelength", np.asarray(wavelengths, dtype=float))})


def build_geo3d(sources: dict[str, np.ndarray], detectors: dict[str, np.ndarray],
                landmarks: dict[str, np.ndarray] | None = None) -> xr.DataArray:
    """
    Optode positions as a cedalion labeled point cloud.

    Parameters
    ----------
    sources, detectors : dict[str, np.ndarray]
        Optode label to position in mm, shape (3,).
    landmarks : dict[str, np.ndarray] | None
        Head landmarks in the same frame, e.g. "Nz", "LPA", "RPA", which cedalion's
        scalp plots project with.

    Returns
    -------
    xr.DataArray
        Dims (label, pos), in mm, typed SOURCE, DETECTOR and LANDMARK.
    """
    landmarks = landmarks or {}
    labels = [*sources, *detectors, *landmarks]
    types = ([cdc.PointType.SOURCE] * len(sources) + [cdc.PointType.DETECTOR] * len(detectors)
             + [cdc.PointType.LANDMARK] * len(landmarks))
    positions = np.array([*sources.values(), *detectors.values(), *landmarks.values()],
                         dtype=float).reshape(-1, 3)
    return cdc.build_labeled_points(positions, units="mm", labels=labels, types=types)


def check_regular_time(time: np.ndarray, rate: float, where: str):
    """
    Raise if the time stamps do not start at 0 and step by exactly 1 / `rate`. The
    builders above put every signal on a clock that starts at 0, so a file whose
    stamps say otherwise would shift it unnoticed.
    """
    expected = np.arange(len(time)) / rate
    deviation = np.abs(time - expected).max()
    if deviation > _RATE_TOLERANCE_S:
        raise ValueError(f"{where}: time stamps deviate from 0, 1/{rate:g}, 2/{rate:g}, "
                         f"... by up to {deviation:.6f} s.")
