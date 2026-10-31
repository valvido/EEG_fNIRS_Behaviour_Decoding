"""
File readers shared by the dataset readers, one per file type: FIF, SNIRF, MAT.
"""
from pathlib import Path

import cedalion.io
import mne
import pandas as pd
import scipy.io
import xarray as xr


def read_fif(path: Path) -> mne.io.Raw:
    """
    The EEG of a FIF file as recorded, all channels, loaded into memory.
    """
    return mne.io.read_raw_fif(path, preload=True, verbose=False)


def read_snirf(path: Path) -> tuple[xr.DataArray, xr.DataArray, pd.DataFrame]:
    """
    The amplitude, optode geometry and stim table of a SNIRF file.

    Returns
    -------
    amp : xr.DataArray
        Amplitude, dims (channel, wavelength, time); the time axis carries units "s",
        which some SNIRF files omit and cedalion's epoching needs.
    geo3d : xr.DataArray
        Optode positions in mm (cedalion loads them without units).
    stim : pd.DataFrame
        Stim table sorted by onset.
    """
    rec = cedalion.io.read_snirf(path)[0]
    amp = rec["amp"]
    if "units" not in amp.time.attrs:
        amp.time.attrs["units"] = "s"
    geo3d = rec.geo3d.pint.dequantify().pint.quantify("mm")
    stim = rec.stim.sort_values("onset").reset_index(drop=True)
    return amp, geo3d, stim


def read_mat(path: Path) -> dict:
    """
    The variables of a MATLAB file (format v7.2 or older), by name.

    Singleton dimensions are squeezed away, so a scalar comes back as a number, a
    vector as a 1-D array and a cell array as a 1-D object array. MATLAB's own header
    entries ("__header__", ...) are left out.
    """
    variables = scipy.io.loadmat(path, squeeze_me=True)
    return {name: value for name, value in variables.items() if not name.startswith("__")}
