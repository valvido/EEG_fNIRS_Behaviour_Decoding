"""
fNIRS conversion from amplitude to HbO, and the filters shared with the fNIRS
ground-truth sources.
"""
import cedalion.nirs.cw as cw
import cedalion.sigproc.motion as motion
import numpy as np
import xarray as xr
from cedalion import units
from cedalion.math.resample import resample

from src.config import FnirsPreprocessingConfig


def fnirs_to_hbo(amp: xr.DataArray, geo3d: xr.DataArray,
                 cfg: FnirsPreprocessingConfig) -> xr.DataArray:
    """
    Amplitude to band-passed HbO: optical density, motion correction (TDDR, then
    wavelet), haemoglobin concentration, band-pass. Runs on the whole recording, so
    the filters have no edge artefacts inside the part that is kept later.

    Parameters
    ----------
    amp : xr.DataArray
        Pruned fNIRS amplitude, dims (channel, wavelength, time).
    geo3d : xr.DataArray
        Optode positions (mm).
    cfg : FnirsPreprocessingConfig
        DPF and band.

    Returns
    -------
    xr.DataArray
        HbO without units, dims (time, channel).
    """
    return od_to_hbo(motion_corrected_od(amp), geo3d, cfg)


def motion_corrected_od(amp: xr.DataArray) -> xr.DataArray:
    """
    Amplitude to optical density, motion corrected with TDDR, then wavelet.

    Parameters
    ----------
    amp : xr.DataArray
        fNIRS amplitude, dims (channel, wavelength, time).

    Returns
    -------
    xr.DataArray
        Optical density, same dims.
    """
    return motion.wavelet(motion.tddr(cw.int2od(amp)))


def od_to_hbo(od: xr.DataArray, geo3d: xr.DataArray,
              cfg: FnirsPreprocessingConfig) -> xr.DataArray:
    """
    Optical density to band-passed HbO: haemoglobin concentration (modified
    Beer-Lambert law, the same DPF for every wavelength), HbO, band-pass.

    Parameters
    ----------
    od : xr.DataArray
        Optical density, dims (channel, wavelength, time).
    geo3d : xr.DataArray
        Optode positions (mm).
    cfg : FnirsPreprocessingConfig
        DPF and band.

    Returns
    -------
    xr.DataArray
        HbO without units, dims (time, channel).
    """
    dpf = xr.DataArray([cfg.dpf] * od.sizes["wavelength"], dims="wavelength",
                       coords={"wavelength": od.wavelength})
    conc = cw.od2conc(od, geo3d, dpf)
    hbo = conc.sel(chromo="HbO").pint.dequantify()
    return band_pass(hbo, cfg.bandpass).transpose("time", "channel")


def band_pass(ts: xr.DataArray, band: tuple[float, float]) -> xr.DataArray:
    """
    Zero-phase Butterworth band-pass along time (cedalion freq_filter). Used for the
    HbO data and for the fNIRS ground-truth sources alike.

    Parameters
    ----------
    ts : xr.DataArray
        Any array with a time axis in seconds.
    band : tuple[float, float]
        Pass band (Hz).
    """
    return _as_timeseries(ts).cd.freq_filter(fmin=band[0] * units.Hz,
                                             fmax=band[1] * units.Hz)


def downsample(ts: xr.DataArray, rate: float) -> xr.DataArray:
    """
    Anti-aliased downsampling to `rate` (cedalion resample: low-pass at the new
    Nyquist frequency, then interpolation onto a regular grid). Used for the HbO data
    and for the ground-truth sources alike.

    Parameters
    ----------
    ts : xr.DataArray
        Any array with a time axis in seconds.
    rate : float
        New rate (Hz), lower than the current one.
    """
    ts = _as_timeseries(ts)
    current = 1.0 / float(np.median(np.diff(ts.time.values)))
    if rate >= current:
        raise ValueError(f"Cannot downsample to {rate} Hz: the signal is already at "
                         f"{current:.3f} Hz.")
    out = resample(ts, Fs=rate)
    # The resampler interpolates the samples coordinate into fractional values.
    return out.assign_coords(samples=("time", np.arange(out.sizes["time"])))


def _as_timeseries(ts: xr.DataArray) -> xr.DataArray:
    """
    Add what cedalion's time series functions need when it is missing: a samples
    coordinate and the time unit.
    """
    if "samples" not in ts.coords:
        ts = ts.assign_coords(samples=("time", np.arange(ts.sizes["time"])))
    if "units" not in ts.time.attrs:
        ts = ts.assign_coords(time=ts.time.assign_attrs(units="s"))
    return ts
