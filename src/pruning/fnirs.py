"""
Detect bad fNIRS channels with the cedalion quality criteria, judged on a cleaned
amplitude.
"""
import cedalion.nirs.cw as cw
import cedalion.sigproc.motion as motion
import cedalion.sigproc.quality as quality
import xarray as xr
from cedalion import units

from src.config import FnirsPruningConfig


def detect_bad_fnirs_channels(
        amp: xr.DataArray, geo3d: xr.DataArray, window: tuple[float, float],
        cfg: FnirsPruningConfig) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """
    Detect bad fNIRS channels on a window of the recording. The criteria are judged
    on the cleaned amplitude of the window (see `clean_amplitude`), not on the raw
    one. A channel is kept only if it passes every criterion:
    - source-detector distance within `sd_range_mm`
    - mean amplitude within `amp_range_v`
    - SNR above `snr_threshold`
    - SCI above `sci_threshold` and PSP above `psp_threshold`, both in the same
      window, in at least `time_pass_fraction` of the windows
    A criterion fails a channel if it fails at any wavelength.

    Parameters
    ----------
    amp : xr.DataArray
        Raw fNIRS amplitude of the whole recording, dims (channel, wavelength, time).
    geo3d : xr.DataArray
        Optode positions (mm).
    window : tuple[float, float]
        Start and end (s) of the window the channels are judged on.
    cfg : FnirsPruningConfig
        The filter settings and thresholds.

    Returns
    -------
    bad : tuple[str, ...]
        Bad channels, sorted by name.
    detail : dict[str, tuple[str, ...]]
        Channels failing each criterion: "sd_dist", "mean_amp", "snr", "sci_psp".
    """
    amp = clean_amplitude(amp, window, cfg)
    window_length = cfg.window_length_s * units.s

    _, sd_mask = quality.sd_dist(amp, geo3d, list(cfg.sd_range_mm) * units.mm)
    _, mean_amp_mask = quality.mean_amp(amp, list(cfg.amp_range_v) * units.V)
    _, snr_mask = quality.snr(amp, cfg.snr_threshold)

    # A window is clean only if SCI and PSP both pass in it.
    sci, sci_mask = quality.sci(amp, window_length, cfg.sci_threshold)
    _, psp_mask = quality.psp(amp, window_length, cfg.psp_threshold)
    clean_fraction = (sci_mask & psp_mask).sum("time") / len(sci.time)
    sci_psp_mask = clean_fraction >= cfg.time_pass_fraction

    masks = {"sd_dist": sd_mask, "mean_amp": mean_amp_mask,
             "snr": snr_mask, "sci_psp": sci_psp_mask}
    _, pruned = quality.prune_ch(amp, list(masks.values()), "all")

    detail = {name: _failing_channels(mask) for name, mask in masks.items()}
    return tuple(sorted(str(ch) for ch in pruned)), detail


def clean_amplitude(amp: xr.DataArray, window: tuple[float, float],
                    cfg: FnirsPruningConfig) -> xr.DataArray:
    """
    The amplitude of the window with motion artefacts and slow drift removed.

    The whole recording is converted to optical density, motion corrected (TDDR,
    then wavelet) and high-pass filtered, and only then cut to the window, so the
    filter has no edge artefacts inside it. The window is converted back to
    amplitude with the raw mean of the window as baseline, so the mean amplitude
    criterion still sees the raw intensity level.

    Parameters
    ----------
    amp : xr.DataArray
        Raw fNIRS amplitude of the whole recording, dims (channel, wavelength, time).
    window : tuple[float, float]
        Start and end (s) of the window to return.
    cfg : FnirsPruningConfig
        High-pass cutoff and filter order.

    Returns
    -------
    xr.DataArray
        Cleaned amplitude of the window, same units as `amp`.
    """
    start, end = window
    od = cw.int2od(amp)
    od = motion.tddr(od)
    od = motion.wavelet(od)
    od = od.cd.freq_filter(fmin=cfg.highpass_hz * units.Hz, fmax=0 * units.Hz,
                           butter_order=cfg.butter_order)

    baseline = amp.sel(time=slice(start, end)).mean("time")
    return cw.od2int(od.sel(time=slice(start, end)), baseline)


def _failing_channels(mask: xr.DataArray) -> tuple[str, ...]:
    """
    Channels where the mask is False along any other dimension, the same rule
    quality.prune_ch uses to drop a channel.
    """
    other_dims = [dim for dim in mask.dims if dim != "channel"]
    passes = mask.all(dim=other_dims)
    return tuple(sorted(str(ch) for ch in passes.channel.values[~passes.values]))
