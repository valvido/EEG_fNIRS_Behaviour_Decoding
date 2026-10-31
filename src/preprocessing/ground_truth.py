"""
Put the ground-truth sources through the same transforms as the data, so recovered
components can be scored against them.
"""
import mne
import numpy as np
import xarray as xr

from src.config import PreprocessingConfig
from src.preprocessing.align import on_time_axis
from src.preprocessing.eeg import band_power
from src.preprocessing.fnirs import band_pass
from src.types import GroundTruth


def ground_truth_sources(ground_truth: GroundTruth, time: xr.DataArray,
                         cfg: PreprocessingConfig) -> tuple[xr.DataArray, xr.DataArray]:
    """
    The EEG and fNIRS ground-truth sources, transformed like the data and placed on
    the fNIRS time axis.

    - EEG: the power envelope of the feature band, with the same function as the EEG
      data. This holds for mSPoC too: what it recovers on the EEG side is a band power
      course, so that is what it is scored against.
    - fNIRS: HbO, band-passed with the same function as the fNIRS data.

    No sign convention is applied: band power does not depend on the source's sign
    and the band-pass is linear, so the sign can be fixed later, where it is needed.

    Parameters
    ----------
    ground_truth : GroundTruth
        As loaded.
    time : xr.DataArray
        The fNIRS time axis to place the sources on.
    cfg : PreprocessingConfig
        Feature band, fNIRS band, and the rate and method of the band power.

    Returns
    -------
    s_eeg, s_fnirs : xr.DataArray
        Dims (time, source), on `time`.
    """
    s_eeg = ground_truth.s_eeg
    rate = 1.0 / float(np.median(np.diff(s_eeg.time.values)))
    sources = [str(s) for s in s_eeg.source.values]
    raw = mne.io.RawArray(s_eeg.transpose("source", "time").values,
                          mne.create_info(sources, sfreq=rate, ch_types="eeg"),
                          verbose=False)
    power = band_power(raw, cfg.eeg.feature_band, cfg.fnirs_srate, cfg.downsample_method)
    # RawArray starts its clock at 0; put the result back on the recording's clock.
    power = power.assign_coords(time=power.time + float(s_eeg.time[0]))
    power = power.rename(channel="source")

    hbo = ground_truth.s_fnirs.sel(chromo="HbO").pint.dequantify()
    hbo = band_pass(hbo, cfg.fnirs.bandpass).pint.dequantify()

    return on_time_axis(power, time), on_time_axis(hbo, time)

