"""
Preprocess each pruned run into the signals the models are fit on.
"""
import mne
import xarray as xr

from src.config import PreprocessingConfig
from src.positions import electrode_positions
from src.preprocessing.align import (align_to_window, as_timeseries, finite_span,
                                     stimulus_window, zscore)
from src.preprocessing.check import check_alignment
from src.preprocessing.eeg import band_passed, band_power, clean_eeg
from src.preprocessing.fnirs import downsample, fnirs_to_hbo
from src.preprocessing.ground_truth import ground_truth_sources
from src.types import PreprocessedRecording, PrunedRecording


def preprocess_recordings(recordings: list[PrunedRecording],
                          cfg: PreprocessingConfig) -> list[PreprocessedRecording]:
    """
    Preprocess every run on its own, in the order given. With cfg.eeg.drop_interpolated,
    every EEG channel interpolated in any run is dropped from all runs, so they keep the
    same channels.
    """
    dropped_eeg = ()
    if cfg.eeg.drop_interpolated:
        dropped_eeg = _channels_interpolated_in_any_run(recordings)
    return [preprocess_recording(recording, cfg, dropped_eeg) for recording in recordings]


def _channels_interpolated_in_any_run(recordings: list[PrunedRecording]) -> tuple[str, ...]:
    """The EEG channels pruning interpolated in at least one of the runs, sorted by name."""
    return tuple(sorted({ch for recording in recordings for ch in recording.bad_channels.eeg}))


def preprocess_recording(recording: PrunedRecording, cfg: PreprocessingConfig,
                         dropped_eeg: tuple[str, ...] = ()) -> PreprocessedRecording:
    """
    Preprocess one run.

    1. EEG: broadband filter and ICA, then the feature (band power or band-passed),
       without the channels in `dropped_eeg`.
    2. fNIRS: amplitude to band-passed HbO.
    3. Ground truth: the same transforms, placed on the fNIRS time axis; non-finite
       samples at the edges of that axis are dropped.
    4. mSPoC only: fNIRS and ground truth downsampled to fnirs_mspoc_srate.
    5. Everything cut to the stimulus window. Filtering ran on the whole recording
       first, so the window has no filter edge artefacts.
    6. One time axis. Band power: the EEG is interpolated onto the fNIRS time axis.
       Band-passed: the EEG keeps its own rate, cut into one block of k EEG
       samples per fNIRS sample (k = EEG rate / fNIRS rate), as mSPoC reads it.
       Samples where any signal is not finite are dropped.
    7. Scaling, last, so it describes exactly the samples the models see: every
       signal is z-scored per channel, except band-passed EEG, which is only centred
       since mSPoC reads band power from its variance. The data channels' stds are
       kept (`eeg_std`, `fnirs_std`).
    8. Check: the run stops the pipeline if EEG and fNIRS do not share their time
       span or the kept data covers too little of the window (cfg.checks).

    Parameters
    ----------
    recording : PrunedRecording
        One run after pruning.
    cfg : PreprocessingConfig
    dropped_eeg : tuple[str, ...]
        EEG channels left out of the feature.

    Returns
    -------
    PreprocessedRecording
    """
    bandpower = cfg.eeg_feature == "bandpower"

    # 1. EEG
    n_interpolated = len([ch for ch in recording.bad_channels.eeg
                          if ch in recording.eeg.ch_names])
    # Band-passed EEG is mSPoC's input, which needs a full-rank channel covariance:
    # removing ICA components would break that, so it stops the run instead.
    eeg_clean = clean_eeg(recording.eeg, n_interpolated, cfg.eeg, allow_removal=bandpower)
    if bandpower:
        eeg = band_power(eeg_clean, cfg.eeg.feature_band, cfg.fnirs_srate,
                         cfg.downsample_method)
    else:
        eeg = band_passed(eeg_clean, cfg.eeg.feature_band)
    eeg = eeg.drop_sel(channel=[ch for ch in dropped_eeg if ch in eeg.channel.values])

    # 2. fNIRS
    fnirs = fnirs_to_hbo(recording.fnirs, recording.geo3d, cfg.fnirs)

    # 3. Ground truth, on the fNIRS time axis. Its edges can fall slightly outside
    #    the ground truth's own span; those samples are dropped before any filter.
    sources = {}
    if recording.ground_truth is not None:
        sources["s_eeg"], sources["s_fnirs"] = ground_truth_sources(
            recording.ground_truth, fnirs.time, cfg)
        span = finite_span(fnirs, *sources.values())
        fnirs = fnirs.isel(time=span)
        sources = {name: s.isel(time=span) for name, s in sources.items()}

    # 4. mSPoC: lower fNIRS rate. The ground truth is already on the fNIRS time axis,
    #    so the same downsampling keeps it there.
    if cfg.fnirs_mspoc_srate is not None:
        fnirs = downsample(fnirs, cfg.fnirs_mspoc_srate)
        sources = {name: downsample(s, cfg.fnirs_mspoc_srate) for name, s in sources.items()}

    # 5. and 6. Stimulus window and one time axis
    window = stimulus_window(recording.stim, cfg.trim_pre_sec, cfg.trim_post_sec)
    eeg, fnirs, sources = align_to_window(eeg, fnirs, sources, window, same_axis=bandpower)

    # 7. Scaling. The stds are kept: they scale the ground-truth mixing for the scores.
    eeg_std = _channel_std(eeg) if bandpower else None
    fnirs_std = _channel_std(fnirs)
    eeg = zscore(eeg) if bandpower else eeg - eeg.mean("time")
    fnirs = zscore(fnirs)
    sources = {name: zscore(s) for name, s in sources.items()}

    preprocessed = PreprocessedRecording(
        group=recording.group, run=recording.run,
        eeg=as_timeseries(eeg), fnirs=as_timeseries(fnirs),
        s_eeg=_as_timeseries_or_none(sources.get("s_eeg")),
        s_fnirs=_as_timeseries_or_none(sources.get("s_fnirs")),
        geo3d=recording.geo3d, eeg_geo3d=_electrode_positions(recording.eeg),
        fnirs_optodes=recording.fnirs_optodes, stim=recording.stim, window=window,
        ground_truth=recording.ground_truth, eeg_std=eeg_std, fnirs_std=fnirs_std,
        blocks=recording.blocks,
    )

    # 8. Check
    check_alignment(preprocessed, cfg.checks)
    return preprocessed


def _electrode_positions(raw: mne.io.Raw) -> xr.DataArray:
    """The positions of every EEG channel, as the montage set at loading placed them."""
    names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    return electrode_positions(raw.get_montage(), names)


def _channel_std(da: xr.DataArray) -> xr.DataArray:
    """The std `zscore` divides by, dims (channel,), without other coordinates."""
    std = da.std("time")
    return std.drop_vars([c for c in std.coords if c != "channel"])


def _as_timeseries_or_none(da: xr.DataArray | None) -> xr.DataArray | None:
    return None if da is None else as_timeseries(da)
