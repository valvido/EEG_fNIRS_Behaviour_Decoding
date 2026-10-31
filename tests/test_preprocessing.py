"""
Tests for preprocessing: the stimulus window, the time-axis helpers, that the EEG
feature is band power, that an event stays at its time through the whole cut and
alignment, which runs the alignment check stops, and that the ground truth goes
through the same transforms as the data.
"""
import dataclasses

import mne
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.config import AlignmentCheckConfig, PreprocessingConfig
from src.preprocessing.align import (align_to_window, cut_to_blocks, finite_span,
                                     finite_times, on_time_axis, stimulus_window, zscore)
from src.preprocessing.check import check_alignment
from src.preprocessing.eeg import band_passed, band_power, clean_eeg
from src.preprocessing.fnirs import band_pass, downsample
from src.preprocessing.ground_truth import ground_truth_sources
from src.preprocessing.split import split_into_blocks
from src.types import GroundTruth, PreprocessedRecording


def make_series(values: np.ndarray, rate: float, start: float = 0.0) -> xr.DataArray:
    """A (time, channel) array with one channel per column of `values`."""
    values = values.reshape(len(values), -1)
    time = start + np.arange(len(values)) / rate
    return xr.DataArray(values, dims=("time", "channel"),
                        coords={"time": time, "channel": [f"ch{i}" for i in range(values.shape[1])]})


# -------------------------------------------------------------------------
# Stimulus window
# -------------------------------------------------------------------------
def test_stimulus_window_spans_first_onset_to_last_stimulus_end():
    stim = pd.DataFrame({"onset": [30.0, 10.0, 20.0], "duration": [4.0, 4.0, 15.0],
                         "trial_type": ["left", "right", "left"]})
    # Last end is 20 + 15 = 35, not the last onset's 30 + 4.
    assert stimulus_window(stim, pre_sec=5.0, post_sec=8.0) == pytest.approx((5.0, 43.0))


def test_stimulus_window_without_stimuli_raises():
    stim = pd.DataFrame({"onset": [], "duration": [], "trial_type": []})
    with pytest.raises(ValueError, match="no stimuli"):
        stimulus_window(stim, pre_sec=5.0, post_sec=8.0)


# -------------------------------------------------------------------------
# Time-axis helpers
# -------------------------------------------------------------------------
def test_finite_times_drops_a_sample_where_any_array_is_not_finite():
    a = make_series(np.array([1.0, 2.0, 3.0, 4.0]), rate=1.0)
    b = make_series(np.array([[1.0, 1.0], [np.nan, 1.0], [1.0, 1.0], [1.0, np.inf]]), rate=1.0)
    assert finite_times(a, b).tolist() == [True, False, True, False]


def test_finite_span_trims_only_the_edges():
    a = make_series(np.array([np.nan, 1.0, np.nan, 1.0, 1.0, np.nan]), rate=1.0)
    assert finite_span(a) == slice(1, 5)


def test_zscore_gives_each_channel_mean_zero_and_std_one():
    rng = np.random.default_rng(0)
    da = make_series(3.0 + 2.0 * rng.standard_normal((500, 3)), rate=10.0)
    z = zscore(da)
    np.testing.assert_allclose(z.mean("time"), 0.0, atol=1e-12)
    np.testing.assert_allclose(z.std("time"), 1.0)


def test_on_time_axis_interpolates_and_leaves_nan_outside_the_span():
    da = make_series(np.array([0.0, 1.0, 2.0]), rate=1.0)          # t = 0, 1, 2
    out = on_time_axis(da, xr.DataArray([0.5, 1.5, 2.5], dims="time"))
    np.testing.assert_allclose(out.values[:2, 0], [0.5, 1.5])
    assert np.isnan(out.values[2, 0])


def test_downsample_refuses_to_upsample():
    da = make_series(np.zeros(100), rate=2.0)
    with pytest.raises(ValueError, match="already at"):
        downsample(da, rate=5.0)


# -------------------------------------------------------------------------
# Ground truth goes through the same transforms as the data
# -------------------------------------------------------------------------
EEG_RATE, FNIRS_RATE, DURATION_S = 250.0, 7.4, 60.0


def alpha_bursts(seed: int) -> np.ndarray:
    """A 10 Hz carrier whose amplitude changes slowly: a band power course to find."""
    t = np.arange(0.0, DURATION_S, 1 / EEG_RATE)
    envelope = 1.0 + 0.8 * np.sin(2 * np.pi * 0.05 * t + seed)
    noise = 0.1 * np.random.default_rng(seed).standard_normal(t.size)
    return envelope * np.sin(2 * np.pi * 10.0 * t) + noise


def slow_course(seed: int) -> np.ndarray:
    """A slow haemodynamic-like course plus drift, at the fNIRS rate."""
    t = np.arange(0.0, DURATION_S, 1 / FNIRS_RATE)
    return np.sin(2 * np.pi * 0.08 * t + seed) + 0.5 * t / DURATION_S


def test_ground_truth_sources_match_the_data_transform():
    cfg = PreprocessingConfig()
    s_eeg = np.stack([alpha_bursts(0), alpha_bursts(1)], axis=1)
    s_fnirs = np.stack([slow_course(0), slow_course(1)], axis=1)
    fnirs_time = np.arange(0.0, DURATION_S, 1 / FNIRS_RATE)
    eeg_time = np.arange(0.0, DURATION_S, 1 / EEG_RATE)
    ground_truth = GroundTruth(
        s_eeg=xr.DataArray(s_eeg, dims=("time", "source"),
                           coords={"time": eeg_time, "source": ["right_hand", "left_hand"]}),
        s_fnirs=xr.DataArray(np.stack([s_fnirs, -s_fnirs], axis=-1), dims=("time", "source", "chromo"),
                             coords={"time": fnirs_time, "source": ["right_hand", "left_hand"],
                                     "chromo": ["HbO", "HbR"]}),
        A_eeg=xr.DataArray(), A_fnirs=xr.DataArray(),
    )
    time = xr.DataArray(fnirs_time, dims="time")
    got_eeg, got_fnirs = ground_truth_sources(ground_truth, time, cfg)

    # The same signals as data, through the data's own functions.
    raw = mne.io.RawArray(s_eeg.T, mne.create_info(["right_hand", "left_hand"], EEG_RATE, "eeg"), verbose=False)
    want_eeg = on_time_axis(band_power(raw, cfg.eeg.feature_band, cfg.fnirs_srate,
                                       cfg.downsample_method), time)
    want_fnirs = band_pass(make_series(s_fnirs, FNIRS_RATE), cfg.fnirs.bandpass)

    assert got_eeg.dims == got_fnirs.dims == ("time", "source")
    np.testing.assert_allclose(got_eeg.values, want_eeg.values, equal_nan=True)
    np.testing.assert_allclose(got_fnirs.values, want_fnirs.values)


# -------------------------------------------------------------------------
# EEG feature
# -------------------------------------------------------------------------
def test_band_power_is_the_squared_amplitude():
    # Power, not amplitude: a 10 Hz sine of amplitude a gives a ** 2 away from the edges.
    t = np.arange(0.0, DURATION_S, 1 / EEG_RATE)
    amplitudes = np.array([1.0, 3.0])
    raw = mne.io.RawArray(amplitudes[:, None] * np.sin(2 * np.pi * 10.0 * t),
                          mne.create_info(["C3", "C4"], EEG_RATE, "eeg"), verbose=False)
    cfg = PreprocessingConfig()
    power = band_power(raw, cfg.eeg.feature_band, cfg.fnirs_srate, cfg.downsample_method)
    middle = power.sel(time=slice(10.0, DURATION_S - 10.0))
    np.testing.assert_allclose(middle.values, np.broadcast_to(amplitudes ** 2, middle.shape),
                               rtol=0.02)


# -------------------------------------------------------------------------
# Timing: an event at a known time stays at that time
# -------------------------------------------------------------------------
# Both modalities get an event at EVENT_S on the shared clock: an alpha burst in the EEG
# and a haemodynamic-like bump in the fNIRS. The fNIRS starts slightly after the EEG, as
# in the real files. Through the real feature, filter, cut and alignment functions, both
# peaks must stay at EVENT_S; a shifted EEG or fNIRS time axis would move them.
EVENT_S, RECORDING_S = 40.0, 120.0
EEG_HZ, FNIRS_HZ, FNIRS_START_S = 250.0, 7.4, 0.115
STIM = pd.DataFrame({"onset": [30.0, 50.0], "duration": [4.0, 4.0],
                     "trial_type": ["left", "right"]})


def eeg_with_alpha_burst() -> mne.io.Raw:
    """Two EEG channels: a 10 Hz carrier whose amplitude peaks at EVENT_S."""
    t = np.arange(0.0, RECORDING_S, 1 / EEG_HZ)
    envelope = 0.1 + np.exp(-(t - EVENT_S) ** 2 / (2 * 1.0 ** 2))
    data = np.stack([envelope * np.sin(2 * np.pi * 10.0 * t)] * 2)
    return mne.io.RawArray(data, mne.create_info(["C3", "C4"], EEG_HZ, "eeg"), verbose=False)


def fnirs_with_bump() -> xr.DataArray:
    """Two HbO channels with a slow bump peaking at EVENT_S, starting at FNIRS_START_S."""
    t = FNIRS_START_S + np.arange(0.0, RECORDING_S - FNIRS_START_S, 1 / FNIRS_HZ)
    bump = np.exp(-(t - EVENT_S) ** 2 / (2 * 3.0 ** 2))
    return make_series(np.stack([bump, bump], axis=1), FNIRS_HZ, start=FNIRS_START_S)


def peak_time(da: xr.DataArray) -> float:
    return float(da.isel(channel=0).idxmax("time"))


def test_band_power_and_hbo_keep_the_event_time_through_cut_and_alignment():
    cfg = PreprocessingConfig()
    eeg = band_power(eeg_with_alpha_burst(), cfg.eeg.feature_band, cfg.fnirs_srate,
                     cfg.downsample_method)
    fnirs = band_pass(fnirs_with_bump(), cfg.fnirs.bandpass)
    window = stimulus_window(STIM, cfg.trim_pre_sec, cfg.trim_post_sec)
    eeg, fnirs, sources = align_to_window(eeg, fnirs, {"s_fnirs": fnirs.copy()}, window,
                                          same_axis=True)

    # One time axis for everything, inside the window and filling it.
    np.testing.assert_array_equal(eeg.time.values, fnirs.time.values)
    np.testing.assert_array_equal(sources["s_fnirs"].time.values, fnirs.time.values)
    assert window[0] <= float(fnirs.time[0]) < window[0] + 1 / FNIRS_HZ
    assert window[1] - 1 / FNIRS_HZ < float(fnirs.time[-1]) <= window[1]

    # The events stay where they were put, to within one sample.
    assert peak_time(eeg) == pytest.approx(EVENT_S, abs=1 / FNIRS_HZ)
    assert peak_time(fnirs) == pytest.approx(EVENT_S, abs=1 / FNIRS_HZ)


def test_band_passed_eeg_keeps_its_rate_over_the_fnirs_span_and_the_event_time():
    cfg = PreprocessingConfig()
    eeg = band_passed(eeg_with_alpha_burst(), cfg.eeg.feature_band)
    fnirs = downsample(band_pass(fnirs_with_bump(), cfg.fnirs.bandpass), rate=1.0)
    window = stimulus_window(STIM, cfg.trim_pre_sec, 20.0)
    eeg, fnirs, _ = align_to_window(eeg, fnirs, {}, window, same_axis=False)

    # EEG at its own rate, starting with the fNIRS, one block of k samples per fNIRS
    # sample: the last block ends one fNIRS sample after the last fNIRS time point.
    k = int(EEG_HZ / 1.0)
    assert 1 / float(np.median(np.diff(eeg.time.values))) == pytest.approx(EEG_HZ)
    assert eeg.sizes["time"] == k * fnirs.sizes["time"]
    assert float(eeg.time[0]) - float(fnirs.time[0]) == pytest.approx(0.0, abs=1 / EEG_HZ)
    assert float(eeg.time[-1]) - float(fnirs.time[-1]) == pytest.approx(1.0 - 1 / EEG_HZ,
                                                                        abs=1 / EEG_HZ)

    # The burst, read as power over 1 s, and the bump stay at the event time.
    power = (eeg ** 2).rolling(time=int(EEG_HZ), center=True).mean()
    assert peak_time(power) == pytest.approx(EVENT_S, abs=0.2)
    assert peak_time(fnirs) == pytest.approx(EVENT_S, abs=1.0)


def test_cut_to_blocks_puts_each_eeg_block_on_its_fnirs_sample():
    # EEG at 250 Hz cut to the fNIRS span, as before the fix: one block short.
    fnirs = make_series(np.arange(40.0), 1.0, start=10.0)
    eeg = make_series(np.arange(10.0, 49.0 + 1e-9, 1 / 250), 250.0, start=10.0)
    source = fnirs.rename(channel="source")
    eeg, fnirs, sources = cut_to_blocks(eeg, fnirs, {"s_fnirs": source})

    # The last fNIRS sample has no full block and is dropped with its ground truth.
    assert fnirs.sizes["time"] == sources["s_fnirs"].sizes["time"] == 39
    assert eeg.sizes["time"] == 250 * 39
    # Block i, as mSPoC cuts it, starts exactly at fNIRS sample i.
    k = eeg.sizes["time"] // fnirs.sizes["time"]
    block_starts = eeg.time.values[::k]
    np.testing.assert_allclose(block_starts, fnirs.time.values)


def test_cut_to_blocks_needs_a_whole_number_of_eeg_samples_per_fnirs_sample():
    fnirs = make_series(np.zeros(10), 3.0)
    eeg = make_series(np.zeros(1000), 250.0)
    with pytest.raises(ValueError, match="must divide the EEG rate"):
        cut_to_blocks(eeg, fnirs, {})


def eeg_with_eog() -> mne.io.Raw:
    """Four EEG channels of noise and one EOG channel, for ICA."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((5, int(20 * EEG_HZ))) * 1e-5
    info = mne.create_info(["Fp1", "Fp2", "C3", "C4", "EOG"], EEG_HZ,
                           ["eeg", "eeg", "eeg", "eeg", "eog"])
    return mne.io.RawArray(data, info, verbose=False)


@pytest.mark.parametrize("found, allow, raises", [
    ([], False, False), ([1], True, False), ([1], False, True)])
def test_clean_eeg_refuses_to_remove_eye_components_when_told(monkeypatch, found, allow, raises):
    monkeypatch.setattr(mne.preprocessing.ICA, "find_bads_eog",
                        lambda self, raw, verbose=None: (list(found), None))
    cfg = PreprocessingConfig().eeg
    if raises:
        with pytest.raises(ValueError, match=r"ICA found 1 eye component\(s\) \[1\].*mSPoC"):
            clean_eeg(eeg_with_eog(), 0, cfg, allow_removal=allow)
    else:
        clean_eeg(eeg_with_eog(), 0, cfg, allow_removal=allow)


def test_clean_eeg_without_eog_skips_ica_with_a_warning(monkeypatch):
    def fail_if_fitted(*args, **kwargs):
        raise AssertionError("ICA must not be fitted without an EOG channel.")

    monkeypatch.setattr(mne.preprocessing.ICA, "fit", fail_if_fitted)
    eeg = eeg_with_eog().drop_channels(["EOG"])
    cfg = PreprocessingConfig().eeg
    with pytest.warns(UserWarning, match="no EOG channel.*ICA is skipped"):
        cleaned = clean_eeg(eeg, 0, cfg, allow_removal=False)
    filtered = eeg.copy().filter(*cfg.bandpass, method="fir", phase="zero", verbose=False)
    np.testing.assert_allclose(cleaned.get_data(), filtered.get_data())


# -------------------------------------------------------------------------
# Alignment check: which runs stop the pipeline
# -------------------------------------------------------------------------
CHECKS = AlignmentCheckConfig()           # 90% coverage
RATE, N = 7.4, 900                        # about 120 s, like a run's stimulus window


def smooth_noise(seed: int, n: int = N + 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.convolve(rng.standard_normal(n), np.ones(9) / 9, mode="same")


def make_run(window=(0.0, N / RATE), eeg_time_offset: float = 0.0) -> PreprocessedRecording:
    """
    A preprocessed run whose data contain the ground-truth sources.
    """
    rng = np.random.default_rng(1)
    s_eeg = np.stack([smooth_noise(2), smooth_noise(3)], axis=1)
    s_fnirs = np.stack([smooth_noise(4), smooth_noise(5)], axis=1)
    eeg = s_eeg + 0.05 * rng.standard_normal(s_eeg.shape)
    fnirs = s_fnirs + 0.05 * rng.standard_normal(s_fnirs.shape)
    cut = slice(20, 20 + N)

    def series(values, dim="channel", offset=0.0):
        da = make_series(values[cut], RATE)
        da = da.assign_coords(time=da.time + offset)
        return da.rename(channel=dim) if dim != "channel" else da

    sources = lambda v: series(v, "source").assign_coords(source=["right_hand", "left_hand"])
    return PreprocessedRecording(
        group="g", run="run_001", eeg=series(eeg, offset=eeg_time_offset), fnirs=series(fnirs),
        s_eeg=sources(s_eeg), s_fnirs=sources(s_fnirs), geo3d=xr.DataArray(),
        eeg_geo3d=xr.DataArray(), fnirs_optodes={},
        stim=pd.DataFrame(), window=window, ground_truth=None)


def test_check_passes_an_aligned_run():
    check_alignment(make_run(), CHECKS)


def test_check_stops_a_run_whose_data_covers_too_little_of_the_window():
    run = make_run(window=(0.0, 2 * N / RATE))                       # data covers 50%
    with pytest.raises(ValueError, match="covers 50% of the stimulus window"):
        check_alignment(run, CHECKS)


def test_check_stops_a_run_whose_eeg_is_not_on_the_fnirs_time_axis():
    run = make_run(eeg_time_offset=0.5 / RATE)                       # half a sample off
    with pytest.raises(ValueError, match="do not cover the same time span"):
        check_alignment(run, CHECKS)


# -------------------------------------------------------------------------
# Splitting a session into its blocks
# -------------------------------------------------------------------------
BLOCKS = {"block_1": (0.0, 30.0), "block_2": (30.0, 60.0)}


def session(eeg_per_fnirs: int = 1, fnirs_hz: float = 2.0) -> PreprocessedRecording:
    """
    A 60 s preprocessed session of two blocks whose every value is its own time, with
    one task cue per block. `eeg_per_fnirs` > 1 gives band-passed EEG at that many
    samples per fNIRS sample.
    """
    fnirs_time = np.arange(0.0, 60.0, 1 / fnirs_hz)
    eeg_time = np.arange(len(fnirs_time) * eeg_per_fnirs) / (fnirs_hz * eeg_per_fnirs)
    stim = pd.DataFrame({"onset": [5.0, 35.0], "duration": 6.0, "value": 1.0,
                         "trial_type": ["right", "left"]})
    return PreprocessedRecording(
        group="session", run="S1", eeg=make_series(eeg_time, fnirs_hz * eeg_per_fnirs),
        fnirs=make_series(fnirs_time, fnirs_hz), s_eeg=None, s_fnirs=None,
        geo3d=xr.DataArray(), eeg_geo3d=xr.DataArray(), fnirs_optodes={}, stim=stim,
        window=(-2.0, 70.0), ground_truth=None, blocks=BLOCKS)


@pytest.mark.parametrize("eeg_per_fnirs", [1, 4])
def test_split_cuts_every_sample_into_exactly_one_block(eeg_per_fnirs):
    whole = session(eeg_per_fnirs)
    blocks = split_into_blocks([whole], groups=("block_1", "block_2"))
    assert [(b.group, b.run, b.blocks) for b in blocks] == [("block_1", "S1", None),
                                                           ("block_2", "S1", None)]
    for name in ("eeg", "fnirs"):
        joined = np.concatenate([getattr(b, name).values.ravel() for b in blocks])
        np.testing.assert_array_equal(joined, getattr(whole, name).values.ravel())
    first, second = blocks
    assert float(first.fnirs.time[-1]) < 30.0 <= float(second.fnirs.time[0])
    # Every fNIRS sample keeps the EEG samples that follow it.
    assert first.eeg.sizes["time"] == eeg_per_fnirs * first.fnirs.sizes["time"]
    assert float(second.eeg.time[0]) == float(second.fnirs.time[0])


def test_split_gives_each_block_its_cues_and_its_part_of_the_window():
    first, second = split_into_blocks([session()], groups=("block_1", "block_2"))
    assert list(first.stim.onset) == [5.0] and list(second.stim.onset) == [35.0]
    assert first.window == (0.0, 30.0) and second.window == (30.0, 60.0)


def test_split_keeps_only_the_subjects_groups_and_passes_other_runs_on():
    run = dataclasses.replace(session(), group="other", blocks=None)
    out = split_into_blocks([session(), run], groups=("block_2",))
    assert [r.group for r in out] == ["block_2", "other"]
    assert out[1] is run


def test_split_refuses_eeg_without_a_whole_number_of_samples_per_fnirs_sample():
    whole = session(eeg_per_fnirs=4)
    broken = dataclasses.replace(whole, eeg=whole.eeg.isel(time=slice(0, -1)))
    with pytest.raises(ValueError, match="whole number"):
        split_into_blocks([broken], groups=("block_1",))
