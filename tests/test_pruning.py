"""
Tests for pruning: the fNIRS window inside the rest period, the whole rest recording
for the EEG, the fNIRS bad-channel detection on the cleaned amplitude, and the stop
on a bad motor electrode.
"""
import dataclasses

import mne
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import src.pruning.prune as prune
from src.config import FnirsPruningConfig, PruningConfig
from src.pruning.fnirs import clean_amplitude, detect_bad_fnirs_channels
from src.pruning.segment import rest_window
from src.types import BadChannels, RawRecording

DURATION_S = 60.0


def make_rest(onsets: list[float], duration_s: float) -> RawRecording:
    """
    A 60 s rest recording: 2 EEG channels at 100 Hz, 2 fNIRS channels at 10 Hz, one
    stim event per onset. Every sample's value is its own time in seconds, so the
    first and last values of a cut show where it was made.
    """
    info = mne.create_info(["C3", "C4"], sfreq=100.0, ch_types="eeg")
    eeg_time = np.arange(int(DURATION_S * 100) + 1) / 100.0
    eeg = mne.io.RawArray(np.tile(eeg_time, (2, 1)), info, verbose=False)
    time = np.arange(0.0, DURATION_S + 0.05, 0.1)
    fnirs = xr.DataArray(np.broadcast_to(time, (2, 2, time.size)).copy(),
                         dims=("channel", "wavelength", "time"),
                         coords={"channel": ["S1D1", "S1D2"], "wavelength": [760, 850],
                                 "time": time, "source": ("channel", ["S1", "S1"]),
                                 "detector": ("channel", ["D1", "D2"])})
    stim = pd.DataFrame({"onset": onsets, "duration": [duration_s] * len(onsets),
                         "trial_type": ["rest"] * len(onsets)})
    return RawRecording(group="rest", run="rest", eeg=eeg, fnirs=fnirs,
                        geo3d=xr.DataArray(), stim=stim)


# -------------------------------------------------------------------------
# Window
# -------------------------------------------------------------------------
def test_window_starts_at_rest_onset_and_lasts_window_sec():
    assert rest_window(make_rest([10.0], duration_s=40.0), window_sec=25.0) == \
        pytest.approx((10.0, 35.0))


def test_window_ends_with_the_rest_period():
    assert rest_window(make_rest([10.0], duration_s=15.0), window_sec=25.0) == \
        pytest.approx((10.0, 25.0))


def test_window_ends_with_the_recorded_data():
    assert rest_window(make_rest([50.0], duration_s=40.0), window_sec=25.0) == \
        pytest.approx((50.0, DURATION_S))


@pytest.mark.parametrize("onsets", [[], [10.0, 30.0]])
def test_rest_without_exactly_one_stim_event_raises(onsets):
    with pytest.raises(ValueError, match="exactly one stim event"):
        rest_window(make_rest(onsets, duration_s=10.0), window_sec=25.0)


# -------------------------------------------------------------------------
# fNIRS detection
# -------------------------------------------------------------------------
FNIRS_FS = 10.0
FNIRS_DURATION_S = 120.0
WINDOW = (40.0, 65.0)


def make_fnirs(signals: dict[str, np.ndarray], distances_mm: dict[str, float]):
    """
    fNIRS amplitude (V) with one source and one detector per channel, the detector
    `distances_mm[ch]` from the source. Both wavelengths of a channel get the same
    signal.
    """
    channels = list(signals)
    time = np.arange(0.0, FNIRS_DURATION_S, 1 / FNIRS_FS)
    data = np.stack([[signals[ch], signals[ch]] for ch in channels])
    amp = xr.DataArray(
        data, dims=("channel", "wavelength", "time"),
        coords={"channel": channels, "wavelength": [760.0, 850.0], "time": time,
                "samples": ("time", np.arange(time.size)),
                "source": ("channel", ["S1"] * len(channels)),
                "detector": ("channel", [f"D{i}" for i in range(len(channels))])},
    ).pint.quantify("V")
    amp = amp.assign_coords(time=amp.time.assign_attrs(units="s"))

    labels = ["S1"] + [f"D{i}" for i in range(len(channels))]
    positions = [[0.0, 0.0, 0.0]] + [[distances_mm[ch], 0.0, 0.0] for ch in channels]
    geo3d = xr.DataArray(
        positions, dims=("label", "digitized"),
        coords={"label": labels,
                "type": ("label", ["source"] + ["detector"] * len(channels))},
    ).pint.quantify("mm")
    return amp, geo3d


def heartbeat(seed: int) -> np.ndarray:
    """A channel with good coupling: a strong 1.1 Hz heartbeat and little noise."""
    time = np.arange(0.0, FNIRS_DURATION_S, 1 / FNIRS_FS)
    noise = np.random.default_rng(seed).standard_normal(time.size)
    return 0.5 + 0.02 * np.sin(2 * np.pi * 1.1 * time) + 0.002 * noise


def noise_only(seed: int) -> np.ndarray:
    """
    A channel with no coupling: no heartbeat, and noise large next to its mean. The
    noise is multiplicative so the amplitude stays positive, as int2od requires.
    """
    time = np.arange(0.0, FNIRS_DURATION_S, 1 / FNIRS_FS)
    return 0.5 * np.exp(0.5 * np.random.default_rng(seed).standard_normal(time.size))


def test_clean_amplitude_covers_only_the_window():
    amp, _ = make_fnirs({"S1D0": heartbeat(0)}, {"S1D0": 30.0})
    cleaned = clean_amplitude(amp, WINDOW, FnirsPruningConfig())
    assert float(cleaned.time[0]) >= WINDOW[0]
    assert float(cleaned.time[-1]) <= WINDOW[1]


def test_clean_amplitude_keeps_the_raw_window_mean():
    amp, _ = make_fnirs({"S1D0": heartbeat(0)}, {"S1D0": 30.0})
    cleaned = clean_amplitude(amp, WINDOW, FnirsPruningConfig())
    raw_mean = amp.sel(time=slice(*WINDOW)).mean("time")
    np.testing.assert_allclose(cleaned.mean("time").pint.magnitude,
                               raw_mean.pint.magnitude, rtol=1e-2)


def test_coupled_channel_is_kept_and_noise_channel_fails_snr_and_sci_psp():
    amp, geo3d = make_fnirs({"S1D0": heartbeat(0), "S1D1": noise_only(1)},
                            {"S1D0": 30.0, "S1D1": 30.0})
    bad, detail = detect_bad_fnirs_channels(amp, geo3d, WINDOW, FnirsPruningConfig())
    assert bad == ("S1D1",)
    assert detail["snr"] == ("S1D1",)
    assert detail["sci_psp"] == ("S1D1",)
    assert detail["sd_dist"] == ()
    assert detail["mean_amp"] == ()


def test_channel_outside_the_distance_range_fails_sd_dist():
    amp, geo3d = make_fnirs({"S1D0": heartbeat(0), "S1D1": heartbeat(1)},
                            {"S1D0": 30.0, "S1D1": 5.0})
    bad, detail = detect_bad_fnirs_channels(amp, geo3d, WINDOW, FnirsPruningConfig())
    assert bad == ("S1D1",)
    assert detail["sd_dist"] == ("S1D1",)


def test_eeg_is_judged_on_the_whole_rest_and_fnirs_on_the_window(monkeypatch):
    seen = {}

    def fake_eeg(eeg, cfg, random_state):
        seen["eeg"] = eeg
        return (), {}

    def fake_fnirs(amp, geo3d, window, cfg):
        seen["window"] = window
        return (), {}

    monkeypatch.setattr(prune, "detect_bad_eeg_channels", fake_eeg)
    monkeypatch.setattr(prune, "detect_bad_fnirs_channels", fake_fnirs)
    rest = make_rest([10.0], duration_s=40.0)
    prune.detect_bad_channels(rest, PruningConfig(window_sec=25.0))
    assert seen["eeg"].times[0] == 0.0
    assert seen["eeg"].times[-1] == pytest.approx(DURATION_S)
    assert seen["window"] == pytest.approx((10.0, 35.0))


# -------------------------------------------------------------------------
# Motor electrodes
# -------------------------------------------------------------------------
def fake_detection(monkeypatch, bad_eeg: tuple[str, ...]):
    """Make detect_bad_channels report `bad_eeg` as the bad EEG channels."""
    bad = BadChannels(eeg=bad_eeg, fnirs=(), eeg_detail={}, fnirs_detail={})
    monkeypatch.setattr(prune, "detect_bad_channels", lambda rest, cfg: bad)


@pytest.mark.parametrize("bad_eeg", [("C3",), ("C4",), ("C3", "Fz", "C4")])
def test_bad_motor_electrode_stops_pruning(monkeypatch, bad_eeg):
    fake_detection(monkeypatch, bad_eeg)
    rest = make_rest([10.0], duration_s=40.0)
    with pytest.raises(ValueError, match="cannot be used"):
        prune.prune_recordings([rest], rest, PruningConfig())


def test_bad_non_motor_electrode_does_not_stop_pruning(monkeypatch):
    fake_detection(monkeypatch, ("Fz",))
    monkeypatch.setattr(prune, "remove_bad_channels", lambda recording, bad: recording)
    rest = make_rest([10.0], duration_s=40.0)
    pruned, bad = prune.prune_recordings([rest], rest, PruningConfig())
    assert bad.eeg == ("Fz",)
    assert len(pruned) == 1


def test_pruned_run_keeps_the_optodes_of_every_channel_and_its_blocks():
    blocks = {"block_1": (0.0, 30.0), "block_2": (30.0, DURATION_S)}
    recording = dataclasses.replace(make_rest([10.0], duration_s=40.0), blocks=blocks)
    bad = BadChannels(eeg=(), fnirs=("S1D2",), eeg_detail={}, fnirs_detail={})
    pruned = prune.remove_bad_channels(recording, bad)
    assert list(pruned.fnirs.channel.values) == ["S1D1"]
    assert pruned.fnirs_optodes == {"S1D1": ("S1", "D1"), "S1D2": ("S1", "D2")}
    assert pruned.blocks == blocks
