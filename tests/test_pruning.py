"""
Tests for pruning: the window inside the rest period, what each modality is judged
on, the fNIRS bad-channel detection on the cleaned amplitude, and the stop on a bad motor
electrode.
"""
import dataclasses

import mne
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import src.pruning.eeg as eeg_module
import src.pruning.prune as prune
from src.config import EegPruningConfig, FnirsPruningConfig, PruningConfig
from src.pruning.fnirs import clean_amplitude, detect_bad_fnirs_channels
from src.pruning.eeg import detect_bad_eeg_channels_by_recording
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
# What each modality is judged on
# -------------------------------------------------------------------------
def test_eeg_is_judged_whole_on_its_recordings_and_fnirs_on_the_rest_window(monkeypatch):
    seen = {}

    def judge_eeg(eegs, cfg, random_state):
        seen["eeg_end"] = eegs["background"].times[-1]
        return {}, {}

    def judge_fnirs(amp, geo3d, window, cfg):
        seen["fnirs_end"], seen["window"] = float(amp.time[-1]), window
        return (), {}

    monkeypatch.setattr(prune, "detect_bad_eeg_channels_by_recording", judge_eeg)
    monkeypatch.setattr(prune, "detect_bad_fnirs_channels", judge_fnirs)
    reference = make_rest([10.0], duration_s=40.0)
    background = dataclasses.replace(reference, run="background")
    prune.detect_bad_channels([reference], [background], PruningConfig(window_sec=25.0))
    assert seen["eeg_end"] == pytest.approx(DURATION_S)
    assert seen["fnirs_end"] == pytest.approx(DURATION_S)
    assert seen["window"] == pytest.approx((10.0, 35.0))


# -------------------------------------------------------------------------
# EEG per recording
# -------------------------------------------------------------------------
FLAGGED = {"r1": ("C3", "Fz"), "r2": ("Fz", "Pz")}


def test_every_recording_gets_its_own_channels_and_all_criteria_are_reported(monkeypatch):
    monkeypatch.setattr(eeg_module, "detect_bad_eeg_channels", lambda eeg, cfg, seed: (
        FLAGGED[eeg], {"pass1": {"bad_by_flat": FLAGGED[eeg]}}))
    by_recording, detail = detect_bad_eeg_channels_by_recording(
        {name: name for name in FLAGGED}, EegPruningConfig(), random_state=0)
    assert by_recording == FLAGGED
    assert detail == {"pass1": {"bad_by_flat": ("C3", "Fz", "Pz")}}


def named_run(group: str, run: str) -> RawRecording:
    return dataclasses.replace(make_rest([10.0], duration_s=40.0), group=group, run=run)


def judge_eeg_by_name(monkeypatch, flagged: dict[str, tuple[str, ...]]):
    """Replace EEG detection by a lookup, and keep the runs as they are after pruning."""
    judged = []

    def detect(references, eeg_recordings, cfg):
        judged.extend(recording.run for recording in eeg_recordings)
        return BadChannels(eeg=(), fnirs=(), eeg_detail={}, fnirs_detail={},
                           eeg_by_recording=flagged)

    monkeypatch.setattr(prune, "detect_bad_channels", detect)
    monkeypatch.setattr(prune, "remove_bad_channels",
                        lambda recording, bad: (recording.run, bad.eeg))
    return judged


def test_run_without_a_background_gets_the_channels_of_its_own_recording(monkeypatch):
    judge_eeg_by_name(monkeypatch, {"session1": ("T7",), "session3": ("T8",)})
    runs = [named_run("session", "session1"), named_run("session", "session3")]
    pruned, _ = prune.prune_recordings(runs, [runs[0]], {}, PruningConfig())
    assert pruned == [("session1", ("T7",)), ("session3", ("T8",))]


def test_runs_of_a_group_get_the_channels_of_its_background(monkeypatch):
    judged = judge_eeg_by_name(monkeypatch, {"bg_a": ("T7",), "bg_b": ("T8",)})
    runs = [named_run("a", "run_1"), named_run("a", "run_2"), named_run("b", "run_1")]
    backgrounds = {"a": named_run("a", "bg_a"), "b": named_run("b", "bg_b")}
    pruned, _ = prune.prune_recordings(runs, [runs[0]], backgrounds, PruningConfig())
    assert pruned == [("run_1", ("T7",)), ("run_2", ("T7",)), ("run_1", ("T8",))]
    assert judged == ["bg_a", "bg_a", "bg_b"]


def test_bad_motor_electrode_in_one_recording_stops_pruning_and_names_it(monkeypatch):
    judge_eeg_by_name(monkeypatch, {"session1": (), "session3": ("C4",)})
    runs = [named_run("session", "session1"), named_run("session", "session3")]
    with pytest.raises(ValueError, match="session3"):
        prune.prune_recordings(runs, [runs[0]], {}, PruningConfig())


def test_result_lists_the_channels_bad_in_any_recording_and_each_recording_once(monkeypatch):
    calls = []

    def detect(eegs, cfg, random_state):
        calls.append(sorted(eegs))
        return {"r1": ("C3", "Fz"), "r2": ("Fz", "Pz")}, {}

    monkeypatch.setattr(prune, "detect_bad_eeg_channels_by_recording", detect)
    monkeypatch.setattr(prune, "detect_bad_fnirs_channels", lambda *args: ((), {}))
    r1, r2 = named_run("g", "r1"), named_run("g", "r2")
    bad = prune.detect_bad_channels([r1], [r1, r1, r2], PruningConfig())
    assert calls == [["r1", "r2"]]
    assert bad.eeg == ("C3", "Fz", "Pz")
    assert bad.eeg_by_recording == {"r1": ("C3", "Fz"), "r2": ("Fz", "Pz")}


def test_fnirs_channels_bad_in_any_reference_are_bad(monkeypatch):
    flagged = {"session1": (("S1D1",), {"snr": ("S1D1",)}),
               "session3": (("S1D2",), {"sci_psp": ("S1D2",)}),
               "session5": ((), {})}
    monkeypatch.setattr(prune, "detect_bad_eeg_channels_by_recording", lambda *args: ({}, {}))
    monkeypatch.setattr(prune, "detect_bad_fnirs_channels",
                        lambda amp, geo3d, window, cfg: flagged[amp.attrs["run"]])
    references = []
    for run in flagged:
        reference = named_run("session", run)
        reference.fnirs.attrs["run"] = run
        references.append(reference)
    bad = prune.detect_bad_channels(references, [], PruningConfig())
    assert bad.fnirs == ("S1D1", "S1D2")
    assert bad.fnirs_by_recording == {run: channels for run, (channels, _) in flagged.items()}
    assert bad.fnirs_detail == {"snr": ("S1D1",), "sci_psp": ("S1D2",)}


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


# -------------------------------------------------------------------------
# Motor electrodes
# -------------------------------------------------------------------------
def fake_detection(monkeypatch, bad_eeg: tuple[str, ...]):
    """Make detect_bad_channels report `bad_eeg` as the bad EEG channels."""
    bad = BadChannels(eeg=bad_eeg, fnirs=(), eeg_detail={}, fnirs_detail={},
                      eeg_by_recording={"rest": bad_eeg})
    monkeypatch.setattr(prune, "detect_bad_channels", lambda references, eeg_recordings, cfg: bad)


@pytest.mark.parametrize("bad_eeg", [("C3",), ("C4",), ("C3", "Fz", "C4")])
def test_bad_motor_electrode_stops_pruning(monkeypatch, bad_eeg):
    fake_detection(monkeypatch, bad_eeg)
    rest = make_rest([10.0], duration_s=40.0)
    with pytest.raises(ValueError, match="cannot be used"):
        prune.prune_recordings([rest], [rest], {}, PruningConfig())


def test_motor_electrodes_come_from_the_config(monkeypatch):
    fake_detection(monkeypatch, ("C3",))
    monkeypatch.setattr(prune, "remove_bad_channels", lambda recording, bad: recording)
    rest = make_rest([10.0], duration_s=40.0)
    cfg = PruningConfig(eeg=EegPruningConfig(motor_channels=("C4",)))
    pruned, bad = prune.prune_recordings([rest], [rest], {}, cfg)
    assert bad.eeg == ("C3",)


def test_motor_electrode_missing_from_the_montage_stops_pruning(monkeypatch):
    fake_detection(monkeypatch, ())
    rest = make_rest([10.0], duration_s=40.0)
    cfg = PruningConfig(eeg=EegPruningConfig(motor_channels=("C3", "FCC4h")))
    with pytest.raises(ValueError, match="FCC4h"):
        prune.prune_recordings([rest], [rest], {}, cfg)


def test_bad_non_motor_electrode_does_not_stop_pruning(monkeypatch):
    fake_detection(monkeypatch, ("Fz",))
    monkeypatch.setattr(prune, "remove_bad_channels", lambda recording, bad: recording)
    rest = make_rest([10.0], duration_s=40.0)
    pruned, bad = prune.prune_recordings([rest], [rest], {}, PruningConfig())
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
