"""
Tests for loading: the stop on non-finite raw EEG, the report of what was loaded, the
ground-truth sources labelled by hand, the Buccino and Shin readers on small MAT files
laid out like the datasets', and the montage applied from the subject config.
"""
import dataclasses
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
import scipy.io
import xarray as xr
from cedalion.dataclasses import PointType

from src.config import EegMontageConfig, FnirsMontageConfig, MontageConfig, SubjectConfig
from src.config.load import PROJECT_ROOT
from src.loading import (buccino, check_data_format, load_backgrounds, load_recordings,
                         shin2017, synthetic)
from src.loading.load import _check_eeg_finite, loading_report
from src.loading.montage import apply_montage, check_montage, read_layout
from src.loading.rest import EDGE_S, mark_rest_period
from src.loading.synthetic import _tidy_sources, source_hands
from src.types import RawRecording


class FakeSubject:
    subject_id = "de000"


def make_recording(run: str, eeg_values: np.ndarray) -> RawRecording:
    """
    A recording with 2 EEG channels and 1 EOG channel holding `eeg_values` (3 rows).
    """
    info = mne.create_info(["C3", "C4", "EOG1"], sfreq=100.0,
                           ch_types=["eeg", "eeg", "eog"])
    eeg = mne.io.RawArray(eeg_values, info, verbose=False)
    return RawRecording(group="g1", run=run, eeg=eeg, fnirs=xr.DataArray(),
                        geo3d=xr.DataArray(), stim=pd.DataFrame())


def test_finite_eeg_passes():
    _check_eeg_finite(FakeSubject(), [make_recording("run_001", np.zeros((3, 50)))])


def test_nan_in_eeg_stops_and_names_every_broken_run():
    broken = np.zeros((3, 50))
    broken[1, 0] = np.nan
    recordings = [make_recording("run_001", broken),
                  make_recording("run_002", np.zeros((3, 50))),
                  make_recording("run_003", broken)]
    with pytest.raises(ValueError) as err:
        _check_eeg_finite(FakeSubject(), recordings)
    message = str(err.value)
    assert "g1/run_001" in message and "g1/run_003" in message
    assert "run_002" not in message
    assert "1/2 EEG channels" in message and "C4" in message


def test_nan_outside_eeg_channels_is_ignored():
    values = np.zeros((3, 50))
    values[2, 0] = np.nan
    _check_eeg_finite(FakeSubject(), [make_recording("run_001", values)])


def test_loading_report_lists_runs_by_group():
    subject = FakeSubject()
    subject.paths = {"data": "data/de000"}
    recordings = [make_recording("run_001", np.zeros((3, 50))),
                  make_recording("run_002", np.zeros((3, 50)))]
    assert loading_report(subject, recordings) == (
        "Loaded 2 runs\n"
        "  g1: run_001, run_002\n"
        "  data: data/de000")


def with_fnirs(recording: RawRecording, start: float, end: float) -> RawRecording:
    """`recording` with one fNIRS channel sampled at 1 Hz from `start` to `end` (s)."""
    time = np.arange(start, end + 0.5)
    fnirs = xr.DataArray(np.ones((1, 1, len(time))), dims=("channel", "wavelength", "time"),
                         coords={"time": time})
    return dataclasses.replace(recording, fnirs=fnirs)


def synthetic_background(group: str) -> RawRecording:
    """The 175 s background recording of `group`."""
    recording = dataclasses.replace(make_recording(f"de000_{group}", np.zeros((3, 50))),
                                    group=group)
    return with_fnirs(recording, 0.0, 175.0)


def test_synthetic_reference_is_the_background_of_the_group_recorded_earliest():
    groups = ("sitting_post_rest_1", "sitting_pre_rest_2", "sitting_pre_rest_1")
    [reference] = synthetic.select_pruning_recordings(
        [], {group: synthetic_background(group) for group in groups})
    assert reference.group == "sitting_pre_rest_1"


def test_synthetic_rest_is_the_first_minute_of_the_background_without_its_edges():
    [reference] = synthetic.select_pruning_recordings(
        [], {"sitting_pre_rest_1": synthetic_background("sitting_pre_rest_1")})
    assert list(reference.stim.trial_type) == ["rest"]
    assert reference.stim.onset.iloc[0] == EDGE_S
    assert reference.stim.duration.iloc[0] == synthetic.REST_S - 2 * EDGE_S


def test_rest_period_shorter_than_its_edges_is_refused():
    recording = with_fnirs(make_recording("run_001", np.zeros((3, 50))), 0.0, 100.0)
    with pytest.raises(ValueError, match="no rest period"):
        mark_rest_period(recording, 10.0, 10.0 + 2 * EDGE_S)


def test_rest_period_starting_before_the_fnirs_is_refused():
    recording = with_fnirs(make_recording("run_001", np.zeros((3, 50))), 0.0, 100.0)
    with pytest.raises(ValueError, match="before the fNIRS"):
        mark_rest_period(recording, -10.0, 50.0)


def synthetic_subject(tmp_path, groups=("sitting_pre_rest_1", "sitting_post_rest_1")):
    return SubjectConfig(subject_id="de000", name="de000", format="synthetic",
                         paths={"data": tmp_path / "data", "background": tmp_path}, groups=groups,
                         montage=MontageConfig(eeg=STANDARD_EEG,
                                               fnirs=FnirsMontageConfig(recording=True)))


def test_synthetic_background_is_one_recording_per_group_of_the_subject(tmp_path, monkeypatch):
    read = []

    def read_files(eeg, fnirs):
        read.append((eeg.name, fnirs.name))
        return make_recording(fnirs.stem, np.zeros((3, 50)))

    monkeypatch.setattr(synthetic, "_read_rest_files", read_files)
    backgrounds = synthetic.read_background(synthetic_subject(tmp_path))
    assert read == [("de000_sitting_pre_rest_1-raw.fif", "de000_sitting_pre_rest_1.snirf"),
                    ("de000_sitting_post_rest_1-raw.fif", "de000_sitting_post_rest_1.snirf")]
    assert [background.group for background in backgrounds] == ["sitting_pre_rest_1",
                                                                "sitting_post_rest_1"]


def test_synthetic_missing_background_recording_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(synthetic, "_run_dirs", lambda subject, group: [])
    with pytest.raises(FileNotFoundError, match="background recordings do not exist"):
        synthetic.check(synthetic_subject(tmp_path))


def test_real_dataset_has_no_backgrounds(tmp_path):
    assert load_backgrounds(write_session(tmp_path / "S0.mat")) == {}
    assert load_backgrounds(write_shin_subject(tmp_path)) == {}


def test_backgrounds_are_keyed_by_group(tmp_path, monkeypatch):
    background = dataclasses.replace(make_recording("background", np.zeros((3, 50))), group="a")
    monkeypatch.setattr(synthetic, "read_background", lambda subject: [background])
    monkeypatch.setattr("src.loading.load.apply_montage", lambda recordings, montage: recordings)
    assert load_backgrounds(synthetic_subject(tmp_path)) == {"a": background}


def test_background_eeg_with_nan_stops_loading(tmp_path, monkeypatch):
    values = np.zeros((3, 50))
    values[0, 3] = np.nan
    subject = synthetic_subject(tmp_path)
    monkeypatch.setattr(synthetic, "read_background",
                        lambda subject: [make_recording("background", values)])
    monkeypatch.setattr("src.loading.load.apply_montage", lambda recordings, montage: recordings)
    with pytest.raises(ValueError, match="background"):
        load_backgrounds(subject)


def test_synthetic_group_with_an_unknown_name_is_refused():
    with pytest.raises(ValueError, match="sitting_pre_rest"):
        synthetic.select_pruning_recordings(
            [], {"standing_rest_1": synthetic_background("standing_rest_1")})


SIMULATION = {"source_config": {"trial_source_map": {"right": ["src_a"], "left": ["src_b"]}}}


def test_source_hands_come_from_the_simulations_trial_source_map():
    subject = FakeSubject()
    subject.simulation = SIMULATION
    assert source_hands(subject) == {"src_a": "right_hand", "src_b": "left_hand"}


def test_source_hands_need_a_trial_source_map():
    subject = FakeSubject()
    subject.simulation = {}
    with pytest.raises(ValueError, match="trial_source_map"):
        source_hands(subject)


def test_ground_truth_sources_are_relabelled_by_hand_right_hand_first():
    da = xr.DataArray(np.zeros((3, 2)), dims=("time", "generator"),
                      coords={"generator": ["src_b", "src_a"]})
    tidy = _tidy_sources(da, {"src_a": "right_hand", "src_b": "left_hand"}, "file.nc")
    assert list(tidy.source.values) == ["right_hand", "left_hand"]


def test_ground_truth_with_an_unknown_source_name_is_refused():
    da = xr.DataArray(np.zeros((3, 1)), dims=("time", "source"), coords={"source": ["src_x"]})
    with pytest.raises(ValueError, match="trial_source_map"):
        _tidy_sources(da, {"src_a": "right_hand"}, "file.nc")


# -------------------------------------------------------------------------
# Buccino reader
# -------------------------------------------------------------------------
EEG_RATE, FNIRS_RATE, SESSION_S = 10, 5, 60
# Two blocks, block_2 from 30 s; every trial type occurs twice.
EVENTS = [(0, 10), (1, 1), (9, 7), (15, 1), (23, 8),
          (30, 10), (31, 1), (39, 3), (45, 1), (47, 4), (50, 1), (52, 7), (54, 8),
          (55, 3), (56, 4)]


STANDARD_EEG = EegMontageConfig(standard="standard_1020")
# No optode positions: what a subject looks like before its probe is known.
BUCCINO_MONTAGE = MontageConfig(eeg=STANDARD_EEG)


def cell(*items) -> np.ndarray:
    """A MATLAB cell array holding `items` in order."""
    out = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        out[i] = item
    return out


def write_session(path, events=EVENTS, eeg_time=None) -> SubjectConfig:
    """
    A session file with EEG channels Fc5, C3 (1 µV everywhere) and fNIRS channels
    C3A, C4L, and the subject config of its two blocks.
    """
    eeg_time = np.arange(SESSION_S * EEG_RATE) / EEG_RATE if eeg_time is None else eeg_time
    fnirs_time = np.arange(SESSION_S * FNIRS_RATE) / FNIRS_RATE
    eeg = cell(EEG_RATE, 2, cell("Fc5", "C3"), eeg_time, np.ones((len(eeg_time), 2)))
    intensity = np.full((len(fnirs_time), 2), 0.5)
    fnirs = cell(float(FNIRS_RATE), 2, cell("C3A", "C4L"), fnirs_time, intensity, intensity)
    scipy.io.savemat(path, {"data": cell(eeg, fnirs, np.array(events, dtype=float))})
    return SubjectConfig(subject_id="S0", name="S0", format="buccino",
                         paths={"mat": path}, groups=("block_1", "block_2"),
                         montage=BUCCINO_MONTAGE, test_group="block_2")


def test_buccino_session_is_one_recording_with_its_blocks(tmp_path):
    [session] = buccino.read_recordings(write_session(tmp_path / "S0.mat"))
    session_end = (SESSION_S * EEG_RATE - 1) / EEG_RATE
    assert session.group == "session" and session.run == "S0"
    assert session.blocks == {"block_1": (0.0, 30.0), "block_2": (30.0, session_end)}
    assert session.ground_truth is None


def test_buccino_cues_become_stimuli_lasting_until_the_next_event(tmp_path):
    [session] = buccino.read_recordings(write_session(tmp_path / "S0.mat"))
    stim = session.stim
    assert list(stim.columns) == ["onset", "duration", "value", "trial_type"]
    assert 0.0 not in stim.onset.values and 30.0 not in stim.onset.values
    assert list(stim.trial_type[:4]) == ["rest", "right", "rest", "left"]
    # left ends with the block start at 30 s; the last cue (left_arm, 56 s) is followed
    # by no event and lasts as long as the median left_arm cue (47 s to 50 s).
    assert stim.duration.iloc[3] == pytest.approx(7.0)
    assert stim.duration.iloc[-1] == pytest.approx(3.0)


def test_buccino_eeg_is_in_volts_with_the_files_names_and_no_positions(tmp_path):
    [session] = buccino.read_recordings(write_session(tmp_path / "S0.mat"))
    assert session.eeg.ch_names == ["Fc5", "C3"]
    assert np.allclose(session.eeg.get_data(), 1e-6)
    assert session.eeg.get_montage() is None


def test_buccino_fnirs_has_placeholder_optodes_without_positions(tmp_path):
    [session] = buccino.read_recordings(write_session(tmp_path / "S0.mat"))
    amp = session.fnirs
    assert amp.dims == ("channel", "wavelength", "time")
    assert list(amp.wavelength.values) == list(buccino.WAVELENGTHS_NM)
    assert list(amp.source.values) == ["C3A_S", "C4L_S"]
    assert list(amp.detector.values) == ["C3A_D", "C4L_D"]
    assert session.geo3d.sizes["label"] == 0


def test_buccino_is_refused_until_its_optode_positions_are_set(tmp_path):
    with pytest.raises(ValueError, match="montage.fnirs is not set"):
        check_data_format(write_session(tmp_path / "S0.mat"))


def test_buccino_groups_must_name_blocks_in_the_session(tmp_path):
    subject = write_session(tmp_path / "S0.mat")
    with pytest.raises(ValueError, match="do not name a block"):
        buccino.check(SubjectConfig(subject_id="S0", name="S0", format="buccino",
                                    paths=subject.paths, groups=("run_1",),
                                    montage=BUCCINO_MONTAGE))
    with pytest.raises(ValueError, match="block_3"):
        buccino.read_recordings(SubjectConfig(subject_id="S0", name="S0", format="buccino",
                                              paths=subject.paths,
                                              groups=("block_1", "block_3"),
                                              montage=BUCCINO_MONTAGE))


def test_buccino_rest_is_the_whole_session_without_its_edges(tmp_path):
    session = buccino.read_recordings(write_session(tmp_path / "S0.mat"))
    [reference] = buccino.select_pruning_recordings(session, {})
    fnirs_end = float(session[0].fnirs.time[-1])
    assert reference.blocks is None
    assert list(reference.stim.trial_type) == ["rest"]
    assert reference.stim.onset.iloc[0] == pytest.approx(EDGE_S)
    assert reference.stim.duration.iloc[0] == pytest.approx(fnirs_end - 2 * EDGE_S)


def test_buccino_unknown_event_code_is_refused(tmp_path):
    subject = write_session(tmp_path / "S0.mat", events=[*EVENTS, (58, 5)])
    with pytest.raises(ValueError, match="unknown event codes \\[5\\]"):
        buccino.read_recordings(subject)


def test_buccino_irregular_time_stamps_are_refused(tmp_path):
    eeg_time = np.arange(SESSION_S * EEG_RATE) / EEG_RATE
    eeg_time[10:] += 0.05
    subject = write_session(tmp_path / "S0.mat", eeg_time=eeg_time)
    with pytest.raises(ValueError, match="time stamps deviate"):
        buccino.read_recordings(subject)


# -------------------------------------------------------------------------
# Shin reader
# -------------------------------------------------------------------------
SHIN_EEG_RATE, SHIN_NIRS_RATE, SHIN_SESSION_S = 10, 5, 120
# The NIRS clock runs NIRS_LAG_S ahead of the EEG clock: it starts recording that much
# before the EEG, enough for the minute of rest before the first instruction (28 s), and
# records NIRS_EXTRA_S longer.
NIRS_LAG_S, NIRS_EXTRA_S = 40.0, 55.0
SHIN_ONSETS_MS = [30000, 50000, 70000, 90000]
SHIN_EEG_CODES = [16, 32, 32, 16]
SHIN_GROUPS = ("session1_block1", "session1_block2", "session3_block1")
SHIN_MONTAGE = MontageConfig(
    eeg=STANDARD_EEG,
    fnirs=FnirsMontageConfig(layout=PROJECT_ROOT / "configs/montages/shin2017_fnirs.tsv",
                             standard="standard_1005"))


def write_shin_subject(folder: Path, nirs_jitter_ms=(0, 40, -40, 0),
                       nirs_codes=(1, 2, 2, 1), groups=SHIN_GROUPS) -> SubjectConfig:
    """
    BBCI files of six sessions (motor imagery first, then alternating), all with the
    same 4 trials: EEG channels C3, C4 (1 µV), VEOG, HEOG; NIRS channels C5CP5, C5C3
    at 0.3 V (760 nm) and 0.6 V (850 nm). The NIRS markers lag the EEG's by
    NIRS_LAG_S plus `nirs_jitter_ms`.
    """
    eeg_dir, nirs_dir = folder / "eeg", folder / "nirs"
    eeg_dir.mkdir(parents=True)
    nirs_dir.mkdir(parents=True)
    n_eeg = SHIN_SESSION_S * SHIN_EEG_RATE
    n_nirs = int((SHIN_SESSION_S + NIRS_EXTRA_S) * SHIN_NIRS_RATE)
    titles = ["MI", "MA"] * 3
    eeg_cnt = {"clab": cell("C3", "C4", "VEOG", "HEOG"), "fs": SHIN_EEG_RATE, "yUnit": "uV",
               "x": np.ones((n_eeg, 4))}
    nirs_x = np.hstack([np.full((n_nirs, 2), 0.3), np.full((n_nirs, 2), 0.6)])
    nirs_cnt = {"clab": cell("C5CP5lowWL", "C5C3lowWL", "C5CP5highWL", "C5C3highWL"),
                "fs": SHIN_NIRS_RATE, "yUnit": "V", "x": nirs_x,
                "wavelengths": np.array([760, 850])}
    eeg_mrk = {"time": np.array(SHIN_ONSETS_MS), "event": {"desc": np.array(SHIN_EEG_CODES)}}
    nirs_time = np.array(SHIN_ONSETS_MS) + int(NIRS_LAG_S * 1000) + np.array(nirs_jitter_ms)
    nirs_mrk = {"time": nirs_time, "event": {"desc": np.array(nirs_codes)}}
    sessions = lambda content: cell(*[{**content, "title": title} for title in titles])
    scipy.io.savemat(eeg_dir / "cnt.mat", {"cnt": sessions(eeg_cnt)})
    scipy.io.savemat(eeg_dir / "mrk.mat", {"mrk": cell(*[eeg_mrk] * 6)})
    scipy.io.savemat(nirs_dir / "cnt.mat", {"cnt": sessions(nirs_cnt)})
    scipy.io.savemat(nirs_dir / "mrk.mat", {"mrk": cell(*[nirs_mrk] * 6)})
    scipy.io.savemat(nirs_dir / "mnt.mat", {"mnt": {
        "clab": cell("C5CP5", "C5C3"), "source": {"clab": cell("-", "C5")},
        "detector": {"clab": cell("CP5", "C3")}, "sd": np.array([[2, 1], [2, 2]])}})
    return SubjectConfig(subject_id="VP000", name="VP000", format="shin2017",
                         paths={"eeg": eeg_dir, "nirs": nirs_dir}, groups=groups,
                         montage=SHIN_MONTAGE, test_group=groups[-1])


def test_shin_sessions_are_recordings_with_two_blocks_each(tmp_path):
    session1, session3 = shin2017.read_recordings(write_shin_subject(tmp_path))
    session_end = (SHIN_SESSION_S * SHIN_EEG_RATE - 1) / SHIN_EEG_RATE
    assert (session1.group, session1.run, session3.run) == ("session", "session1", "session3")
    # The second block starts with the instruction of its first trial, 2 s before 70 s.
    assert session1.blocks == {"session1_block1": (0.0, 68.0),
                               "session1_block2": (68.0, session_end)}
    assert list(session3.blocks) == ["session3_block1", "session3_block2"]
    assert session1.ground_truth is None


def test_shin_trials_last_the_task_from_the_eeg_markers(tmp_path):
    [session, _] = shin2017.read_recordings(write_shin_subject(tmp_path))
    assert list(session.stim.columns) == ["onset", "duration", "value", "trial_type"]
    assert list(session.stim.onset) == [30.0, 50.0, 70.0, 90.0]
    assert list(session.stim.trial_type) == ["left", "right", "right", "left"]
    assert (session.stim.duration == shin2017.TASK_S).all()


def test_shin_nirs_is_put_on_the_eeg_clock_by_the_marker_lag(tmp_path):
    [session, _] = shin2017.read_recordings(write_shin_subject(tmp_path))
    time = session.fnirs.time.values
    # NIRS sample i was taken at i / rate on its own clock, NIRS_LAG_S earlier on the
    # EEG clock. Samples before the EEG are kept, those after it cut away.
    assert time[0] == pytest.approx(-NIRS_LAG_S)
    assert time[1] == pytest.approx(1 / SHIN_NIRS_RATE - NIRS_LAG_S)
    assert time[-1] <= session.eeg.times[-1] < time[-1] + 1 / SHIN_NIRS_RATE


def test_shin_nirs_wavelengths_and_optodes_come_from_the_files(tmp_path):
    [session, _] = shin2017.read_recordings(write_shin_subject(tmp_path))
    amp = session.fnirs
    assert amp.dims == ("channel", "wavelength", "time")
    assert list(amp.channel.values) == ["C5CP5", "C5C3"]
    assert list(amp.source.values) == ["C5", "C5"] and list(amp.detector.values) == ["CP5", "C3"]
    assert np.allclose(amp.sel(wavelength=760.0), 0.3)
    assert np.allclose(amp.sel(wavelength=850.0), 0.6)
    assert session.geo3d.sizes["label"] == 0


def test_shin_eeg_is_in_volts_with_its_eog_channels_typed(tmp_path):
    [session, _] = shin2017.read_recordings(write_shin_subject(tmp_path))
    assert session.eeg.ch_names == ["C3", "C4", "VEOG", "HEOG"]
    assert session.eeg.get_channel_types() == ["eeg", "eeg", "eog", "eog"]
    assert np.allclose(session.eeg.get_data(), 1e-6)


def test_shin_markers_that_disagree_between_the_devices_are_refused(tmp_path):
    with pytest.raises(ValueError, match="cannot be aligned"):
        shin2017.read_recordings(write_shin_subject(tmp_path / "late",
                                                    nirs_jitter_ms=(0, 0, 500, 0)))
    with pytest.raises(ValueError, match="disagree on the trials"):
        shin2017.read_recordings(write_shin_subject(tmp_path / "other_hand",
                                                    nirs_codes=(1, 2, 1, 1)))


def test_shin_groups_must_name_blocks_of_motor_imagery_sessions(tmp_path):
    subject = write_shin_subject(tmp_path)
    for groups in [("session2_block1",), ("session1_block3",), ("block_1",)]:
        with pytest.raises(ValueError, match="do not name a block"):
            shin2017.check(dataclasses.replace(subject, groups=groups, test_group=None))


def test_shin_rest_is_the_minute_before_the_first_instruction_of_every_session(tmp_path):
    sessions = shin2017.read_recordings(write_shin_subject(tmp_path))
    references = shin2017.select_pruning_recordings(sessions, {})
    assert [reference.run for reference in references] == ["session1", "session3"]
    # The first instruction is 2 s before the first task, at 30 s.
    for reference in references:
        assert reference.blocks is None
        assert list(reference.stim.trial_type) == ["rest"]
        assert reference.stim.onset.iloc[0] == pytest.approx(28.0 - 60.0 + EDGE_S)
        assert reference.stim.duration.iloc[0] == pytest.approx(60.0 - 2 * EDGE_S)


def test_shin_loads_with_the_datasets_montage(tmp_path):
    subject = write_shin_subject(tmp_path)
    check_data_format(subject)
    session, _ = load_recordings(subject)
    assert list(session.fnirs.source.values) == ["C5", "C5"]
    labels = set(session.geo3d.label.values)
    assert {"C5", "CP5", "C3"} <= labels
    assert session.eeg.get_montage() is not None


# -------------------------------------------------------------------------
# Montage
# -------------------------------------------------------------------------
def write_optodes(path) -> dict[str, np.ndarray]:
    """An optode file placing the Buccino session's placeholder optodes; its positions."""
    positions = {"C3A_S": [-40.0, 10.0, 70.0], "C3A_D": [-40.0, -20.0, 70.0],
                 "C4L_S": [55.0, -5.0, 60.0], "C4L_D": [70.0, -5.0, 45.0],
                 "Nz": [0.0, 90.0, 0.0]}
    kinds = {"S": "SOURCE", "D": "DETECTOR"}
    rows = [f"{label}\t" + "\t".join(f"{v:f}" for v in xyz) +
            f"\tPointType.{kinds.get(label[-1], 'LANDMARK')}" for label, xyz in positions.items()]
    path.write_text("labels\tX\tY\tZ\tPointType\tcrs=digitized\tunits=mm\n"
                    + "\n".join(rows) + "\n")
    return {label: np.array(xyz) for label, xyz in positions.items()}


def test_standard_eeg_montage_respells_and_places_the_channels(tmp_path):
    [session] = buccino.read_recordings(write_session(tmp_path / "S0.mat"))
    optodes = tmp_path / "optodes.tsv"
    write_optodes(optodes)
    montage = MontageConfig(eeg=STANDARD_EEG, fnirs=FnirsMontageConfig(file=optodes))
    [placed] = apply_montage([session], montage)
    assert placed.eeg.ch_names == ["FC5", "C3"]
    assert np.isfinite(placed.eeg.get_montage().get_positions()["ch_pos"]["FC5"]).all()
    assert session.eeg.ch_names == ["Fc5", "C3"]         # the reader's recording is untouched


def test_eeg_montage_file_places_the_channels_where_it_says(tmp_path):
    [session] = buccino.read_recordings(write_session(tmp_path / "S0.mat"))
    cap = tmp_path / "cap.sfp"
    cap.write_text("FC5 -0.07 0.02 0.05\nC3 -0.06 -0.01 0.07\n")
    optodes = tmp_path / "optodes.tsv"
    write_optodes(optodes)
    montage = MontageConfig(eeg=EegMontageConfig(file=cap), fnirs=FnirsMontageConfig(file=optodes))
    [placed] = apply_montage([session], montage)
    ch_pos = placed.eeg.get_montage().get_positions()["ch_pos"]
    expected = mne.channels.read_custom_montage(cap).get_positions()["ch_pos"]
    assert np.allclose(ch_pos["C3"], expected["C3"])


def test_optode_file_replaces_the_recordings_positions(tmp_path):
    [session] = buccino.read_recordings(write_session(tmp_path / "S0.mat"))
    optodes = tmp_path / "optodes.tsv"
    positions = write_optodes(optodes)
    montage = MontageConfig(eeg=STANDARD_EEG, fnirs=FnirsMontageConfig(file=optodes))
    [placed] = apply_montage([session], montage)
    geo3d = placed.geo3d.pint.to("mm").pint.dequantify()
    assert np.allclose(geo3d.sel(label="C4L_D").values, positions["C4L_D"])


def test_optodes_without_a_position_are_refused(tmp_path):
    [session] = buccino.read_recordings(write_session(tmp_path / "S0.mat"))
    montage = MontageConfig(eeg=STANDARD_EEG, fnirs=FnirsMontageConfig(recording=True))
    with pytest.raises(ValueError, match="C3A_D.*no position"):
        apply_montage([session], montage)


def test_eeg_channel_missing_from_the_montage_is_refused(tmp_path):
    [session] = buccino.read_recordings(write_session(tmp_path / "S0.mat"))
    cap = tmp_path / "cap.sfp"
    cap.write_text("C3 -0.06 -0.01 0.07\nCz 0.0 0.0 0.09\n")
    optodes = tmp_path / "optodes.tsv"
    write_optodes(optodes)
    montage = MontageConfig(eeg=EegMontageConfig(file=cap), fnirs=FnirsMontageConfig(file=optodes))
    with pytest.raises(ValueError, match="'Fc5'.*not in the montage"):
        apply_montage([session], montage)


def test_check_montage_refuses_an_unknown_template_and_absent_files(tmp_path):
    subject = write_session(tmp_path / "S0.mat")
    fnirs = FnirsMontageConfig(recording=True)
    unknown = MontageConfig(eeg=EegMontageConfig(standard="standard_9999"), fnirs=fnirs)
    with pytest.raises(ValueError, match="standard_9999"):
        check_montage(dataclasses.replace(subject, montage=unknown))
    absent = MontageConfig(eeg=EegMontageConfig(file=tmp_path / "cap.elc"), fnirs=fnirs)
    with pytest.raises(FileNotFoundError, match="cap.elc"):
        check_montage(dataclasses.replace(subject, montage=absent))


@pytest.mark.parametrize("make", [
    lambda: EegMontageConfig(),
    lambda: EegMontageConfig(standard="standard_1020", file=Path("cap.elc")),
    lambda: FnirsMontageConfig(),
    lambda: FnirsMontageConfig(recording=True, file=Path("optodes.tsv")),
    lambda: FnirsMontageConfig(file=Path("optodes.tsv"), layout=Path("layout.tsv"),
                               standard="standard_1005"),
])
def test_a_montage_must_name_exactly_one_source(make):
    with pytest.raises(ValueError, match="exactly one"):
        make()


@pytest.mark.parametrize("make", [
    lambda: FnirsMontageConfig(layout=Path("layout.tsv")),
    lambda: FnirsMontageConfig(recording=True, standard="standard_1005"),
])
def test_an_fnirs_layout_and_its_template_go_together(make):
    with pytest.raises(ValueError, match="go together"):
        make()


LAYOUT_MONTAGE = dict(eeg=EegMontageConfig(standard="standard_1005"))


def write_layout(path, rows=(("C3A", "FCC5h", "FCC3h"), ("C4L", "FCC6h", "CCP6h"))) -> Path:
    """A layout file for the Buccino session's two channels, with a comment line."""
    lines = ["# reconstructed from a figure", "channel\tsource\tdetector",
             *("\t".join(row) for row in rows)]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_fnirs_layout_wires_the_channels_to_their_optode_sites(tmp_path):
    [session] = buccino.read_recordings(write_session(tmp_path / "S0.mat"))
    layout = write_layout(tmp_path / "layout.tsv")
    montage = MontageConfig(**LAYOUT_MONTAGE,
                            fnirs=FnirsMontageConfig(layout=layout, standard="standard_1005"))
    [placed] = apply_montage([session], montage)
    assert list(placed.fnirs.source.values) == ["FCC5h", "FCC6h"]
    assert list(placed.fnirs.detector.values) == ["FCC3h", "CCP6h"]
    types = dict(zip(placed.geo3d.label.values, placed.geo3d.type.values))
    assert types["FCC5h"] == PointType.SOURCE and types["CCP6h"] == PointType.DETECTOR
    assert {"Nz", "LPA", "RPA"} <= set(types)


def test_fnirs_layout_sites_share_the_frame_of_the_eeg_electrodes(tmp_path):
    [session] = buccino.read_recordings(write_session(tmp_path / "S0.mat"))
    layout = write_layout(tmp_path / "layout.tsv")
    montage = MontageConfig(**LAYOUT_MONTAGE,
                            fnirs=FnirsMontageConfig(layout=layout, standard="standard_1005"))
    [placed] = apply_montage([session], montage)
    # The same template, placed as an EEG electrode, lands on the same point.
    info = mne.create_info(["FCC5h"], sfreq=1.0, ch_types="eeg")
    info.set_montage("standard_1005")
    electrode_mm = info.get_montage().get_positions()["ch_pos"]["FCC5h"] * 1000.0
    optode_mm = placed.geo3d.pint.to("mm").pint.dequantify().sel(label="FCC5h").values
    assert np.allclose(optode_mm, electrode_mm)


def test_fnirs_channel_missing_from_the_layout_is_refused(tmp_path):
    [session] = buccino.read_recordings(write_session(tmp_path / "S0.mat"))
    layout = write_layout(tmp_path / "layout.tsv", rows=[("C3A", "FCC5h", "FCC3h")])
    montage = MontageConfig(**LAYOUT_MONTAGE,
                            fnirs=FnirsMontageConfig(layout=layout, standard="standard_1005"))
    with pytest.raises(ValueError, match="'C4L'.*not in the fNIRS layout"):
        apply_montage([session], montage)


def test_fnirs_layout_site_missing_from_the_template_is_refused(tmp_path):
    [session] = buccino.read_recordings(write_session(tmp_path / "S0.mat"))
    layout = write_layout(tmp_path / "layout.tsv")
    montage = MontageConfig(**LAYOUT_MONTAGE,
                            fnirs=FnirsMontageConfig(layout=layout, standard="standard_1020"))
    with pytest.raises(ValueError, match="not in the standard_1020 template"):
        apply_montage([session], montage)


def test_fnirs_layout_with_a_site_used_as_source_and_detector_is_refused(tmp_path):
    layout = write_layout(tmp_path / "layout.tsv",
                          rows=[("C3A", "FCC5h", "FCC3h"), ("C4L", "FCC3h", "CCP6h")])
    with pytest.raises(ValueError, match="FCC3h.*both a source and a detector"):
        read_layout(layout)


def test_buccino_layout_wires_every_channel_of_the_dataset():
    layout = read_layout(PROJECT_ROOT / "configs" / "montages" / "buccino_fnirs.tsv")
    sources = {source for source, _ in layout.values()}
    detectors = {detector for _, detector in layout.values()}
    assert len(layout) == 34 and len(sources) == 12 and len(detectors) == 12
    template = set(mne.channels.make_standard_montage("standard_1005").ch_names)
    assert sources | detectors <= template
    assert layout["C3A"] == ("FCC5h", "FCC3h")
