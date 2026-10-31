"""
Tests for loading: the stop on non-finite raw EEG, the report of what was loaded, the
ground-truth sources labelled by hand, the Buccino reader on a small MAT file laid
out like the dataset's, and the montage applied from the subject config.
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
from src.loading import buccino, check_data_format
from src.loading.load import _check_eeg_finite, loading_report
from src.loading.montage import apply_montage, check_montage, read_layout
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
    subject.paths = {"data": "data/de000", "rest_eeg": "rest-raw.fif",
                     "rest_fnirs": "rest.snirf"}
    recordings = [make_recording("run_001", np.zeros((3, 50))),
                  make_recording("run_002", np.zeros((3, 50)))]
    assert loading_report(subject, recordings) == (
        "Loaded 2 runs\n"
        "  g1: run_001, run_002\n"
        "  data: data/de000\n"
        "  rest_eeg: rest-raw.fif\n"
        "  rest_fnirs: rest.snirf")


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


def test_buccino_rest_is_the_whole_session_as_one_rest_period(tmp_path):
    rest = buccino.read_rest(write_session(tmp_path / "S0.mat"))
    assert rest.group == "rest" and rest.blocks is None
    assert len(rest.stim) == 1 and rest.stim.trial_type.iloc[0] == "rest"
    assert rest.stim.onset.iloc[0] == 0.0
    assert rest.stim.duration.iloc[0] == pytest.approx(rest.eeg.times[-1])


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
