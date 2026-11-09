"""
Tests for the results module: a run written with save_run reads back with load_run
into the same RunResult, with models that still work.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pint_xarray  # noqa: F401  (registers .pint)
import pytest
import xarray as xr
import yaml
from cedalion.dataclasses import PointType

from src.config import load_experiment, load_subject, subject_snapshot
from src.results import load_inputs, load_run, save_run
from src.results.load import _fnirs_optodes
from src.results.tables import score_columns
from src.types import BadChannels, Geometry, RunResult
from src.unmixing.concat import concat_runs
from src.unmixing.fit import transform
from src.unmixing.run import unmix
from test_unmixing import (CONFIGS, HANDS, NO_TRUTH, ground_truth, runs_without_truth,
                           synthetic_runs)


@pytest.fixture(scope="module")
def result() -> RunResult:
    cfg = load_experiment(CONFIGS / "experiment.yaml")
    unmixing_cfg = dataclasses.replace(
        cfg.unmixing, methods=("CCA", "tCCA"), metrics=("joint_avg", "isi_avg"), nested=True,
        n_jobs=1, grid_overrides={"tCCA": {"time_shifts": [[0, 1], [0, 2]]}})
    cfg = dataclasses.replace(cfg, unmixing=unmixing_cfg)
    runs = synthetic_runs()
    labels = ["S1", "D1", "Cz"]
    geo3d = xr.DataArray(np.arange(9.0).reshape(3, 3), dims=("label", "pos"),
                         coords={"label": labels, "type": ("label", [PointType.SOURCE,
                                 PointType.DETECTOR, PointType.LANDMARK])}).pint.quantify("mm")
    eeg_labels = [*(f"e{i}" for i in range(6)), "Nz"]
    eeg_geo3d = xr.DataArray(
        np.arange(21.0).reshape(7, 3), dims=("label", "pos"),
        coords={"label": eeg_labels,
                "type": ("label", [PointType.ELECTRODE] * 6 + [PointType.LANDMARK])}
    ).pint.quantify("mm")
    gt = ground_truth()
    return RunResult(
        cfg=cfg, subject=load_subject(CONFIGS / "subject.yaml"),
        bad_channels=BadChannels(eeg=("e1",), fnirs=("f9",),
                                 eeg_detail={"pass1": {"bad_by_flat": ("e1",)}},
                                 eeg_by_recording={"background_1": ("e1",)},
                                 fnirs_detail={"snr": ("f9",)}),
        alignment=pd.DataFrame([{"group": r.group, "run": r.run, "eeg_samples": 100} for r in runs]),
        geometry=Geometry(eeg_channels=tuple(f"e{i}" for i in range(6)),
                          fnirs_channels=tuple(f"f{i}" for i in range(5)), geo3d=geo3d,
                          eeg_geo3d=eeg_geo3d,
                          fnirs_optodes={f"f{i}": ("S1", "D1") for i in (0, 1, 2, 3, 4, 9)},
                          A_eeg=gt.A_eeg, A_fnirs=gt.A_fnirs),
        unmixing=unmix(runs, unmixing_cfg, "b", HANDS), inputs=runs,
        started="2026-10-21T17:00:00", finished="2026-10-21T17:01:00")


@pytest.fixture(scope="module")
def run_dir(result, tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("results")
    run_dir = save_run(result, base_dir, tmp_path_factory.mktemp("preprocessed"))
    assert run_dir.parts[-4:] == ("runs", "test_experiment", "subject", "2026-10-21T17-00-00")
    index = pd.read_csv(base_dir / "runs" / "index.csv")
    assert len(index) == 1
    assert base_dir / "runs" / index["folder"][0] == run_dir
    assert index["dataset"][0] == "subject"
    return run_dir


@pytest.fixture(scope="module")
def loaded(run_dir):
    return load_run(run_dir)


def test_inputs_go_to_their_own_folder_and_read_back(result, run_dir):
    info = json.loads((run_dir / "results" / "run.json").read_text())
    folder = Path(info["inputs"])
    assert folder.parts[-2:] == ("subject", "2026-10-21T17-00-00")
    assert not folder.is_relative_to(run_dir.parents[3])         # not inside results/
    assert json.loads((folder / "preprocessed.json").read_text())["run_folder"] == str(run_dir)

    inputs = load_inputs(run_dir)
    assert [(r.group, r.run) for r in inputs] == [(r.group, r.run) for r in result.inputs]
    for a, b in zip(inputs, result.inputs):
        for name in ("eeg", "fnirs", "s_eeg", "s_fnirs"):
            np.testing.assert_allclose(getattr(a, name).values, getattr(b, name).values,
                                       rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(getattr(a, name).time.values,
                                          getattr(b, name).time.values)
        pd.testing.assert_frame_equal(a.stim, b.stim, check_dtype=False)
        assert a.window == b.window
        xr.testing.assert_allclose(a.eeg_std, b.eeg_std)
        xr.testing.assert_allclose(a.fnirs_std, b.fnirs_std)


def test_a_run_without_inputs_says_so(result, tmp_path):
    no_save = dataclasses.replace(result.cfg, save_inputs=False)
    run_dir = save_run(dataclasses.replace(result, cfg=no_save), tmp_path)
    assert json.loads((run_dir / "results" / "run.json").read_text())["inputs"] is None
    with pytest.raises(FileNotFoundError, match="saved no inputs"):
        load_inputs(run_dir)


def test_subject_snapshot_keeps_project_paths_relative(result):
    paths = subject_snapshot(result.subject)["paths"]
    assert paths["data"] == "data/test_fixture/test_subject"
    assert not any(Path(p).is_absolute() for p in paths.values())


def test_configs_and_metadata_round_trip(result, loaded):
    assert loaded.cfg == result.cfg
    assert loaded.subject == result.subject
    assert loaded.bad_channels == result.bad_channels
    pd.testing.assert_frame_equal(loaded.alignment, result.alignment, check_dtype=False)
    assert (loaded.started, loaded.finished) == (result.started, result.finished)


def test_geometry_round_trips(result, loaded):
    g, h = result.geometry, loaded.geometry
    assert (h.eeg_channels, h.fnirs_channels) == (g.eeg_channels, g.fnirs_channels)
    np.testing.assert_array_equal(h.geo3d.pint.dequantify().values, g.geo3d.pint.dequantify().values)
    assert str(h.geo3d.pint.units) == str(g.geo3d.pint.units)
    assert list(h.geo3d.type.values) == list(g.geo3d.type.values)
    xr.testing.assert_equal(h.eeg_geo3d.pint.dequantify(), g.eeg_geo3d.pint.dequantify())
    assert str(h.eeg_geo3d.pint.units) == str(g.eeg_geo3d.pint.units)
    assert h.fnirs_optodes == g.fnirs_optodes
    xr.testing.assert_equal(h.A_eeg, g.A_eeg)
    xr.testing.assert_equal(h.A_fnirs, g.A_fnirs)


def selection_key(s):
    return (s.test_group, s.method, s.metric)


def test_selections_scores_and_recovery_round_trip(result, loaded):
    for name, before, after in [("non_nested", result.unmixing.selections, loaded.unmixing.selections),
                                *[(g, result.unmixing.nested[g], loaded.unmixing.nested[g])
                                  for g in result.unmixing.nested]]:
        before, after = sorted(before, key=selection_key), sorted(after, key=selection_key)
        assert [selection_key(s) for s in after] == [selection_key(s) for s in before], name
        for b, a in zip(before, after):
            assert a.scores == b.scores
            assert a.inner_score == pytest.approx(b.inner_score)
            assert str(a.params) == str(b.params)
            xr.testing.assert_allclose(a.recovery.u, b.recovery.u)
            xr.testing.assert_allclose(a.recovery.s_eeg, b.recovery.s_eeg)
            xr.testing.assert_allclose(a.recovery.r_train_fnirs, b.recovery.r_train_fnirs)
            xr.testing.assert_allclose(a.recovery.pattern_eeg, b.recovery.pattern_eeg)
            xr.testing.assert_allclose(a.recovery.pattern_fnirs, b.recovery.pattern_fnirs)
            assert a.recovery.gain_eeg is not None
            xr.testing.assert_allclose(a.recovery.gain_eeg, b.recovery.gain_eeg)
            xr.testing.assert_allclose(a.recovery.gain_fnirs, b.recovery.gain_fnirs)
            assert a.recovery.run_starts == b.recovery.run_starts
            assert a.recovery.runs == b.recovery.runs
            assert len(a.recovery.stim) > 0
            pd.testing.assert_frame_equal(a.recovery.stim, b.recovery.stim, check_dtype=False)
            assert [f.val_group for f in a.folds] == [f.val_group for f in b.folds]
            for fa, fb in zip(a.folds, b.folds):
                assert fa.scores == fb.scores
                xr.testing.assert_allclose(fa.recovery.v, fb.recovery.v)
    assert all(len(s.folds) == 2 for s in loaded.unmixing.selections)


def test_a_run_without_ground_truth_round_trips(result, tmp_path):
    runs = runs_without_truth()
    geometry = dataclasses.replace(result.geometry, A_eeg=None, A_fnirs=None)
    real = dataclasses.replace(
        result, cfg=dataclasses.replace(result.cfg, unmixing=NO_TRUTH, save_inputs=False),
        geometry=geometry, unmixing=unmix(runs, NO_TRUTH, "b", HANDS), inputs=runs)
    loaded = load_run(save_run(real, tmp_path))

    before, = real.unmixing.selections
    after, = loaded.unmixing.selections
    np.testing.assert_equal(score_columns(after.scores), score_columns(before.scores))
    assert after.scores.best_pairs == {}
    for recovery in (after.recovery, *(f.recovery for f in after.folds)):
        assert recovery.s_eeg is None and recovery.r_train_eeg is None
    xr.testing.assert_allclose(after.recovery.pattern_eeg, before.recovery.pattern_eeg)
    assert loaded.geometry.A_eeg is None


def test_loaded_models_give_the_same_components(result, loaded):
    test_runs = concat_runs([r for r in synthetic_runs() if r.group == "b"])
    for b, a in zip(sorted(result.unmixing.selections, key=selection_key),
                    sorted(loaded.unmixing.selections, key=selection_key)):
        for x, y in zip(transform(b.fitted, test_runs), transform(a.fitted, test_runs)):
            np.testing.assert_allclose(y.values, x.values, equal_nan=True)


def test_landscape_round_trips(result, loaded):
    key = lambda p: (p.test_group, p.method, p.val_group, str(p.params))
    before = sorted(result.unmixing.landscape, key=key)
    after = sorted(loaded.unmixing.landscape, key=key)
    assert [key(p) for p in after] == [key(p) for p in before]
    for a, b in zip(after, before):
        # Unselected combinations have NaN pattern_r, so compare NaN as equal.
        np.testing.assert_equal(score_columns(a.scores), score_columns(b.scores))
        assert a.error == b.error
        np.testing.assert_array_equal(list(a.metrics.values()), list(b.metrics.values()))
    key = lambda c: (c.test_group, c.method, str(c.params))
    before = sorted(result.unmixing.combinations, key=key)
    after = sorted(loaded.unmixing.combinations, key=key)
    assert [key(c) for c in after] == [key(c) for c in before]
    for a, b in zip(after, before):
        np.testing.assert_array_equal(list(a.means.values()), list(b.means.values()))


def test_save_run_never_overwrites(result, tmp_path):
    no_inputs = dataclasses.replace(result, cfg=dataclasses.replace(result.cfg, save_inputs=False))
    save_run(no_inputs, tmp_path)
    with pytest.raises(FileExistsError):
        save_run(no_inputs, tmp_path)


def test_old_run_without_montage_falls_back_to_the_template_with_a_warning(result, tmp_path):
    no_inputs = dataclasses.replace(result, cfg=dataclasses.replace(result.cfg, save_inputs=False))
    data_dir = save_run(no_inputs, tmp_path) / "results"
    # Strip what runs saved before the montage was part of the subject did not have.
    geometry = xr.load_dataset(data_dir / "geometry.nc", engine="scipy")
    geometry = geometry.drop_vars(["eeg_geo3d", "label_eeg", "type_eeg"])
    del geometry.attrs["eeg_geo3d_units"]
    geometry.to_netcdf(data_dir / "geometry.nc", engine="scipy")
    subject = yaml.safe_load((data_dir / "subject.yaml").read_text())
    del subject["montage"]
    (data_dir / "subject.yaml").write_text(yaml.safe_dump(subject))

    with pytest.warns(UserWarning, match="standard_1020") as caught:
        old = load_run(data_dir.parent)
    assert len(caught) == 2
    assert old.subject.montage.eeg.standard == "standard_1020"
    assert old.subject.montage.fnirs.recording
    assert "Cz" in old.geometry.eeg_geo3d.label.values
    assert {"Nz", "LPA", "RPA"} <= set(old.geometry.eeg_geo3d.label.values)


def test_old_geometry_reads_the_fnirs_optodes_from_the_channel_names():
    geometry = xr.Dataset(coords={"fnirs_channel": ["S1D1", "S10D12"]})
    with pytest.warns(UserWarning, match="S<n>D<m>"):
        optodes = _fnirs_optodes(geometry, dropped_fnirs=("S2D3",))
    assert optodes == {"S1D1": ("S1", "D1"), "S10D12": ("S10", "D12"), "S2D3": ("S2", "D3")}
