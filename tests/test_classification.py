"""
Tests for the classification module.
"""
import dataclasses
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint_xarray  # noqa: F401  (registers .pint)
import pytest
import xarray as xr

from src.classification import classify
from src.classification.features import (epoch_components, epoch_span, interval_means,
                                         rest_trials, run_trials, subtract_baseline)
from src.config import ClassificationConfig, load_subject
from src.results import load_run, save_run
from src.types import BadChannels, Geometry, PreprocessedRecording, RunResult
from src.unmixing.run import unmix
from src.visualization import plot_classification
from test_unmixing import CFG, CONFIGS, HANDS, RATE, ground_truth, mixing

CLS = ClassificationConfig()


# -------------------------------------------------------------------------
# Config, epochs and trials
# -------------------------------------------------------------------------
def test_config_needs_two_distinct_classes_and_known_feature_sets():
    with pytest.raises(ValueError, match="at least two classes"):
        ClassificationConfig(classes=("left",))
    with pytest.raises(ValueError, match="twice"):
        ClassificationConfig(classes=("left", "left"))
    with pytest.raises(ValueError, match="feature_sets must be one of"):
        ClassificationConfig(feature_sets=("eeg", "hbr"))


def test_epoch_span_covers_the_baseline_and_every_interval():
    assert epoch_span(CLS) == (-2.0, 12.0)
    after = ClassificationConfig(baseline_window=(0.0, 0.5), eeg_intervals=((1.0, 2.0),),
                                 fnirs_intervals=((1.0, 3.0),))
    assert epoch_span(after) == (0.0, 3.0)


def epochs_of(values: np.ndarray, reltime: np.ndarray) -> xr.DataArray:
    """Epochs (epoch, reltime, component) with one component."""
    return xr.DataArray(values[:, :, None], dims=("epoch", "reltime", "component"),
                        coords={"reltime": reltime})


def test_baseline_and_interval_means_of_a_step():
    reltime = np.arange(-2.0, 6.01, 0.5)
    step = np.where(reltime >= 1.0, 3.0, 0.0) + 5.0             # offset 5, step of 3 at 1 s
    epochs = subtract_baseline(epochs_of(np.stack([step, 2 * step]), reltime), (-2.0, 0.0))
    features = interval_means(epochs, ((0.0, 0.5), (2.0, 4.0)))
    np.testing.assert_allclose(features, [[0.0, 3.0], [0.0, 6.0]])


def test_an_interval_is_averaged_over_its_finite_part_and_nan_only_when_empty():
    reltime = np.arange(-2.0, 6.01, 0.5)
    values = np.where(reltime < 3.0, 1.0, 3.0)[None, :]
    values[0, reltime >= 3.5] = np.nan                           # the run ends at 3 s
    features = interval_means(epochs_of(values, reltime), ((2.0, 4.0), (4.0, 6.0)))
    np.testing.assert_allclose(features[0, 0], (1 + 1 + 3) / 3)  # 2, 2.5 and 3 s
    assert np.isnan(features[0, 1])


def test_an_epoch_past_the_end_of_the_run_is_cut_not_dropped():
    time = np.arange(0.0, 30.0, 0.1)
    comps = xr.DataArray(np.ones((len(time), 1)), dims=("time", "component"),
                         coords={"time": time})
    trials = pd.DataFrame({"onset": [5.0, 25.0], "duration": 4.0, "trial_type": ["left", "right"]})
    epochs = epoch_components(comps, trials, (-2.0, 12.0))
    assert list(epochs.trial_type.values) == ["left", "right"]
    late = epochs.isel(epoch=1, component=0)
    assert np.isfinite(late.sel(reltime=slice(-2.0, 4.8))).all()
    assert np.isnan(late.sel(reltime=slice(5.0, 12.0))).all()


def test_rest_trials_fill_the_stretches_no_epoch_covers():
    # Epochs cover [8, 22] and [38, 52]. Only 22 to 38 (16 s) holds a 14 s epoch,
    # centred: it starts at 23, so its onset is 25.
    stim = pd.DataFrame({"onset": [10.0, 40.0], "duration": 4.0, "trial_type": ["left", "right"]})
    rest = rest_trials(stim, (-2.0, 12.0), 0.0, 60.0, "rest")
    np.testing.assert_allclose(rest.onset, [25.0])
    assert (rest.trial_type == "rest").all()
    # A long stretch holds several, back to back: 22 to 60 holds two, starting at
    # 27 and 41, so their onsets are 29 and 43.
    rest = rest_trials(stim.iloc[:1], (-2.0, 12.0), 0.0, 60.0, "rest")
    np.testing.assert_allclose(rest.onset, [29.0, 43.0])


def test_run_trials_take_rest_from_the_stim_table_when_it_has_rest():
    cfg = ClassificationConfig(classes=("left", "right", "rest"))
    stim = pd.DataFrame({"onset": [10.0, 40.0, 70.0], "duration": 4.0,
                         "trial_type": ["left", "rest", "other"]})
    trials = run_trials(stim, cfg, 0.0, 200.0)
    assert list(trials.trial_type) == ["left", "rest"]
    cut = run_trials(stim[stim.trial_type != "rest"], cfg, 0.0, 200.0)
    assert "rest" in set(cut.trial_type) and "other" not in set(cut.trial_type)


def test_run_trials_merge_the_trial_types_of_a_side():
    stim = pd.DataFrame({"onset": [10.0, 30.0, 50.0, 70.0, 90.0], "duration": 4.0,
                         "trial_type": ["left", "right_arm", "rest", "left_arm", "right"]})
    trial_classes = {"right": ("right", "right_arm"), "left": ("left", "left_arm")}
    trials = run_trials(stim, CLS, 0.0, 200.0, trial_classes)
    assert list(trials.trial_type) == ["left", "right", "left", "right"]
    assert list(run_trials(stim, CLS, 0.0, 200.0).trial_type) == ["left", "right"]


# -------------------------------------------------------------------------
# Classification of runs whose sources respond to the trials
# -------------------------------------------------------------------------
def event_runs(groups=("a", "b", "c"), n_trials: int = 12, spacing: float = 16.0,
               runs_per_group: int = 2) -> list[PreprocessedRecording]:
    """
    Runs with left and right trials, `spacing` s apart. The right-hand source rises
    for 6 s after each right onset, the left-hand source after each left onset;
    both modalities see the same courses, mixed into their channels with noise.
    """
    gt = ground_truth()
    duration = spacing * (n_trials + 1)
    time = np.arange(int(duration * RATE)) / RATE
    out = []
    for g, group in enumerate(groups):
        for r in range(runs_per_group):
            rng = np.random.default_rng(100 * g + r)
            onsets = spacing * (1 + np.arange(n_trials))
            types = rng.permutation(["left", "right"] * (n_trials // 2))
            s = 0.3 * rng.standard_normal((len(time), 2))
            for onset, kind in zip(onsets, types):
                s[(time >= onset) & (time < onset + 6.0), 0 if kind == "right" else 1] += 1.0
            own = 0.3 * rng.standard_normal(len(time))
            s_fnirs = np.stack([s[:, 0], 0.7 * s[:, 1] + 0.7 * own], axis=1)
            eeg = s @ (mixing(6, 2) ** 2).T + 0.1 * rng.standard_normal((len(time), 6))
            fnirs = s_fnirs @ mixing(5, 3).T + 0.1 * rng.standard_normal((len(time), 5))

            def da(values, labels, dim):
                return xr.DataArray(values, dims=("time", dim), coords={"time": time, dim: labels})

            sources = ["right_hand", "left_hand"]
            out.append(PreprocessedRecording(
                group=group, run=f"run_00{r + 1}",
                eeg=da(eeg, [f"e{i}" for i in range(6)], "channel"),
                fnirs=da(fnirs, [f"f{i}" for i in range(5)], "channel"),
                s_eeg=da(s, sources, "source"), s_fnirs=da(s_fnirs, sources, "source"),
                geo3d=xr.DataArray(), eeg_geo3d=xr.DataArray(), fnirs_optodes={},
                stim=pd.DataFrame({"onset": onsets, "duration": 4.0, "value": 1.0,
                                   "trial_type": types}),
                window=(0.0, duration), ground_truth=gt))
    return out


CCA_ONLY = dataclasses.replace(CFG.unmixing, methods=("CCA",), metrics=("joint_avg", "isi_avg"),
                               n_jobs=1)
FAST = ClassificationConfig(eeg_intervals=((0.5, 4.5),), fnirs_intervals=((2.0, 5.0), (5.0, 8.0)))


def test_classify_tells_left_from_right_on_the_held_out_group():
    runs = event_runs()
    unmixing = unmix(runs, CCA_ONLY, "b", HANDS)
    result = classify(runs, unmixing, FAST)

    assert result.test_group == "b" and result.nested is None
    assert [(s.metric, s.feature_set) for s in result.scores] == [
        (m, f) for m in ("joint_avg", "isi_avg") for f in ("eeg_fnirs", "eeg", "fnirs")]
    for s in result.scores:
        assert s.test_group == "b" and s.classes == ("left", "right")
        assert s.n_train == 4 * 12 and s.n_test == 2 * 12
        assert s.dropped_train == s.dropped_test == 0
        assert s.accuracy > 0.9
        assert set(s.recall) == {"left", "right"}


def test_classify_with_rest_cut_from_the_gaps_and_nested_groups():
    runs = event_runs(spacing=30.0, n_trials=8)
    unmixing = unmix(runs, dataclasses.replace(CCA_ONLY, metrics=("joint_avg",), nested=True),
                     "b", HANDS)
    cfg = dataclasses.replace(FAST, classes=("left", "right", "rest"), feature_sets=("eeg_fnirs",))
    result = classify(runs, unmixing, cfg)

    assert set(result.nested) == {"a", "b", "c"}
    assert result.nested["b"] is result.scores
    for scores in result.nested.values():
        (s,) = scores
        # Epochs are 10 s (-2 to 8 s), so the 20 s free before the first trial, between
        # trials and after the last one hold two each: 18 per run, 2 test runs.
        assert sum(s.confusion["rest"].values()) == 2 * 18
        assert s.accuracy > 0.8
        assert list(s.confusion) == ["left", "right", "rest"]


def test_classify_stops_when_no_gap_holds_a_rest_epoch():
    # 9 s between onsets, and 10 s epochs.
    runs = event_runs(spacing=9.0)
    unmixing = unmix(runs, dataclasses.replace(CCA_ONLY, metrics=("joint_avg",)), "b", HANDS)
    cfg = dataclasses.replace(FAST, classes=("left", "right", "rest"))
    with pytest.raises(ValueError, match=r"no training trials of class\(es\) \['rest'\].*10 s"):
        classify(runs, unmixing, cfg)


def test_classify_warns_when_epochs_reach_the_next_trial(caplog):
    runs = event_runs(spacing=10.0)
    unmixing = unmix(runs, dataclasses.replace(CCA_ONLY, metrics=("joint_avg",)), "b", HANDS)
    with caplog.at_level(logging.WARNING):
        classify(runs, unmixing, CLS, logging.getLogger("test"))
    assert "past the shortest gap between two onsets (10.0 s)" in caplog.text


def test_a_temporal_model_classifies_every_feature_set_on_the_same_trials():
    runs = event_runs()
    tcca = dataclasses.replace(CCA_ONLY, methods=("tCCA",), metrics=("joint_avg",),
                               grid_overrides={"tCCA": {"time_shifts": [[0, 1]]}})
    result = classify(runs, unmix(runs, tcca, "b", HANDS), FAST)
    counts = {(s.n_train, s.n_test, s.dropped_train, s.dropped_test) for s in result.scores}
    assert len(counts) == 1
    (n_train, n_test, dropped_train, dropped_test), = counts
    assert n_train + dropped_train == 4 * 12 and n_test + dropped_test == 2 * 12
    assert all(s.accuracy > 0.8 for s in result.scores)


# -------------------------------------------------------------------------
# Results and figures
# -------------------------------------------------------------------------
def classified_run() -> RunResult:
    runs = event_runs()
    unmixing_cfg = dataclasses.replace(CCA_ONLY, nested=True)
    cfg = dataclasses.replace(CFG, unmixing=unmixing_cfg, classification=FAST, save_inputs=False)
    unmixing = unmix(runs, unmixing_cfg, "b", HANDS)
    gt = ground_truth()
    geo3d = xr.DataArray(np.zeros((1, 3)), dims=("label", "pos"),
                         coords={"label": ["S1"]}).pint.quantify("mm")
    return RunResult(
        cfg=cfg, subject=load_subject(CONFIGS / "subject.yaml"),
        bad_channels=BadChannels(eeg=(), fnirs=(), eeg_detail={}, fnirs_detail={}),
        alignment=pd.DataFrame([{"group": r.group, "run": r.run} for r in runs]),
        geometry=Geometry(eeg_channels=tuple(f"e{i}" for i in range(6)),
                          fnirs_channels=tuple(f"f{i}" for i in range(5)), geo3d=geo3d,
                          eeg_geo3d=geo3d, fnirs_optodes={}, A_eeg=gt.A_eeg,
                          A_fnirs=gt.A_fnirs),
        unmixing=unmixing, inputs=runs, started="2026-10-24T12:00:00",
        finished="2026-10-24T12:01:00", classification=classify(runs, unmixing, FAST))


def test_classification_round_trips_through_the_run_folder(tmp_path):
    result = classified_run()
    loaded = load_run(save_run(result, tmp_path))
    assert loaded.classification == result.classification
    table = pd.read_csv(next(tmp_path.glob("runs/*/*/*/results/classification.csv")))
    assert {"accuracy", "recall_left", "confusion_left_as_right"} <= set(table.columns)
    assert set(table.evaluation) == {"non_nested", "nested"}


def test_classification_figure_draws_per_metric():
    result = classified_run()
    for metric in result.cfg.unmixing.metrics:
        fig = plot_classification(result, metric)
        assert fig.axes[0].get_ylabel() == "test accuracy"
        plt.close(fig)
    with pytest.raises(ValueError, match="no classification"):
        plot_classification(dataclasses.replace(result, classification=None), "joint_avg")
