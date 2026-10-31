"""
Smoke tests for the visualization module: every plot draws from a run result. The
head maps (plot_patterns, plot_bad_channels) need a real montage and probe, so they
are checked on saved runs, not here.
"""
import dataclasses

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.config import load_experiment, load_subject
from src.types import BadChannels, RunResult
from src.unmixing.run import unmix
from src.visualization import (plot_inputs, plot_isi, plot_nested_spread, plot_pair_assignment,
                               plot_pattern_recovery,
                               plot_selection_landscape, plot_source_recovery,
                               plot_time_courses, save_figures, selected_params)
from src.visualization.style import shade_stimuli
from test_unmixing import CONFIGS, HANDS, NO_TRUTH, runs_without_truth, synthetic_runs


@pytest.fixture(scope="module")
def result() -> RunResult:
    cfg = load_experiment(CONFIGS / "experiment.yaml")
    unmixing_cfg = dataclasses.replace(
        cfg.unmixing, methods=("CCA", "RidgeCCA", "tCCA"), metrics=("joint_avg", "isi_avg"),
        nested=True, n_jobs=1,
        grid_overrides={"RidgeCCA": {"l2_reg": {"eeg": [0.01, 1.0], "fnirs": [0.01, 1.0]}},
                        "tCCA": {"time_shifts": [[0, 1], [0, 2]]}})
    runs = synthetic_runs()
    return RunResult(
        cfg=dataclasses.replace(cfg, unmixing=unmixing_cfg),
        subject=load_subject(CONFIGS / "subject.yaml"),
        bad_channels=BadChannels(eeg=(), fnirs=(), eeg_detail={}, fnirs_detail={}),
        alignment=pd.DataFrame(), geometry=None, unmixing=unmix(runs, unmixing_cfg, "b", HANDS),
        inputs=runs, started="2026-10-22T12:00:00", finished="2026-10-22T12:01:00")


def test_every_plot_draws(result):
    for metric in ("joint_avg", "isi_avg"):
        plot_source_recovery(result, metric)
        plot_pattern_recovery(result, metric)
        plot_isi(result, metric)
        plot_nested_spread(result, metric, "recovery")
        plot_nested_spread(result, metric, "pattern")
        plt.close("all")
    for method in ("RidgeCCA", "tCCA"):
        plot_selection_landscape(result, method, "joint_avg")
    for selection in result.unmixing.selections:
        plot_time_courses(selection)
        plot_pair_assignment(selection, result.cfg.unmixing.assignment)
        for fold in selection.folds:
            plot_time_courses(selection, fold.val_group)
        plt.close("all")
    plot_inputs(result.inputs, groups=["a", "c"])
    plt.close("all")


def test_selected_params_has_one_row_per_method(result):
    table = selected_params(result)
    assert list(table.index) == ["CCA", "RidgeCCA", "tCCA"]
    assert list(table.columns) == ["joint_avg", "isi_avg"]


def test_plots_name_what_is_missing(result):
    with pytest.raises(ValueError, match="No selection by metric"):
        plot_source_recovery(result, "canonical_correlation_avg")
    with pytest.raises(ValueError, match="No fold"):
        plot_time_courses(result.unmixing.selections[0], "b")


def test_save_figures_writes_the_run_figures(result, tmp_path):
    folder = save_figures(result, tmp_path / "figures")
    names = {p.relative_to(folder).as_posix() for p in folder.rglob("*.*")}
    assert {"source_recovery_joint_avg.png", "isi_isi_avg.png", "pattern_recovery_joint_avg.png",
            "selected_params.csv",
            "landscape_RidgeCCA_joint_avg.png", "nested_recovery_joint_avg.png",
            "inputs_test.png", "inputs_train.png", "joint_avg/CCA/test.png",
            "joint_avg/CCA/fold_a.png", "joint_avg/CCA/fold_c.png",
            "joint_avg/CCA/assignment.png"} <= names
    assert "landscape_CCA_joint_avg.png" not in names


def test_landscape_skips_a_method_with_nothing_tuned(result):
    assert plot_selection_landscape(result, "CCA", "joint_avg") is None


@pytest.fixture(scope="module")
def real(result) -> RunResult:
    """`result` as real data gives it: no ground truth."""
    runs = runs_without_truth()
    return dataclasses.replace(result, cfg=dataclasses.replace(result.cfg, unmixing=NO_TRUTH),
                               unmixing=unmix(runs, NO_TRUTH, "b", HANDS), inputs=runs)


def test_time_courses_without_ground_truth_show_the_components_alone(real):
    selection, = real.unmixing.selections
    for fig in (plot_time_courses(selection), plot_time_courses(selection, "a")):
        assert len(fig.axes) == 2 * len(HANDS)            # one column, EEG and fNIRS per hand
        assert "canonical r" in fig.axes[0].get_ylabel()
    plt.close("all")


def test_save_figures_without_ground_truth_writes_only_what_needs_none(real, tmp_path):
    folder = save_figures(real, tmp_path / "figures")
    names = {p.relative_to(folder).as_posix() for p in folder.rglob("*.*")}
    assert {"selected_params.csv", "inputs_test.png", "inputs_train.png",
            "canonical_correlation_avg/CCA/test.png",
            "canonical_correlation_avg/CCA/fold_a.png"} <= names
    assert not any(name.startswith(("source_recovery", "pattern_recovery", "isi_", "nested_"))
                   or name.endswith("assignment.png") for name in names)


def test_stimuli_mark_movement_trials_with_their_onset_and_leave_rest_out():
    stim = pd.DataFrame({"onset": [0.0, 5.0, 10.0, 15.0], "duration": 4.0,
                         "trial_type": ["rest", "right", "left_arm", "rest"]})
    fig, ax = plt.subplots()
    shade_stimuli(ax, stim)
    assert [line.get_xdata()[0] for line in ax.lines] == [5.0, 10.0]
    assert len(ax.patches) == 2
    plt.close(fig)
