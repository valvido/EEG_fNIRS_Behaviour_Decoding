"""
Tests for the unmixing module.
"""
import dataclasses
import glob
import types
from pathlib import Path

import cedalion.models.glm as glm
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from cedalion import units

from src.config import AssignmentConfig, load_experiment
from src.types import GroundTruth, JoinedRuns, PairLabels, PreprocessedRecording
from src.unmixing.concat import concat_runs
from src.unmixing.fit import check_blocks, fit_model, shift_to_eeg_time, transform
from src.unmixing.folds import groups_of, leave_one_group_out, split
from src.unmixing.laplacian import build_laplacians, fnirs_positions
from src.unmixing.models import build_model, check_config, param_grid
from src.unmixing.metrics import fold_mean, fold_value
from src.unmixing.orient import haufe_pattern, spatial_filter
from src.unmixing.assignment import (STRATEGIES, TrainingPairs, assign_pairs, best_pairs,
                                     build_task_references)
from src.unmixing.run import find_hands, unmix
from src.unmixing.scores import (TooFewSamples, components, correlations, ground_truth_mixing,
                                 isi, score)
from src.unmixing.select import select_and_score
from src.visualization.orient import canonical_sign, oriented


def series(n: int, rate: float, start: float = 0.0) -> xr.DataArray:
    """A (time, channel) array of zeros with a regular time axis."""
    return xr.DataArray(np.zeros((n, 2)), dims=("time", "channel"),
                        coords={"time": start + np.arange(n) / rate, "channel": ["a", "b"]})


# -------------------------------------------------------------------------
# mSPoC blocks
# -------------------------------------------------------------------------
def test_check_blocks_accepts_k_eeg_samples_per_fnirs_sample():
    check_blocks(series(250 * 40, 250.0, start=10.0), series(40, 1.0, start=10.0))


def test_check_blocks_rejects_eeg_cut_to_the_fnirs_span():
    # The EEG ends at the last fNIRS sample, one block short: blocks of 243, not 250.
    eeg = series(250 * 39 + 1, 250.0)
    with pytest.raises(ValueError, match="would cut blocks of 243"):
        check_blocks(eeg, series(40, 1.0))


def test_check_blocks_accepts_the_inexact_rate_of_the_recordings():
    # A recorded 250 Hz EEG can be at 250.0032 Hz: blocks of 250 drift by about 4 ms over 300 s.
    check_blocks(series(250 * 300, 250.0032, start=-0.0017), series(300, 1.0))


def test_check_blocks_rejects_eeg_starting_after_the_fnirs():
    with pytest.raises(ValueError, match="EEG block 0 starts at 0.500 s"):
        check_blocks(series(250 * 40, 250.0, start=0.5), series(40, 1.0))


def test_check_blocks_rejects_a_rate_that_does_not_divide():
    with pytest.raises(ValueError, match="must divide the EEG rate"):
        check_blocks(series(1000, 250.0), series(12, 3.0))


# -------------------------------------------------------------------------
# Joining runs and splitting by group
# -------------------------------------------------------------------------
def recording(group: str, run: str, n: int = 20, rate: float = 7.4, start: float = 3.0,
              value: float = 0.0, eeg_rate: float | None = None,
              fnirs_channels=("f1", "f2"), eeg_channels=("C3", "C4")) -> PreprocessedRecording:
    """
    A preprocessed run whose signals are all `value`. Without `eeg_rate` the EEG is
    on the fNIRS time axis (band power); with it, the EEG has k = eeg_rate / rate
    samples per fNIRS sample (mSPoC).
    """
    def da(n_t, r, labels, dim):
        return xr.DataArray(np.full((n_t, len(labels)), value), dims=("time", dim),
                            coords={"time": start + np.arange(n_t) / r, dim: list(labels)})

    n_eeg, r_eeg = (n, rate) if eeg_rate is None else (n * round(eeg_rate / rate), eeg_rate)
    sources = ("right_hand", "left_hand")
    return PreprocessedRecording(
        group=group, run=run, eeg=da(n_eeg, r_eeg, eeg_channels, "channel"),
        fnirs=da(n, rate, fnirs_channels, "channel"), s_eeg=da(n, rate, sources, "source"),
        s_fnirs=da(n, rate, sources, "source"), geo3d=xr.DataArray(),
        eeg_geo3d=xr.DataArray(), fnirs_optodes={}, stim=pd.DataFrame(),
        window=(0.0, 1.0), ground_truth=None)


def test_concat_runs_continues_one_regular_axis():
    runs = [recording("a", "run_001", n=20, value=1.0, start=3.0),
            recording("b", "run_001", n=30, value=2.0, start=50.0)]
    joined = concat_runs(runs)

    dt = 1 / 7.4
    np.testing.assert_allclose(np.diff(joined.fnirs.time.values), dt)
    np.testing.assert_array_equal(joined.eeg.time.values, joined.fnirs.time.values)
    np.testing.assert_array_equal(joined.s_fnirs.time.values, joined.fnirs.time.values)
    assert joined.run_starts == pytest.approx((0.0, 20 * dt))
    assert joined.runs == ("a/run_001", "b/run_001")
    # Each run's samples keep their values, at their place on the new axis.
    assert (joined.fnirs.sel(time=slice(None, 19.5 * dt)) == 1.0).all()
    assert (joined.s_eeg.sel(time=slice(19.5 * dt, None)) == 2.0).all()


def stim(*onsets: float) -> pd.DataFrame:
    """A stimulus table with alternating hands, 4 s each."""
    return pd.DataFrame({"onset": onsets, "duration": 4.0, "value": 1.0,
                         "trial_type": ["right", "left"] * (len(onsets) // 2) + ["right"] * (len(onsets) % 2)})


def test_concat_runs_moves_the_stimuli_with_their_run():
    # Run a covers 3.0 to 5.6 s, run b 50.0 to 53.9 s on their own clocks.
    runs = [dataclasses.replace(recording("a", "run_001", n=20, start=3.0), stim=stim(1.0, 4.0)),
            dataclasses.replace(recording("b", "run_001", n=30, start=50.0),
                                stim=stim(51.0, 52.5, 60.0))]
    joined = concat_runs(runs)

    dt = 1 / 7.4
    # Onsets before or after a run's data are left out; the rest move with the run.
    assert list(joined.stim["run"]) == ["a/run_001", "b/run_001", "b/run_001"]
    np.testing.assert_allclose(joined.stim["onset"], [1.0, 20 * dt + 1.0, 20 * dt + 2.5])
    assert list(joined.stim["trial_type"]) == ["left", "right", "left"]


def test_concat_runs_keeps_mspoc_blocks_on_their_fnirs_samples():
    runs = [recording("a", "run_001", n=12, rate=1.0, eeg_rate=250.0, value=1.0),
            recording("b", "run_001", n=9, rate=1.0, eeg_rate=250.0, value=2.0, start=40.0)]
    joined = concat_runs(runs)
    check_blocks(joined.eeg, joined.fnirs)
    # Block 12, the first of run b, holds run b's EEG.
    assert (joined.eeg.isel(time=slice(12 * 250, 13 * 250)) == 2.0).all()
    assert (joined.eeg.isel(time=slice(11 * 250, 12 * 250)) == 1.0).all()


def test_concat_runs_puts_channels_in_the_first_runs_order():
    runs = [recording("a", "run_001"),
            recording("b", "run_001", fnirs_channels=("f2", "f1"), eeg_channels=("C4", "C3"))]
    joined = concat_runs(runs)
    assert list(joined.fnirs.channel.values) == ["f1", "f2"]
    assert list(joined.eeg.channel.values) == ["C3", "C4"]


def test_concat_runs_rejects_runs_with_other_channels():
    runs = [recording("a", "run_001"), recording("b", "run_001", fnirs_channels=("f1", "f3"))]
    with pytest.raises(ValueError, match=r"b/run_001: fNIRS channels differ.*f2.*f3"):
        concat_runs(runs)


def test_concat_runs_rejects_runs_at_another_rate():
    runs = [recording("a", "run_001"), recording("b", "run_001", rate=10.0)]
    with pytest.raises(ValueError, match="b/run_001: fNIRS rate 10 Hz"):
        concat_runs(runs)


def test_concat_runs_averages_the_inverse_stds_weighted_by_run_length():
    std = lambda values: xr.DataArray(values, dims="channel", coords={"channel": ["f1", "f2"]})
    runs = [dataclasses.replace(recording("a", "run_001", n=20), fnirs_std=std([1.0, 4.0])),
            dataclasses.replace(recording("b", "run_001", n=30), fnirs_std=std([2.0, 4.0]))]
    joined = concat_runs(runs)
    # f1: (20 * 1/1 + 30 * 1/2) / 50; f2 has the same std in both runs.
    np.testing.assert_allclose(joined.fnirs_gain.sel(channel=["f1", "f2"]), [0.7, 0.25])
    assert joined.eeg_gain is None                                  # no run scaled its EEG


def test_concat_runs_rejects_runs_scaled_differently():
    std = xr.DataArray([1.0, 1.0], dims="channel", coords={"channel": ["f1", "f2"]})
    runs = [dataclasses.replace(recording("a", "run_001"), fnirs_std=std),
            recording("b", "run_001")]
    with pytest.raises(ValueError, match="Some runs have their fNIRS std"):
        concat_runs(runs)


RUNS = [recording(g, r) for g in ("a", "b", "c") for r in ("run_001", "run_002")]


def test_split_holds_out_the_test_group():
    dev, test = split(RUNS, "b")
    assert {rec.group for rec in test} == {"b"} and len(test) == 2
    assert {rec.group for rec in dev} == {"a", "c"} and len(dev) == 4


def test_split_rejects_a_missing_test_group():
    with pytest.raises(ValueError, match="No runs of test group 'x'"):
        split(RUNS, "x")


def test_leave_one_group_out_validates_on_each_group_once():
    folds = leave_one_group_out(RUNS)
    assert [groups_of(val) for _, val in folds] == [["a"], ["b"], ["c"]]
    for train, val in folds:
        assert not set(groups_of(train)) & set(groups_of(val))
        assert len(train) + len(val) == len(RUNS)


def test_leave_one_group_out_folds_come_in_sorted_group_order():
    runs = [recording(g, "run_001") for g in ("c", "a", "b")]
    folds = leave_one_group_out(runs)
    assert [groups_of(val) for _, val in folds] == [["a"], ["b"], ["c"]]
    assert [groups_of(train) for train, _ in folds] == [["c", "b"], ["c", "a"], ["a", "b"]]


def test_leave_one_group_out_needs_two_groups():
    with pytest.raises(ValueError, match="at least two groups"):
        leave_one_group_out([rec for rec in RUNS if rec.group == "a"])



# -------------------------------------------------------------------------
# Methods and grids
# -------------------------------------------------------------------------
# Fixture configs, so no test depends on the real experiments.
CONFIGS = Path(__file__).parent / "configs"
CFG = load_experiment(CONFIGS / "experiment.yaml")
HANDS = ["right_hand", "left_hand"]


def unmixing(**changes):
    return dataclasses.replace(CFG.unmixing, **changes)


def test_param_grid_tries_every_eeg_fnirs_pair():
    grid = param_grid("ElasticNetCCA", CFG.unmixing)
    assert len(grid) == 25 * 16
    assert [0.02, 0.2] in [g["l1_reg"] for g in grid]


def test_param_grid_gives_time_shifts_as_arrays_and_cca_one_empty_combination():
    shifts = [g["time_shifts"] for g in param_grid("tCCA", CFG.unmixing)]
    assert all(isinstance(t, np.ndarray) for t in shifts) and len(shifts) == 3
    assert param_grid("CCA", CFG.unmixing) == [{}]


def test_grid_overrides_replace_a_whole_key():
    cfg = unmixing(grid_overrides={"tCCA": {"time_shifts": [[0, 2]]}})
    assert [list(g["time_shifts"]) for g in param_grid("tCCA", cfg)] == [[0, 2]]


def test_param_grid_rejects_time_shifts_for_a_non_temporal_method():
    cfg = unmixing(grid_overrides={"CCA": {"time_shifts": [[0, 2]]}})
    with pytest.raises(ValueError, match="CCA has no hyperparameter 'time_shifts'"):
        param_grid("CCA", cfg)


@pytest.mark.parametrize("changes, feature, message", [
    ({"methods": ("tCCA", "Foo")}, "bandpower", "Unknown method"),
    ({"methods": ("mSPoC",), "n_components": 2}, "bandpower", "needs eeg_feature: bandpassed"),
    ({"methods": ("CCA",)}, "bandpassed", "needs eeg_feature: bandpower"),
    ({"methods": ("mSPoC",), "n_components": None}, "bandpassed", "mSPoC needs n_components"),
    ({"laplacian": {"eeg": {"method": "nope"}}}, "bandpower", "laplacian.eeg.method"),
])
def test_check_config_rejects(changes, feature, message):
    with pytest.raises(ValueError, match=message):
        check_config(unmixing(**changes), feature)


@pytest.mark.parametrize("path", sorted(glob.glob("configs/experiments/**/*.yaml",
                                                  recursive=True)))
def test_check_config_accepts_every_experiment(path):
    cfg = load_experiment(path)
    check_config(cfg.unmixing, cfg.preprocessing.eeg_feature)


def test_temporal_models_leave_their_eeg_components_unshifted():
    model = build_model("tCCA", {"time_shifts": np.array([0.0, 2.0])}, 2)
    assert not model.shift_source


# -------------------------------------------------------------------------
# Fitting, sign and transform
# -------------------------------------------------------------------------
RATE = 10.0


def two_sources(n: int, seed: int = 0) -> np.ndarray:
    """Two independent smooth sources, dims (time, source)."""
    rng = np.random.default_rng(seed)
    kernel = np.ones(15) / 15
    return np.stack([np.convolve(rng.standard_normal(n), kernel, mode="same")
                     for _ in range(2)], axis=1)


def mixing(n_channels: int, seed: int) -> np.ndarray:
    """(channel, source): each source peaks positively at its own channel."""
    rng = np.random.default_rng(seed)
    A = 0.3 * rng.random((n_channels, 2))
    A[0, 0] = A[1, 1] = 2.0
    return A


def joined(s: np.ndarray, delay_s: float = 0.0, noise: float = 0.1, seed: int = 1,
           n_eeg: int = 6, n_fnirs: int = 5) -> JoinedRuns:
    """
    EEG band power = s (A_eeg ** 2)' + noise, as band power adds over sources with the
    square of their field; fNIRS = s_fnirs(t - delay) A_fnirs' + noise, both at RATE,
    with the ground truth on the same axis (the fNIRS source delayed like the fNIRS).
    The second fNIRS source shares only part of its course with the EEG source, so
    the two canonical correlations differ; equal ones would leave CCA free to return
    any rotation of the two pairs.
    """
    rng = np.random.default_rng(seed)
    lag = int(round(delay_s * RATE))
    own = two_sources(len(s), seed=seed + 100)[:, 1]
    s_fnirs = np.roll(np.stack([s[:, 0], 0.7 * s[:, 1] + 0.7 * own], axis=1), lag, axis=0)
    eeg = s @ (mixing(n_eeg, 2) ** 2).T + noise * rng.standard_normal((len(s), n_eeg))
    fnirs = s_fnirs @ mixing(n_fnirs, 3).T + noise * rng.standard_normal((len(s), n_fnirs))
    time = np.arange(len(s)) / RATE

    def da(values, labels, dim):
        return xr.DataArray(values, dims=("time", dim), coords={"time": time, dim: labels})

    sources = ["right_hand", "left_hand"]
    return JoinedRuns(eeg=da(eeg, [f"e{i}" for i in range(n_eeg)], "channel"),
                      fnirs=da(fnirs, [f"f{i}" for i in range(n_fnirs)], "channel"),
                      s_eeg=da(s, sources, "source"), s_fnirs=da(s_fnirs, sources, "source"),
                      run_starts=(0.0,), runs=("a/run_001",), stim=pd.DataFrame(),
                      ground_truth=None)


def corr(a: xr.DataArray, b: xr.DataArray) -> float:
    """Correlation over the time points both share and where both are finite."""
    a, b = xr.align(a, b)
    keep = np.isfinite(a.values) & np.isfinite(b.values)
    return float(np.corrcoef(a.values[keep], b.values[keep])[0, 1])


def best_match(components: xr.DataArray, source: xr.DataArray) -> float:
    """The correlation of the component most correlated with `source`, signed."""
    rs = [corr(components.sel(component=k), source) for k in components.component.values]
    return rs[int(np.argmax(np.abs(rs)))]


def test_haufe_pattern_recovers_the_mixing_of_clean_data():
    data = joined(two_sources(3000), noise=0.01)
    fitted = fit_model("CCA", {}, data, n_components=2)
    pattern = haufe_pattern(spatial_filter(fitted.model, "eeg"), data.eeg)
    A = mixing(6, 2) ** 2                                           # band power mixing
    for k in range(2):
        r = [abs(np.corrcoef(pattern.isel(component=k), A[:, j])[0, 1])
             for j in range(2)]
        assert max(r) > 0.99


def test_plots_flip_a_component_and_its_pattern_together():
    time = np.arange(4.0)
    comps = xr.DataArray(np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]]).T,
                         dims=("time", "component"), coords={"time": time, "component": [0, 1]})
    patterns = xr.DataArray([[-3.0, 1.0], [1.0, 2.0]], dims=("channel", "component"),
                            coords={"channel": ["a", "b"], "component": [0, 1]})
    recovery = types.SimpleNamespace(u=comps, v=comps, pattern_eeg=patterns, pattern_fnirs=patterns)

    component, pattern = oriented(recovery, "CCA", "fnirs", 0)
    np.testing.assert_array_equal(component.values, -comps.sel(component=0).values)
    np.testing.assert_array_equal(pattern.values, [3.0, -1.0])
    component, pattern = oriented(recovery, "CCA", "fnirs", 1)
    np.testing.assert_array_equal(component.values, comps.sel(component=1).values)
    # mSPoC's EEG component is band power: its pattern flips, the time course does not.
    component, pattern = oriented(recovery, "mSPoC", "eeg", 0)
    np.testing.assert_array_equal(component.values, comps.sel(component=0).values)
    np.testing.assert_array_equal(pattern.values, [3.0, -1.0])


def test_canonical_sign_makes_each_pattern_positive_at_its_peak():
    A = xr.DataArray([[1.0, -3.0], [-2.0, 0.5]], dims=("channel", "component"),
                     coords={"component": [0, 1]})
    np.testing.assert_array_equal(canonical_sign(A).values, [-1.0, -1.0])
    flipped = A * canonical_sign(A)
    assert all(flipped.isel(component=k).values[np.argmax(np.abs(flipped.isel(component=k).values))] > 0
               for k in range(2))


def test_transform_keeps_the_sign_the_model_gives():
    fitted = fit_model("CCA", {}, joined(two_sources(3000, seed=0)), n_components=2)
    test = joined(two_sources(3000, seed=5), seed=6)
    u, v = transform(fitted, test)

    raw_u, raw_v = fitted.model.transform(test.eeg, test.fnirs)
    np.testing.assert_allclose(u.values, raw_u.values)
    np.testing.assert_allclose(v.values, raw_v.values)
    for source in ("right_hand", "left_hand"):
        assert abs(best_match(u, test.s_eeg.sel(source=source))) > 0.9
        assert abs(best_match(v, test.s_fnirs.sel(source=source))) > 0.9


def test_tcca_transform_gives_the_pair_as_fitted_on_the_full_time_axis():
    data = joined(two_sources(3000), delay_s=2.0)
    fitted = fit_model("tCCA", {"time_shifts": np.array([0.0, 2.0])}, data, n_components=2)
    u_paired, v = transform(fitted, data)
    assert set(np.atleast_1d(fitted.model.optimal_shift)) == {2.0}
    np.testing.assert_array_equal(u_paired.time.values, data.fnirs.time.values)
    np.testing.assert_array_equal(v.time.values, data.fnirs.time.values)
    # Each EEG component follows its fNIRS component and the delayed source, not
    # the EEG source. The second fNIRS source shares only part of its course with
    # the EEG source (`joined`), so its pair correlates less.
    for k in range(2):
        assert corr(u_paired.isel(component=k), v.isel(component=k)) > 0.6
    eeg_source_delayed = data.s_eeg.shift(time=int(2.0 * RATE))
    for source in ("right_hand", "left_hand"):
        assert abs(best_match(u_paired, eeg_source_delayed.sel(source=source))) > 0.9
        assert abs(best_match(u_paired, data.s_eeg.sel(source=source))) < 0.5


def test_tcca_shift_to_eeg_time_matches_cedalions_own_shift():
    data = joined(two_sources(3000), delay_s=2.0)
    fitted = fit_model("tCCA", {"time_shifts": np.array([0.0, 1.0, 2.0])}, data, n_components=2)
    u = shift_to_eeg_time(fitted, transform(fitted, data)[0])

    fitted.model.shift_source = True
    reference, _ = fitted.model.transform(data.eeg, data.fnirs)
    fitted.model.shift_source = False
    reference = reference.transpose("time", ...).values
    shifts = np.atleast_1d(fitted.model.optimal_shift)
    for k, shift in enumerate(shifts):
        # cedalion ends its output the smallest shift early and pads larger shifts
        # with zeros; every sample it has a value for must agree.
        n_valid = 3000 - int(round(shift * RATE))
        np.testing.assert_allclose(u.values[:n_valid, k], reference[:n_valid, k])
        assert np.isnan(u.values[n_valid:, k]).all()
    for source in ("right_hand", "left_hand"):
        assert abs(best_match(u, data.s_eeg.sel(source=source))) > 0.9


def test_shift_to_eeg_time_leaves_an_unshifted_component_whole():
    data = joined(two_sources(3000), delay_s=2.0)
    fitted = fit_model("tCCA", {"time_shifts": np.array([0.0, 2.0])}, data, n_components=2)
    fitted.model.optimal_shift = np.array([0.0, 2.0])
    u = shift_to_eeg_time(fitted, transform(fitted, data)[0])
    assert np.isfinite(u.isel(component=0)).all()
    assert np.isnan(u.isel(component=1)[-20:]).all()
    assert np.isfinite(u.isel(component=1)[:-20]).all()


def test_tcca_canonical_correlation_compares_the_pair_as_fitted():
    gt = ground_truth()
    train = dataclasses.replace(joined(two_sources(3000, seed=0), delay_s=2.0), ground_truth=gt)
    test = dataclasses.replace(joined(two_sources(3000, seed=5), delay_s=2.0, seed=6),
                               ground_truth=gt)
    fitted = fit_model("tCCA", {"time_shifts": np.array([0.0, 2.0])}, train, n_components=2)
    scores, _ = score(fitted, train, test, AssignmentConfig(), HANDS, with_patterns=False)
    # Compared after the shift to the EEG's time, these would be about 0.
    assert all(r > 0.6 for r in scores.canonical_r.values())
    assert all(r > 0.9 for r in scores.r_eeg.values())


def test_transform_refuses_a_model_that_shifts_its_own_components():
    data = joined(two_sources(3000), delay_s=2.0)
    fitted = fit_model("tCCA", {"time_shifts": np.array([0.0, 2.0])}, data, n_components=2)
    fitted.model.shift_source = True
    with pytest.raises(ValueError, match="shift_source=True"):
        transform(fitted, data)


def mspoc_data(n_fnirs: int = 200, k: int = 50, seed: int = 0) -> JoinedRuns:
    """
    Band-passed EEG at k samples per fNIRS sample whose power in two channels follows
    the fNIRS sources.
    """
    rng = np.random.default_rng(seed)
    s = np.abs(two_sources(n_fnirs, seed)) + 0.2
    carrier = rng.standard_normal((n_fnirs * k, 4))
    power = np.repeat(s, k, axis=0)
    carrier[:, :2] *= np.sqrt(power)
    eeg_time = np.arange(n_fnirs * k) / k
    fnirs = s @ mixing(3, 3).T + 0.05 * rng.standard_normal((n_fnirs, 3))
    sources = ["right_hand", "left_hand"]
    da = lambda v, t, labels, dim: xr.DataArray(v, dims=("time", dim),
                                                 coords={"time": t, dim: labels})
    return JoinedRuns(eeg=da(carrier, eeg_time, ["e0", "e1", "e2", "e3"], "channel"),
                      fnirs=da(fnirs, np.arange(n_fnirs, dtype=float), ["f0", "f1", "f2"], "channel"),
                      s_eeg=da(s, np.arange(n_fnirs, dtype=float), sources, "source"),
                      s_fnirs=da(s, np.arange(n_fnirs, dtype=float), sources, "source"),
                      run_starts=(0.0,), runs=("a/run_001",), stim=pd.DataFrame(),
                      ground_truth=None)


MSPOC_PARAMS = {"time_shifts": np.array([0.0, 2.0]), "N_restarts": 2, "max_iter": 30,
                "tol": 1e-3}


def test_mspoc_fits_and_gives_both_components_on_one_time_axis():
    data = mspoc_data()
    fitted = fit_model("mSPoC", MSPOC_PARAMS, data, n_components=2, seed=0)
    u, v = transform(fitted, data)
    assert u.sizes["component"] == v.sizes["component"] == 2
    np.testing.assert_array_equal(u.time.values, v.time.values)


def test_mspoc_shift_to_eeg_time_matches_cedalions_own_shift():
    data = mspoc_data()
    fitted = fit_model("mSPoC", MSPOC_PARAMS, data, n_components=2, seed=0)
    u = shift_to_eeg_time(fitted, transform(fitted, data)[0])

    fitted.model.shift_source = True
    reference, _ = fitted.model.transform(data.eeg, data.fnirs)
    fitted.model.shift_source = False
    reference = reference.transpose("time", ...).values
    for k, shift in enumerate(np.atleast_1d(fitted.model.optimal_shift)):
        n_valid = data.fnirs.sizes["time"] - int(shift)          # fNIRS at 1 Hz
        np.testing.assert_allclose(u.values[:n_valid, k], reference[:n_valid, k])
        assert np.isnan(u.values[n_valid:, k]).all()


def test_mspoc_refuses_eeg_that_is_not_cut_into_blocks():
    data = mspoc_data()
    short = dataclasses.replace(data, eeg=data.eeg.isel(time=slice(0, -49)))
    with pytest.raises(ValueError, match="EEG samples per fNIRS sample"):
        fit_model("mSPoC", MSPOC_PARAMS, short, n_components=2, seed=0)



# -------------------------------------------------------------------------
# Assignment of the pairs to the hands
# -------------------------------------------------------------------------


def r_table(rows) -> xr.DataArray:
    """Correlations, dims (component, source): one row per pair, columns right_hand, left_hand."""
    return xr.DataArray(np.asarray(rows, dtype=float), dims=("component", "source"),
                        coords={"component": np.arange(len(rows)), "source": HANDS})


def label_pairs(r_eeg, r_fnirs, cfg):
    """The ground_truth strategy on given training correlations."""
    pairs = TrainingPairs(fitted=None, data=None, u=None, v=None, r_eeg=r_eeg, r_fnirs=r_fnirs)
    return STRATEGIES["ground_truth"].assign(pairs, HANDS, cfg)


def test_assignment_by_ground_truth_needs_ground_truth():
    pairs = TrainingPairs(fitted=None, data=None, u=None, v=None, r_eeg=None, r_fnirs=None)
    with pytest.raises(ValueError, match="needs ground truth"):
        assign_pairs(pairs, HANDS, AssignmentConfig())


def motor_channel_config(**changes) -> AssignmentConfig:
    """The motor_channels strategy with the channels `mixing` puts each source on."""
    channels = {"eeg": {"right_hand": ("e0",), "left_hand": ("e1",)},
                "fnirs": {"right_hand": ("f0",), "left_hand": ("f1",)}}
    return AssignmentConfig(**{"method": "motor_channels", "motor_channels": channels,
                               **changes})


def test_motor_channels_agrees_with_the_ground_truth_on_clean_data():
    data = joined(two_sources(3000), noise=0.01)
    fitted = fit_model("CCA", {}, data, n_components=2)
    u, _, v = components(fitted, data)
    pairs = TrainingPairs(fitted=fitted, data=data, u=u, v=v,
                          r_eeg=correlations(u, data.s_eeg),
                          r_fnirs=correlations(v, data.s_fnirs))
    by_channels = assign_pairs(pairs, HANDS, motor_channel_config())
    by_truth = assign_pairs(pairs, HANDS, AssignmentConfig())
    assert by_channels.pairs == by_truth.pairs


def test_motor_channels_runs_without_ground_truth():
    data = dataclasses.replace(joined(two_sources(500)), s_eeg=None, s_fnirs=None)
    fitted = fit_model("CCA", {}, data, n_components=2)
    pairs = TrainingPairs(fitted=fitted, data=data, u=None, v=None, r_eeg=None, r_fnirs=None)
    labels = assign_pairs(pairs, HANDS, motor_channel_config())
    assert sorted(labels.pairs.values()) == [0, 1]


def test_motor_channels_skips_a_modality_with_weight_0():
    data = joined(two_sources(500))
    fitted = fit_model("CCA", {}, data, n_components=2)
    pairs = TrainingPairs(fitted=fitted, data=data, u=None, v=None, r_eeg=None, r_fnirs=None)
    cfg = AssignmentConfig(method="motor_channels", fnirs_weight=0.0,
                           motor_channels={"eeg": {"right_hand": ("e0",), "left_hand": ("e1",)}})
    assert sorted(assign_pairs(pairs, HANDS, cfg).pairs.values()) == [0, 1]


def test_motor_channels_rejects_channels_missing_from_the_data():
    data = joined(two_sources(500))
    fitted = fit_model("CCA", {}, data, n_components=2)
    pairs = TrainingPairs(fitted=fitted, data=data, u=None, v=None, r_eeg=None, r_fnirs=None)
    cfg = motor_channel_config(motor_channels={
        "eeg": {"right_hand": ("C3",), "left_hand": ("e1",)},
        "fnirs": {"right_hand": ("f0",), "left_hand": ("f1",)}})
    with pytest.raises(ValueError, match="None of the eeg motor channels"):
        assign_pairs(pairs, HANDS, cfg)


def test_motor_channels_method_needs_channels_for_every_weighted_modality():
    with pytest.raises(ValueError, match=r"needs motor_channels for \['fnirs'\]"):
        AssignmentConfig(method="motor_channels",
                         motor_channels={"eeg": {"right_hand": ("C3",), "left_hand": ("C4",)}})


TRIAL_CLASSES = {"right": ("right", "right_arm"), "left": ("left", "left_arm")}
BOXCAR = glm.DiracDelta()


def trial_stim(n: int, period_s: float = 15.0, duration_s: float = 6.0) -> pd.DataFrame:
    """Trials every `period_s` over n samples at RATE: right, left, right_arm, left_arm, ..."""
    onsets = np.arange(5.0, n / RATE - period_s, period_s)
    kinds = ["right", "left", "right_arm", "left_arm"]
    return pd.DataFrame({"onset": onsets, "duration": duration_s, "value": 0.5,
                         "trial_type": [kinds[i % 4] for i in range(len(onsets))]})


def task_data(n: int = 3000, seed: int = 0) -> JoinedRuns:
    """`joined` with two sources that follow the right and the left trials of `trial_stim`."""
    stim = trial_stim(n)
    time = np.arange(n) / RATE
    rng = np.random.default_rng(seed)
    kernel = np.ones(15) / 15
    s = np.stack([np.convolve(_in_trials(time, stim, TRIAL_CLASSES[side]), kernel, mode="same")
                  + 0.3 * np.convolve(rng.standard_normal(n), kernel, mode="same")
                  for side in ("right", "left")], axis=1)
    return dataclasses.replace(joined(s, noise=0.05), stim=stim)


def _in_trials(time: np.ndarray, stim: pd.DataFrame, trial_types) -> np.ndarray:
    """1.0 during the trials of `trial_types`, else 0.0."""
    trials = stim[stim.trial_type.isin(trial_types)]
    return np.any([(time >= o) & (time < o + d) for o, d in zip(trials.onset, trials.duration)],
                  axis=0).astype(float)


def test_task_reference_is_a_boxcar_over_the_hands_trial_types():
    data = task_data(1000)
    hand_trials = {"right_hand": TRIAL_CLASSES["right"], "left_hand": TRIAL_CLASSES["left"]}
    reference = build_task_references(data, hand_trials, BOXCAR)
    time = data.fnirs.time.values
    for hand, side in (("right_hand", "right"), ("left_hand", "left")):
        expected = _in_trials(time, data.stim, TRIAL_CLASSES[side])
        on = reference.sel(source=hand).values > 0.5
        assert np.mean(on == (expected > 0.5)) > 0.98


def test_task_reference_uses_only_the_cue_timing():
    data = task_data(1000)
    louder = dataclasses.replace(data, stim=data.stim.assign(value=1.0))
    hand_trials = {"right_hand": TRIAL_CLASSES["right"]}
    np.testing.assert_allclose(build_task_references(data, hand_trials, BOXCAR),
                               build_task_references(louder, hand_trials, BOXCAR))


def test_task_reference_of_a_hand_without_trials_is_zero():
    reference = build_task_references(task_data(1000), {"right_hand": ("nothing",)}, BOXCAR)
    assert not reference.values.any()


def test_fnirs_task_reference_peaks_after_the_trial_onset():
    data = task_data(1000)
    hrf = glm.Gamma(tau=0 * units.s, sigma=3 * units.s, T=0 * units.s)
    reference = build_task_references(data, {"left_hand": ("left",)}, hrf).sel(source="left_hand")
    first = data.stim[data.stim.trial_type == "left"].onset.iloc[0]
    one_trial = reference.sel(time=slice(first, first + 14.0))
    assert 2.0 < float(one_trial.idxmax()) - first < 10.0


def test_task_regressors_agree_with_the_ground_truth():
    data = task_data()
    fitted = fit_model("CCA", {}, data, n_components=2)
    u, _, v = components(fitted, data)
    pairs = TrainingPairs(fitted=fitted, data=data, u=u, v=v,
                          r_eeg=correlations(u, data.s_eeg), r_fnirs=correlations(v, data.s_fnirs),
                          trial_classes=TRIAL_CLASSES)
    by_trials = assign_pairs(pairs, HANDS, AssignmentConfig(method="task_regressors"))
    assert by_trials.pairs == assign_pairs(pairs, HANDS, AssignmentConfig()).pairs


def test_task_regressors_need_the_trial_classes():
    data = task_data(500)
    fitted = fit_model("CCA", {}, data, n_components=2)
    u, _, v = components(fitted, data)
    pairs = TrainingPairs(fitted=fitted, data=data, u=u, v=v, r_eeg=None, r_fnirs=None)
    with pytest.raises(ValueError, match="trial_classes"):
        assign_pairs(pairs, HANDS, AssignmentConfig(method="task_regressors"))


def test_fnirs_hrf_rejects_unknown_parameters():
    with pytest.raises(ValueError, match="fnirs_hrf"):
        AssignmentConfig(method="task_regressors", fnirs_hrf={"tau": 0.0, "delay": 2.0})


def test_unknown_assignment_method_is_rejected():
    with pytest.raises(ValueError, match="assignment.method"):
        check_config(unmixing(assignment=AssignmentConfig(method="nope")), "bandpower")


def test_label_pairs_swaps_when_pair_1_carries_the_right_hand():
    r = r_table([[0.1, 0.8], [-0.7, 0.2], [0.0, 0.0]])
    labels = label_pairs(r, r, AssignmentConfig())
    assert labels.pairs == {"right_hand": 1, "left_hand": 0}
    assert labels.score == pytest.approx(2 * (0.7 + 0.8))


def test_label_pairs_weights_decide_when_the_modalities_disagree():
    r_eeg = r_table([[0.9, 0.1], [0.1, 0.9]])          # EEG: pair 0 is right_hand
    r_fnirs = r_table([[0.1, 0.6], [0.6, 0.1]])        # fNIRS: pair 0 is left_hand
    assert label_pairs(r_eeg, r_fnirs, AssignmentConfig()).pairs == {"right_hand": 0, "left_hand": 1}
    eeg_light = AssignmentConfig(eeg_weight=0.1, fnirs_weight=1.0)
    assert label_pairs(r_eeg, r_fnirs, eeg_light).pairs == {"right_hand": 1, "left_hand": 0}


def test_label_pairs_looks_only_at_the_top_pairs():
    r = r_table([[0.5, 0.1], [0.1, 0.4], [0.9, 0.1], [0.1, 0.9]])
    assert label_pairs(r, r, AssignmentConfig(n_top=2)).pairs == {"right_hand": 0, "left_hand": 1}
    assert label_pairs(r, r, AssignmentConfig(n_top=4)).pairs == {"right_hand": 2, "left_hand": 3}


def test_best_pairs_finds_a_source_outside_the_top_pairs():
    r = r_table([[0.3, 0.2], [0.2, 0.3], [0.0, 0.1], [0.1, 0.0], [0.9, 0.1]])
    assert best_pairs(r, r, AssignmentConfig()) == {"right_hand": 4, "left_hand": 1}


# -------------------------------------------------------------------------
# Scores and metrics
# -------------------------------------------------------------------------
def test_correlations_match_by_time_and_skip_nan():
    t = np.arange(100) / RATE
    rng = np.random.default_rng(0)
    s = rng.standard_normal(100)
    comps = xr.DataArray(np.stack([s[:90], -s[:90]], axis=1), dims=("time", "component"),
                         coords={"time": t[:90], "component": [0, 1]})
    comps[:5, 0] = np.nan
    sources = xr.DataArray(np.stack([s, rng.standard_normal(100)], axis=1),
                           dims=("time", "source"), coords={"time": t, "source": ["right_hand", "left_hand"]})
    r = correlations(comps, sources)
    assert float(r.sel(component=0, source="right_hand")) == pytest.approx(1.0)
    assert float(r.sel(component=1, source="right_hand")) == pytest.approx(-1.0)


def test_components_mask_the_eeg_around_run_starts_for_temporal_models_only():
    data = joined(two_sources(3000), delay_s=2.0)
    two_runs = dataclasses.replace(data, run_starts=(0.0, 150.0))
    tcca = fit_model("tCCA", {"time_shifts": np.array([0.0, 2.0])}, data, n_components=2)
    u, u_paired, v = components(tcca, two_runs)
    t = u.time.values
    for k, shift in enumerate(tcca.model.optimal_shift):
        # The max(tau) wide window after each run start, the first included: as it
        # is in the pair, and moved back by tau_k on the EEG's time.
        paired_out = (t < 2.0 - 1e-9) | ((t >= 150.0 - 1e-9) & (t < 152.0 - 1e-9))
        assert np.isnan(u_paired.values[paired_out, k]).all()
        assert np.isfinite(u_paired.values[~paired_out, k]).all()
        out = (t < 2.0 - shift - 1e-9) | ((t >= 150.0 - shift - 1e-9) & (t < 152.0 - shift - 1e-9))
        assert np.isnan(u.values[out, k]).all()
        assert np.isfinite(u.values[~out & (t < 300.0 - shift - 1e-9), k]).all()
    assert np.isfinite(v.values).all()
    cca = fit_model("CCA", {}, data, n_components=2)
    u_cca, u_paired_cca, _ = components(cca, two_runs)
    assert np.isfinite(u_cca.values).all()
    xr.testing.assert_identical(u_cca, u_paired_cca)


def ground_truth(n_eeg: int = 6, n_fnirs: int = 5) -> GroundTruth:
    """
    The ground truth of `joined`, in the simulator's layout. A_eeg is the EEG field;
    `joined` mixes its EEG band power by its square.
    """
    A_eeg = xr.DataArray(mixing(n_eeg, 2), dims=("channel", "source"),
                         coords={"channel": [f"e{i}" for i in range(n_eeg)], "source": ["right_hand", "left_hand"]})
    A = mixing(n_fnirs, 3)
    A_fnirs = xr.DataArray(np.stack([np.stack([A, A], -1), np.stack([1.8 * A, A], -1)], 2),
                           dims=("channel", "source", "wavelength", "chromo"),
                           coords={"channel": [f"f{i}" for i in range(n_fnirs)],
                                   "source": ["right_hand", "left_hand"], "wavelength": [760.0, 850.0],
                                   "chromo": ["HbO", "HbR"]})
    return GroundTruth(s_eeg=None, s_fnirs=None, A_eeg=A_eeg, A_fnirs=A_fnirs)


def fake_fitted(W: np.ndarray, channels: list[str], method: str = "CCA"):
    """A FittedModel of `method` whose EEG filter is W (channel, component)."""
    Wx = xr.DataArray(W, dims=("channel", "lat"), coords={"channel": channels})
    model = types.SimpleNamespace(Wx=Wx, latent_featureX_name="lat", x_std=None)
    return types.SimpleNamespace(model=model, method=method)


def test_isi_is_zero_for_a_perfect_unmixing_in_any_channel_order():
    gt = ground_truth()
    A = gt.A_eeg.values ** 2                                        # band power mixing
    W = np.linalg.pinv(A).T                                         # W' A = I
    data = dataclasses.replace(joined(two_sources(100)), ground_truth=gt)
    labels = PairLabels(pairs={"right_hand": 0, "left_hand": 1}, score=0.0)
    channels = [f"e{i}" for i in range(6)]
    assert isi(fake_fitted(W, channels), labels, data, "eeg") == pytest.approx(0.0, abs=1e-9)
    order = [3, 0, 5, 1, 4, 2]
    shuffled = fake_fitted(W[order], [channels[i] for i in order])
    assert isi(shuffled, labels, data, "eeg") == pytest.approx(0.0, abs=1e-9)
    swapped = PairLabels(pairs={"right_hand": 1, "left_hand": 0}, score=0.0)
    assert isi(fake_fitted(W, channels), swapped, data, "eeg") == pytest.approx(0.0, abs=1e-9)
    mixed = fake_fitted(np.ones((6, 2)), channels)
    assert isi(mixed, labels, data, "eeg") > 0.5


def test_isi_does_not_depend_on_the_filter_scales():
    gt = ground_truth()
    A = gt.A_eeg.values ** 2                                        # band power mixing
    data = dataclasses.replace(joined(two_sources(100)), ground_truth=gt)
    labels = PairLabels(pairs={"right_hand": 0, "left_hand": 1}, score=0.0)
    channels = [f"e{i}" for i in range(6)]
    # Both filters see mostly the right hand; the second is only weaker.
    G = np.array([[1.0, 0.35], [1.0, 0.12]])
    W = np.linalg.pinv(A.T) @ G.T
    same = isi(fake_fitted(W, channels), labels, data, "eeg")
    weak = isi(fake_fitted(W * [1.0, 0.25], channels), labels, data, "eeg")
    assert weak == pytest.approx(same)
    assert same > 0.4


def test_isi_scales_the_mixing_like_the_z_scored_data():
    gt = ground_truth()
    channels = [f"e{i}" for i in range(6)]
    gain = xr.DataArray(1.0 / np.array([10.0, 2.0, 5.0, 1.0, 20.0, 0.5]), dims="channel",
                        coords={"channel": channels})
    # Z-scored band power is mixed by gain * A_eeg ** 2; a perfect filter inverts that
    # mixing.
    W = np.linalg.pinv(gain.values[:, None] * gt.A_eeg.values ** 2).T
    data = dataclasses.replace(joined(two_sources(100)), ground_truth=gt, eeg_gain=gain)
    labels = PairLabels(pairs={"right_hand": 0, "left_hand": 1}, score=0.0)
    assert isi(fake_fitted(W, channels), labels, data, "eeg") == pytest.approx(0.0, abs=1e-9)
    # Against the unscaled mixing the same perfect filter would look mixed.
    unscaled = dataclasses.replace(data, eeg_gain=None)
    assert isi(fake_fitted(W, channels), labels, unscaled, "eeg") > 0.01


def test_isi_judges_band_power_against_the_squared_field():
    gt = ground_truth()
    field = gt.A_eeg.values
    data = dataclasses.replace(joined(two_sources(100)), ground_truth=gt)
    labels = PairLabels(pairs={"right_hand": 0, "left_hand": 1}, score=0.0)
    channels = [f"e{i}" for i in range(6)]
    # Band power adds over sources with the square of their field, band-passed EEG
    # (mSPoC) with the field itself, so each feature's perfect filter inverts its own.
    band_power = fake_fitted(np.linalg.pinv(field ** 2).T, channels)
    band_passed = fake_fitted(np.linalg.pinv(field).T, channels, method="mSPoC")
    assert isi(band_power, labels, data, "eeg") == pytest.approx(0.0, abs=1e-9)
    assert isi(band_passed, labels, data, "eeg") == pytest.approx(0.0, abs=1e-9)
    # A band power filter that inverts the field itself does not separate the sources.
    inverts_field = fake_fitted(np.linalg.pinv(field).T, channels)
    assert isi(inverts_field, labels, data, "eeg") > 0.05


def test_eeg_mixing_needs_the_eeg_feature():
    with pytest.raises(ValueError, match="eeg_feature"):
        ground_truth_mixing(ground_truth(), "eeg")


def test_score_labels_on_train_and_scores_the_same_pairs_on_test():
    gt = ground_truth()
    train = dataclasses.replace(joined(two_sources(3000, seed=0)), ground_truth=gt)
    test = dataclasses.replace(joined(two_sources(3000, seed=5), seed=6), ground_truth=gt)
    fitted = fit_model("CCA", {}, train, n_components=None)
    scores, recovery = score(fitted, train, test, AssignmentConfig(), HANDS)

    assert sorted(scores.labels.pairs.values()) == [0, 1]
    for source in ("right_hand", "left_hand"):
        assert scores.r_eeg[source] > 0.9 and scores.r_fnirs[source] > 0.9
        assert scores.best_pairs[source][0] == scores.labels.pairs[source]
    assert scores.isi_eeg < 0.2 and scores.isi_fnirs < 0.2
    assert all(r > 0.6 for r in scores.canonical_r.values())
    # The recovery holds the test components of every pair next to the test truth.
    assert recovery.u.sizes["component"] == recovery.r_test_eeg.sizes["component"] == 5
    np.testing.assert_array_equal(recovery.s_eeg.values, test.s_eeg.values)
    pairs = scores.labels.pairs
    assert abs(float(recovery.r_test_eeg.sel(component=pairs["right_hand"], source="right_hand"))) \
        == pytest.approx(scores.r_eeg["right_hand"])
    # Leakage: the right hand's ground truth against the pair assigned to the left hand.
    assert abs(float(recovery.r_test_eeg.sel(component=pairs["left_hand"], source="right_hand"))) \
        == pytest.approx(scores.leak_eeg["right_hand"])
    assert scores.leak_fnirs["left_hand"] < 0.2
    # The patterns of the labelled pairs follow the ground-truth mixing.
    for source in ("right_hand", "left_hand"):
        assert abs(scores.pattern_r_eeg[source]) > 0.9 and abs(scores.pattern_r_fnirs[source]) > 0.9
    # They come from the test data covariance, not from training.
    expected = haufe_pattern(spatial_filter(fitted.model, "fnirs"), test.fnirs)
    xr.testing.assert_allclose(recovery.pattern_fnirs, expected)
    on_train = haufe_pattern(spatial_filter(fitted.model, "fnirs"), train.fnirs)
    assert not np.allclose(recovery.pattern_fnirs.values, on_train.values)


def test_score_without_ground_truth_labels_the_pairs_and_keeps_their_canonical_r():
    without_truth = lambda data: dataclasses.replace(data, s_eeg=None, s_fnirs=None)
    train = joined(two_sources(3000, seed=0))
    test = joined(two_sources(3000, seed=5), seed=6)
    fitted = fit_model("CCA", {}, train, n_components=None)
    with_truth, _ = score(fitted, train, test, motor_channel_config(), HANDS)
    scores, recovery = score(fitted, without_truth(train), without_truth(test),
                             motor_channel_config(), HANDS)

    assert scores.labels == with_truth.labels
    assert scores.canonical_r == with_truth.canonical_r
    assert all(r > 0.6 for r in scores.canonical_r.values())
    for field in ("r_eeg", "r_fnirs", "leak_eeg", "leak_fnirs", "pattern_r_eeg", "pattern_r_fnirs"):
        assert set(getattr(scores, field)) == set(HANDS)
        assert np.isnan(list(getattr(scores, field).values())).all(), field
    assert scores.best_pairs == {}
    assert np.isnan(scores.isi_eeg) and np.isnan(scores.isi_fnirs)
    # The recovery keeps the components and patterns, and no ground truth.
    assert recovery.s_eeg is None and recovery.r_train_eeg is None and recovery.r_test_fnirs is None
    assert recovery.u.sizes["component"] == recovery.pattern_eeg.sizes["component"] == 5


def fold(r_eeg, r_fnirs, isi=(0.2, 0.4), canonical=(0.7, 0.5)):
    return types.SimpleNamespace(r_eeg=dict(zip(("right_hand", "left_hand"), r_eeg)),
                                 r_fnirs=dict(zip(("right_hand", "left_hand"), r_fnirs)),
                                 isi_eeg=isi[0], isi_fnirs=isi[1],
                                 canonical_r=dict(zip(("right_hand", "left_hand"), canonical)))


def test_a_fold_value_is_the_mean_of_the_metrics_quantities():
    scores = fold((0.8, -0.6), (0.4, 0.2))
    assert fold_value("joint_avg", scores) == pytest.approx(0.5)
    assert fold_value("isi_avg", scores) == pytest.approx(0.7)
    assert fold_value("canonical_correlation_avg", scores) == pytest.approx(0.6)
    assert np.isnan(fold_value("joint_avg", None))


def test_fold_mean_averages_each_quantity_over_folds_first():
    a, b = fold((0.8, 0.2), (0.4, 0.3)), fold((0.6, 0.4), (0.9, 0.1))
    z = lambda *r: np.tanh(np.mean(np.arctanh(r)))
    expected = np.mean([z(0.8, 0.6), z(0.2, 0.4), z(0.4, 0.9), z(0.3, 0.1)])
    assert fold_mean("joint_avg", [a, b]) == pytest.approx(expected)
    # A combination is only compared when it was judged on every fold.
    assert np.isnan(fold_mean("joint_avg", [a, b, None]))
    assert np.isnan(fold_mean("joint_avg", [a, fold((np.nan, 0.4), (0.9, 0.1))]))
    one_source = fold((0.8, 0.2), (0.4, 0.3), isi=(np.nan, np.nan))
    assert np.isnan(fold_mean("isi_avg", [one_source, one_source]))


def test_canonical_correlation_counts_a_reversed_pair_against_it():
    agree, reversed_ = fold((0, 0), (0, 0), canonical=(0.6, 0.6)), \
        fold((0, 0), (0, 0), canonical=(-0.6, 0.6))
    assert fold_mean("canonical_correlation_avg", [agree]) == pytest.approx(0.6)
    assert fold_mean("canonical_correlation_avg", [reversed_]) == pytest.approx(0.0)


@pytest.mark.parametrize("method, key", [("tCCA", "shift_source"),
                                         ("StructuredSparseTCCA", "temporal_weight"),
                                         ("SparseCCA", "l1reg")])
def test_check_config_rejects_a_grid_key_that_is_no_hyperparameter(method, key):
    cfg = unmixing(methods=(method,), grid_overrides={method: {key: [0.0]}})
    with pytest.raises(ValueError, match=f"{method} has no hyperparameter '{key}'"):
        check_config(cfg, "bandpower")


def test_check_config_rejects_an_unknown_metric():
    with pytest.raises(ValueError, match="Unknown metric"):
        check_config(unmixing(metrics=("joint_avg_sA",)), "bandpower")


# -------------------------------------------------------------------------
# Selection and unmix
# -------------------------------------------------------------------------
def synthetic_runs(groups=("a", "b", "c"), n: int = 1500) -> list[PreprocessedRecording]:
    """Two runs per group, each with its own sources and noise, the same mixing."""
    gt = ground_truth()
    out = []
    for g, group in enumerate(groups):
        for r in range(2):
            seed = 10 * g + r
            j = joined(two_sources(n, seed=seed), seed=seed + 500)
            # Z-scored per channel, as preprocessing does, keeping the stds.
            z = lambda da: (da - da.mean("time")) / da.std("time")
            out.append(PreprocessedRecording(
                group=group, run=f"run_00{r + 1}", eeg=z(j.eeg), fnirs=z(j.fnirs),
                s_eeg=j.s_eeg, s_fnirs=j.s_fnirs, geo3d=xr.DataArray(), eeg_geo3d=xr.DataArray(),
                fnirs_optodes={}, stim=stim(2.0, 9.5, 17.0), window=(0.0, n / RATE), ground_truth=gt,
                eeg_std=j.eeg.std("time"), fnirs_std=j.fnirs.std("time")))
    return out


SMALL = dataclasses.replace(
    CFG.unmixing, methods=("CCA", "RidgeCCA"), metrics=("joint_avg", "isi_avg"),
    grid_overrides={"RidgeCCA": {"l2_reg": {"eeg": [0.01, 1.0], "fnirs": [0.01]}}}, n_jobs=1)


def test_select_and_score_selects_per_metric_and_scores_on_the_test_group():
    selections, landscape, combinations = select_and_score(synthetic_runs(), "b", "RidgeCCA",
                                                           SMALL, HANDS)
    combos = param_grid("RidgeCCA", SMALL)
    assert len(landscape) == 2 * len(combos)                  # 2 dev groups, one fold each
    assert {p.val_group for p in landscape} == {"a", "c"}
    assert {s.metric for s in selections} == {"joint_avg", "isi_avg"}
    for s in selections:
        assert s.test_group == "b"
        assert s.params in param_grid("RidgeCCA", SMALL)
        assert all(r > 0.9 for r in s.scores.r_eeg.values())
        assert all(r > 0.9 for r in s.scores.r_fnirs.values())
        # The selected combination is the best fold mean of its metric.
        best = max(combinations, key=lambda c: c.means[s.metric])
        assert str(best.params) == str(s.params) and best.means[s.metric] == s.inner_score
    assert len(combinations) == len(combos)
    for point in landscape:
        assert point.metrics["joint_avg"] == fold_value("joint_avg", point.scores)


def test_kept_folds_are_the_fold_fits_of_the_selection():
    selections, landscape, _ = select_and_score(synthetic_runs(), "b", "RidgeCCA", SMALL, HANDS,
                                                keep_folds=True)
    for selection in selections:
        assert [f.val_group for f in selection.folds] == ["a", "c"]
        for fold in selection.folds:
            point, = [p for p in landscape if p.val_group == fold.val_group
                      and str(p.params) == str(selection.params)]
            assert fold.scores == point.scores
            assert fold.recovery.runs == (f"{fold.val_group}/run_001", f"{fold.val_group}/run_002")
    assert select_and_score(synthetic_runs(), "b", "CCA", SMALL, HANDS)[0][0].folds == ()


def test_select_and_score_never_trains_on_the_test_group(monkeypatch):
    import src.unmixing.select as select_module
    trained_on = []
    real_fit = select_module.fit_model

    def spy(method, params, data, *args, **kwargs):
        trained_on.append(data.runs)
        return real_fit(method, params, data, *args, **kwargs)

    monkeypatch.setattr(select_module, "fit_model", spy)
    select_and_score(synthetic_runs(), "b", "CCA", SMALL, HANDS)
    assert trained_on and all(not run.startswith("b/") for runs in trained_on for run in runs)


def test_select_and_score_gives_the_same_result_for_any_n_jobs():
    one, _, _ = select_and_score(synthetic_runs(), "a", "RidgeCCA", SMALL, HANDS)
    two, _, _ = select_and_score(synthetic_runs(), "a", "RidgeCCA",
                                 dataclasses.replace(SMALL, n_jobs=2), HANDS)
    assert [(s.metric, s.params, s.inner_score) for s in one] == \
           [(s.metric, s.params, s.inner_score) for s in two]


def test_unmix_reuses_the_non_nested_result_in_nested_evaluation():
    cfg = dataclasses.replace(SMALL, methods=("CCA",), grid_overrides={}, nested=True)
    result = unmix(synthetic_runs(), cfg, "c", HANDS)
    assert {s.test_group for s in result.selections} == {"c"}
    assert set(result.nested) == {"a", "b", "c"}
    assert result.nested["c"] is result.selections
    assert {s.test_group for s in result.nested["a"]} == {"a"}
    assert {p.test_group for p in result.landscape} == {"a", "b", "c"}


def runs_without_truth() -> list[PreprocessedRecording]:
    """`synthetic_runs` as real data has them: without ground truth."""
    return [dataclasses.replace(rec, s_eeg=None, s_fnirs=None, ground_truth=None)
            for rec in synthetic_runs()]


NO_TRUTH = dataclasses.replace(SMALL, methods=("CCA",), metrics=("canonical_correlation_avg",),
                               grid_overrides={}, assignment=motor_channel_config())


def test_unmix_without_ground_truth_assigns_the_pairs_the_ground_truth_would():
    result = unmix(runs_without_truth(), NO_TRUTH, "b", HANDS)
    with_truth = unmix(synthetic_runs(), dataclasses.replace(NO_TRUTH, assignment=AssignmentConfig()),
                       "b", HANDS)
    selection, = result.selections
    assert selection.scores.labels.pairs == with_truth.selections[0].scores.labels.pairs
    assert all(r > 0.6 for r in selection.scores.canonical_r.values())
    assert np.isnan(list(selection.scores.r_eeg.values())).all()
    assert selection.recovery.s_eeg is None
    assert [f.val_group for f in selection.folds] == ["a", "c"]


@pytest.mark.parametrize("changes, message", [
    ({"metrics": ("joint_avg",)}, r"\['joint_avg'\]"),
    ({"assignment": AssignmentConfig()}, "assignment ground_truth"),
])
def test_unmix_without_ground_truth_refuses_what_needs_it(changes, message):
    with pytest.raises(ValueError, match=message):
        unmix(runs_without_truth(), dataclasses.replace(NO_TRUTH, **changes), "b", HANDS)


def test_unmix_refuses_a_hand_the_ground_truth_has_no_source_for():
    with pytest.raises(ValueError, match=r"no source for \['both_hands'\]"):
        unmix(synthetic_runs(), NO_TRUTH, "b", ["right_hand", "both_hands"])


def test_unmix_refuses_runs_with_and_without_ground_truth():
    runs = synthetic_runs()[:3] + runs_without_truth()[3:]
    with pytest.raises(ValueError, match="Some runs have ground truth"):
        unmix(runs, NO_TRUTH, "b", HANDS)


BUCCINO_CLASSES = {"right": ("right", "right_arm"), "left": ("left", "left_arm")}


def with_trials(*trial_types: str) -> PreprocessedRecording:
    return dataclasses.replace(recording("a", "run_001"), stim=pd.DataFrame(
        {"onset": np.arange(len(trial_types), dtype=float), "duration": 1.0, "value": 1.0,
         "trial_type": list(trial_types)}))


def test_find_hands_takes_every_side_with_trials_right_first():
    runs = [with_trials("rest", "left_arm"), with_trials("rest", "right")]
    assert find_hands(runs, BUCCINO_CLASSES, "both") == ["right_hand", "left_hand"]


def test_find_hands_counts_a_side_by_any_of_its_trial_types():
    assert find_hands([with_trials("rest", "right_arm")], BUCCINO_CLASSES, "both") == ["right_hand"]


def test_find_hands_keeps_the_experiments_hand():
    runs = [with_trials("right", "left")]
    assert find_hands(runs, BUCCINO_CLASSES, "left") == ["left_hand"]


def test_find_hands_refuses_runs_without_a_trial_of_any_side():
    with pytest.raises(ValueError, match="No trial of the sides"):
        find_hands([with_trials("rest", "rest")], BUCCINO_CLASSES, "both")


def test_a_labelled_pair_without_enough_samples_stops_the_run(monkeypatch):
    import src.unmixing.scores as scores_module
    real = scores_module.components

    def broken(fitted, data):
        u, u_paired, v = real(fitted, data)
        return u.where(u.time < u.time[5]), u_paired, v          # only 5 EEG time points left
    monkeypatch.setattr(scores_module, "components", broken)
    with pytest.raises(TooFewSamples, match=r"RidgeCCA .*validation .*: pair .* 5 finite"):
        select_and_score(synthetic_runs(), "b", "RidgeCCA", SMALL, HANDS)


def test_nan_filters_raise_and_exclude_the_setting(monkeypatch):
    import src.unmixing.fit as fit_module
    from src.unmixing.fit import DegenerateFit
    real = fit_module.build_model

    class NanAfterFit:
        def __init__(self, model):
            self.model = model

        def __getattr__(self, name):
            return getattr(self.model, name)

        def fit(self, *args, **kwargs):
            self.model.fit(*args, **kwargs)
            if self.model.l2_reg[0] == 1.0:                     # one setting goes NaN
                self.model.Wx = self.model.Wx * np.nan

    monkeypatch.setattr(fit_module, "build_model",
                        lambda *a, **k: NanAfterFit(real(*a, **k)))
    with pytest.raises(DegenerateFit, match="filters of .* pairs are NaN"):
        fit_model("RidgeCCA", {"l2_reg": [1.0, 0.01]}, joined(two_sources(500)), None)
    selections, landscape, combinations = select_and_score(synthetic_runs(), "b", "RidgeCCA",
                                                           SMALL, HANDS)
    bad = [c for c in combinations if c.params["l2_reg"][0] == 1.0]
    assert bad and all(np.isnan(c.means["joint_avg"]) for c in bad)
    assert all(s.params["l2_reg"][0] != 1.0 for s in selections)
    assert all("DegenerateFit" in p.error for p in landscape if p.params["l2_reg"][0] == 1.0)


def test_mspoc_with_eeg_not_in_blocks_stops_before_any_fit(monkeypatch):
    import src.unmixing.select as select

    def no_fit(*args, **kwargs):
        raise AssertionError("a model was fitted")

    monkeypatch.setattr(select, "fit_model", no_fit)
    # Band power runs: EEG on the fNIRS time points, not in blocks of EEG samples.
    with pytest.raises(ValueError, match="blocks of at least 2 EEG samples"):
        select_and_score(synthetic_runs(), "b", "mSPoC", SMALL, HANDS)


# -------------------------------------------------------------------------
# Laplacians from the recording's geometry
# -------------------------------------------------------------------------
def points(positions: dict[str, list[float]]) -> xr.DataArray:
    """A labeled point cloud in mm."""
    return xr.DataArray(np.array(list(positions.values()), dtype=float), dims=("label", "pos"),
                        coords={"label": list(positions)}).pint.quantify("mm")


# Channels named as in the Buccino dataset: their names say nothing about the optodes.
OPTODES = {"C3A": ("FCC5h", "FCC3h"), "C3M": ("CCP3h", "FCC3h"), "C4L": ("FCC6h", "CCP6h")}
GEO3D = points({"FCC5h": [-60.0, 10.0, 60.0], "FCC3h": [-30.0, 10.0, 80.0],
                "CCP3h": [-30.0, -20.0, 80.0], "FCC6h": [60.0, 10.0, 60.0],
                "CCP6h": [60.0, -20.0, 60.0]})


def test_fnirs_positions_are_the_midpoints_of_each_channels_optodes():
    positions = fnirs_positions(["C3A", "C4L"], GEO3D, OPTODES)
    np.testing.assert_allclose(positions, [[-45.0, 10.0, 70.0], [60.0, -5.0, 60.0]])


def test_shared_optode_laplacian_links_channels_that_share_an_optode():
    eeg_geo3d = points({"C3": [-50.0, 0.0, 70.0], "C4": [50.0, 0.0, 70.0]})
    spec = {"eeg": {"method": "knn", "k": 1},
            "fnirs": {"method": "shared_optode", "normed": False}}
    _, Ly, _ = build_laplacians(spec, ["C3", "C4"], list(OPTODES), GEO3D, eeg_geo3d, OPTODES)
    # C3A and C3M share the detector FCC3h; C4L shares nothing.
    np.testing.assert_allclose(Ly, [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
