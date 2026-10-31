"""
Write a run's result to its own folder, as data only: tables, arrays and metadata
that any plot can be drawn from later without rerunning anything.
"""
import dataclasses
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from src.config import experiment_snapshot, subject_snapshot
from src.config.load import PROJECT_ROOT
from src.results.inputs import PREPROCESSED_DIR, relative_to_project, save_inputs
from src.results.tables import classification_row, params_to_json, score_columns
from src.types import Recovery, RunResult, Selection
from src.unmixing import fitted_to_dataset

RESULTS_DIR = PROJECT_ROOT / "results"

# Changes when the folder layout changes, so `load_run` can tell what it reads.
LAYOUT_VERSION = 2

# 17 significant digits: every float reads back exactly.
FLOAT_FORMAT = "%.17g"

# The fields of a Recovery that only data with ground truth has.
GROUND_TRUTH_VARIABLES = ("s_eeg", "s_fnirs", "r_train_eeg", "r_train_fnirs", "r_test_eeg",
                          "r_test_fnirs")


def save_run(result: RunResult, base_dir: Path = RESULTS_DIR,
             inputs_dir: Path = PREPROCESSED_DIR) -> Path:
    """
    Write `result` to results/runs/<experiment>/<dataset>/<started>/results/ and add
    a line for it to results/runs/index.csv. <dataset> is the subject config's `name`.
    The run folder's other subfolder, figures/, is for the figures drawn from it.

    With save_inputs in the experiment, the preprocessed runs (`result.inputs`) go
    to data/preprocessed/<dataset>/<started>/ (see `save_inputs`), and run.json
    points to them; they are data, not results.

    Files (in results/)
    -------------------
    experiment.yaml, subject.yaml
        The configs, every value resolved.
    run.json
        Layout version, times, code version (git commit, uncommitted changes),
        package versions, test group, runs, and the inputs folder (or null).
    bad_channels.json
        Pruning: the bad channels and the criterion that flagged each.
    alignment.csv
        One row per run: how it was cut and aligned.
    geometry.nc
        Channels, optode positions, ground-truth mixing.
    scores.csv
        One row per selected model: non-nested, and nested if it ran. Per source the
        recovery |r|, the signed canonical r, and the pattern |r| (a diagnostic, never
        used for selection).
    landscape.csv
        Every hyperparameter combination in every validation fold, with each
        metric's value. The pattern |r| columns are NaN except for the selected
        combinations.
    combinations.csv
        Every combination's mean over the validation folds, per metric: what
        selection compared.
    classification.csv
        With classification only: one row per selected model and feature set,
        non-nested and nested, with the test accuracy, the recall of every class
        and the confusion counts.
    models/<model>.nc
        Each selected model's fitted state and patterns; `load_run` rebuilds a
        model that can be applied again.
    recovery/<model>.nc
        Each selected model's components on its test group, the ground truth, the
        stimuli, the correlation of every pair with every source, and the channel
        gains the scores scaled the ground-truth mixing by.
    recovery/<model>_fold_<val_group>.nc
        The same for its hyperparameters fit in each validation fold, applied to the
        fold's validation group (non-nested selections). The fold's scores are in
        landscape.csv.

    Parameters
    ----------
    result : RunResult
    base_dir : Path
        Where results/ is.
    inputs_dir : Path
        Where data/preprocessed/ is.

    Returns
    -------
    Path
        The run folder (results/ and figures/ are inside it).
    """
    cfg, subject = result.cfg, result.subject
    runs_dir = Path(base_dir) / "runs"
    run_dir = runs_dir / cfg.name / subject.name / result.started.replace(":", "-")
    run_dir.mkdir(parents=True, exist_ok=False)
    data_dir = run_dir / "results"
    (data_dir / "models").mkdir(parents=True)
    (data_dir / "recovery").mkdir()

    info = _run_info(result)
    info["inputs"] = (relative_to_project(save_inputs(result, run_dir, inputs_dir))
                      if cfg.save_inputs else None)
    _write_yaml(data_dir / "experiment.yaml", experiment_snapshot(cfg))
    _write_yaml(data_dir / "subject.yaml", subject_snapshot(subject))
    _write_json(data_dir / "run.json", info)
    _write_json(data_dir / "bad_channels.json", dataclasses.asdict(result.bad_channels))
    result.alignment.to_csv(data_dir / "alignment.csv", index=False, float_format=FLOAT_FORMAT)
    _geometry_dataset(result.geometry).to_netcdf(data_dir / "geometry.nc", engine="scipy")

    unmixing = result.unmixing
    names = {}
    rows = _selection_rows("non_nested", unmixing.selections, names, data_dir)
    for group, selections in (unmixing.nested or {}).items():
        rows += _selection_rows("nested", selections, names, data_dir)
    pd.DataFrame(rows).to_csv(data_dir / "scores.csv", index=False, float_format=FLOAT_FORMAT)
    pd.DataFrame([_landscape_row(point) for point in unmixing.landscape]).to_csv(
        data_dir / "landscape.csv", index=False, float_format=FLOAT_FORMAT)
    pd.DataFrame([_combination_row(c) for c in unmixing.combinations]).to_csv(
        data_dir / "combinations.csv", index=False, float_format=FLOAT_FORMAT)
    if result.classification is not None:
        classification = result.classification
        rows = [classification_row("non_nested", s) for s in classification.scores]
        for scores in (classification.nested or {}).values():
            rows += [classification_row("nested", s) for s in scores]
        pd.DataFrame(rows).to_csv(data_dir / "classification.csv", index=False,
                                  float_format=FLOAT_FORMAT)
    _append_index(runs_dir / "index.csv", run_dir.relative_to(runs_dir), result, info)
    return run_dir


def _append_index(index: Path, folder: Path, result: RunResult, info: dict):
    """
    Add one line for a finished run to the index of all runs, writing the header if
    the index is new. Written last, so a run listed there was saved completely.
    """
    row = pd.DataFrame([{
        "started": result.started, "finished": result.finished,
        "experiment": result.cfg.name, "dataset": result.subject.name,
        "subject": result.subject.subject_id, "hand": result.cfg.hand,
        "test_group": result.unmixing.test_group,
        "methods": " ".join(result.cfg.unmixing.methods),
        "git_commit": info["git_commit"],
        "git_uncommitted_changes": info["git_uncommitted_changes"],
        "folder": folder.as_posix(),
    }])
    row.to_csv(index, mode="a", header=not index.is_file(), index=False)


def _selection_rows(evaluation: str, selections: list[Selection], names: dict,
                    data_dir: Path) -> list[dict]:
    """Rows for scores.csv; writes each model and its recovery once."""
    rows = []
    for selection in selections:
        key = id(selection.fitted)
        if key not in names:
            names[key] = f"{evaluation}_{selection.test_group}_{selection.method}_{len(names)}"
            fitted_to_dataset(selection.fitted).to_netcdf(
                data_dir / "models" / f"{names[key]}.nc", engine="scipy")
            _recovery_dataset(selection.recovery).to_netcdf(
                data_dir / "recovery" / f"{names[key]}.nc", engine="scipy")
            for fold in selection.folds:
                _recovery_dataset(fold.recovery).to_netcdf(
                    data_dir / "recovery" / f"{names[key]}_fold_{fold.val_group}.nc",
                    engine="scipy")
        rows.append({"evaluation": evaluation, "test_group": selection.test_group,
                     "method": selection.method, "metric": selection.metric,
                     "params": params_to_json(selection.params),
                     "inner_score": selection.inner_score, "model": names[key],
                     **score_columns(selection.scores)})
    return rows


def _landscape_row(point) -> dict:
    row = {"method": point.method, "test_group": point.test_group,
           "val_group": point.val_group, "params": params_to_json(point.params),
           "error": point.error, **point.metrics}
    return {**row, **score_columns(point.scores)}


def _combination_row(combination) -> dict:
    return {"method": combination.method, "test_group": combination.test_group,
            "params": params_to_json(combination.params), **combination.means}


def _recovery_dataset(recovery: Recovery) -> xr.Dataset:
    """
    u and v keep their own (possibly shorter) time axis, named time_components; the
    patterns and gains their own channel axes (channel_eeg, channel_fnirs). A gain,
    and the ground truth and its correlations, are left out where they are None. The
    stimuli are an attribute: JSON records, floats written exactly.
    """
    out = lambda da: da.rename(time="time_components")
    variables = {"u": out(recovery.u), "v": out(recovery.v),
                 "pattern_eeg": recovery.pattern_eeg.rename(channel="channel_eeg"),
                 "pattern_fnirs": recovery.pattern_fnirs.rename(channel="channel_fnirs")}
    for name in GROUND_TRUTH_VARIABLES:
        if getattr(recovery, name) is not None:
            variables[name] = getattr(recovery, name)
    for name, gain, dim in (("gain_eeg", recovery.gain_eeg, "channel_eeg"),
                            ("gain_fnirs", recovery.gain_fnirs, "channel_fnirs")):
        if gain is not None:
            variables[name] = gain.rename(channel=dim)
    ds = xr.Dataset(variables)
    ds = ds.drop_vars([c for c in ds.coords if c not in ds.dims])
    ds = ds.assign_coords(component=ds.component.astype(np.int32))
    ds.attrs = {"run_starts": json.dumps(list(recovery.run_starts)),
                "runs": json.dumps(list(recovery.runs)),
                "stim": json.dumps(recovery.stim.to_dict("records"),
                                   default=lambda o: o.item())}
    return ds


def _geometry_dataset(geometry) -> xr.Dataset:
    geo3d, units = _storable_points(geometry.geo3d)
    eeg_geo3d, eeg_units = _storable_points(geometry.eeg_geo3d)
    # The electrodes keep their own label axis, so one file holds both point clouds.
    eeg_geo3d = eeg_geo3d.rename({name: f"{name}_eeg" for name in ("label", "type")
                                  if name in eeg_geo3d.coords})
    variables = {"geo3d": geo3d, "eeg_geo3d": eeg_geo3d}
    if geometry.A_eeg is not None:
        variables["A_eeg"] = geometry.A_eeg.rename(channel="channel_A_eeg")
        variables["A_fnirs"] = geometry.A_fnirs.rename(channel="channel_A_fnirs")
    ds = xr.Dataset(variables)
    ds = ds.drop_vars([c for c in ds.coords
                       if c not in ds.dims and c not in ("type", "type_eeg")])
    ds = ds.assign_coords(eeg_channel=list(geometry.eeg_channels),
                          fnirs_channel=list(geometry.fnirs_channels))
    # Every fNIRS channel's optodes, dropped channels included, on their own axis.
    optodes = geometry.fnirs_optodes
    ds = ds.assign_coords(optode_channel=list(optodes),
                          optode_source=("optode_channel", [s for s, _ in optodes.values()]),
                          optode_detector=("optode_channel", [d for _, d in optodes.values()]))
    ds.attrs = {"geo3d_units": units, "eeg_geo3d_units": eeg_units}
    return ds


def _storable_points(points: xr.DataArray) -> tuple[xr.DataArray, str]:
    """
    A labeled point cloud as netCDF can hold it, without units and with its point
    types as names, and its units ("" if it has none).
    """
    units = str(points.pint.units) if getattr(points.pint, "units", None) is not None else ""
    points = points.pint.dequantify() if units else points
    if "type" in points.coords:
        points = points.assign_coords(type=("label", [t.name for t in points.type.values]))
    return points.drop_attrs(), units


def _run_info(result: RunResult) -> dict:
    import cedalion
    import mne
    import sklearn
    return {
        "layout_version": LAYOUT_VERSION,
        "experiment": result.cfg.name, "dataset": result.subject.name,
        "subject": result.subject.subject_id,
        "test_group": result.unmixing.test_group,
        "started": result.started, "finished": result.finished,
        "runs": [f"{row.group}/{row.run}" for row in result.alignment.itertuples()],
        "git_commit": _git("rev-parse", "HEAD"),
        "git_uncommitted_changes": bool(_git("status", "--porcelain")),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "xarray": xr.__version__, "cedalion": getattr(cedalion, "__version__", "?"),
                     "mne": mne.__version__, "sklearn": sklearn.__version__},
    }


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _write_yaml(path: Path, data: dict):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _write_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
