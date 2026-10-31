"""
Read a run folder written by `save_run` back into a RunResult, for plots and
analysis without rerunning anything.
"""
import json
import re
import warnings
from pathlib import Path

import mne
import pandas as pd
import xarray as xr
import yaml
from cedalion.dataclasses import PointType

from src.config import ExperimentConfig, experiment_from_snapshot, subject_from_snapshot
from src.config.load import PROJECT_ROOT
from src.positions import electrode_positions
from src.results.inputs import read_inputs
from src.results.save import GROUND_TRUTH_VARIABLES, LAYOUT_VERSION
from src.results.tables import (classification_from_row, params_from_json, params_to_json,
                                scores_from_row)
from src.types import (JOINED_STIM_COLUMNS, BadChannels, ClassificationResult,
                       CombinationScore, FoldRecovery, FoldScore, Geometry,
                       PreprocessedRecording, Recovery, RunResult, Selection, UnmixingResult)
from src.unmixing import METRICS, fitted_from_dataset

# The EEG template every run used before the montage was set in the subject YAML.
# Only results saved back then fall back to it.
LEGACY_EEG_MONTAGE = "standard_1020"
# Runs saved before the optodes of each fNIRS channel were kept are all synthetic, with
# channels named after their optodes, e.g. "S10D12".
_LEGACY_FNIRS_CHANNEL = re.compile(r"(S\d+)(D\d+)")


def load_run(run_dir: str | Path) -> RunResult:
    """
    Parameters
    ----------
    run_dir : str | Path
        A run folder written by `save_run` (the one holding results/).

    Returns
    -------
    RunResult
        As the run returned it, except `inputs`, which is None: they are large and
        live elsewhere; `load_inputs` reads them. Each selected model is rebuilt from
        its saved state and can be applied to new data with `transform`.
    """
    data_dir = Path(run_dir) / "results"
    info = json.loads((data_dir / "run.json").read_text())
    if info["layout_version"] != LAYOUT_VERSION:
        raise ValueError(f"{data_dir} has layout version {info['layout_version']}; this code "
                         f"reads version {LAYOUT_VERSION}.")
    cfg = experiment_from_snapshot(yaml.safe_load((data_dir / "experiment.yaml").read_text()))
    subject = subject_from_snapshot(_with_montage(
        yaml.safe_load((data_dir / "subject.yaml").read_text()), data_dir))
    bad = json.loads((data_dir / "bad_channels.json").read_text())
    bad_channels = BadChannels(
        eeg=tuple(bad["eeg"]), fnirs=tuple(bad["fnirs"]),
        eeg_detail={p: {k: tuple(v) for k, v in d.items()} for p, d in bad["eeg_detail"].items()},
        fnirs_detail={k: tuple(v) for k, v in bad["fnirs_detail"].items()})

    landscape = [FoldScore(method=row["method"], test_group=row["test_group"],
                           val_group=row["val_group"], params=params_from_json(row["params"]),
                           scores=scores_from_row(row),
                           metrics={m: float(row[m]) for m in METRICS},
                           error=None if pd.isna(row["error"]) else row["error"])
                 for row in _read_csv(data_dir / "landscape.csv")]
    combinations = [CombinationScore(method=row["method"], test_group=row["test_group"],
                                     params=params_from_json(row["params"]),
                                     means={m: float(row[m]) for m in METRICS})
                    for row in _read_csv(data_dir / "combinations.csv")]
    fold_scores = {(p.method, p.test_group, p.val_group, params_to_json(p.params)): p.scores
                   for p in landscape}

    models, recoveries, folds = {}, {}, {}
    selections, nested = [], {}
    for row in _read_csv(data_dir / "scores.csv"):
        name = row["model"]
        if name not in models:
            models[name] = fitted_from_dataset(xr.load_dataset(data_dir / "models" / f"{name}.nc",
                                                               engine="scipy"))
            recoveries[name] = _recovery(xr.load_dataset(data_dir / "recovery" / f"{name}.nc",
                                                         engine="scipy"))
            folds[name] = tuple(
                FoldRecovery(val_group=val_group,
                             scores=fold_scores[(row["method"], row["test_group"], val_group,
                                                 params_to_json(params_from_json(row["params"])))],
                             recovery=_recovery(xr.load_dataset(path, engine="scipy")))
                for path in sorted((data_dir / "recovery").glob(f"{name}_fold_*.nc"))
                for val_group in [path.stem.removeprefix(f"{name}_fold_")])
        selection = Selection(method=row["method"], metric=row["metric"],
                              params=params_from_json(row["params"]),
                              inner_score=float(row["inner_score"]), fitted=models[name],
                              scores=scores_from_row(row), recovery=recoveries[name],
                              test_group=row["test_group"], folds=folds[name])
        if row["evaluation"] == "non_nested":
            selections.append(selection)
        else:
            nested.setdefault(row["test_group"], []).append(selection)

    return RunResult(
        cfg=cfg, subject=subject, bad_channels=bad_channels,
        alignment=pd.read_csv(data_dir / "alignment.csv", float_precision="round_trip"),
        geometry=_geometry(xr.load_dataset(data_dir / "geometry.nc", engine="scipy"),
                           dropped_fnirs=bad_channels.fnirs),
        unmixing=UnmixingResult(test_group=info["test_group"], selections=selections,
                                nested=nested or None, landscape=landscape,
                                combinations=combinations),
        inputs=None, started=info["started"], finished=info["finished"],
        classification=_classification(data_dir / "classification.csv", cfg, info["test_group"]))


def _classification(path: Path, cfg: ExperimentConfig,
                    test_group: str) -> ClassificationResult | None:
    """None for a run without classification, or one saved before it existed."""
    if cfg.classification is None or not path.is_file():
        return None
    scores, nested = [], {}
    for row in _read_csv(path):
        score = classification_from_row(row, cfg.classification.classes)
        if row["evaluation"] == "non_nested":
            scores.append(score)
        else:
            nested.setdefault(row["test_group"], []).append(score)
    return ClassificationResult(test_group=test_group, scores=scores, nested=nested or None)


def load_inputs(run_dir: str | Path) -> list[PreprocessedRecording]:
    """
    The preprocessed runs a run's models were fit on, saved when the experiment set
    save_inputs.

    Parameters
    ----------
    run_dir : str | Path
        A folder written by `save_run`.

    Returns
    -------
    list[PreprocessedRecording]
        Read back from float32, so close to but not bit-identical with what the
        models saw. `ground_truth` is None; the mixing is in the run's geometry.

    Raises
    ------
    FileNotFoundError
        If the run saved no inputs, or they are not on this machine.
    """
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "results" / "run.json").read_text())
    if info["inputs"] is None:
        raise FileNotFoundError(f"{run_dir} saved no inputs (save_inputs: false).")
    folder = PROJECT_ROOT / info["inputs"]
    if not folder.is_dir():
        raise FileNotFoundError(f"The inputs of {run_dir} are not at {folder}.")
    geometry = _geometry(xr.load_dataset(run_dir / "results" / "geometry.nc", engine="scipy"))
    return read_inputs(folder, geometry.geo3d, geometry.eeg_geo3d, geometry.fnirs_optodes)


def _points(da: xr.DataArray, units: str | None) -> xr.DataArray:
    """A labeled point cloud as `_storable_points` saved it, with its types and units."""
    if "type" in da.coords:
        da = da.assign_coords(type=("label", [PointType[t] for t in da.type.values]))
    da = da.drop_vars([c for c in da.coords if c not in ("label", "type")])
    return da.pint.quantify(units) if units else da


def _with_montage(subject: dict, data_dir: Path) -> dict:
    """
    A subject snapshot with a montage. Runs saved before the montage was part of the
    subject have none; they all used the legacy EEG template and the optode positions
    of the recording.
    """
    if "montage" in subject:
        return subject
    warnings.warn(f"{data_dir}: the subject snapshot has no montage (saved before it was "
                  f"part of the subject); assuming the {LEGACY_EEG_MONTAGE} template and "
                  f"the recording's optode positions.")
    return {**subject, "montage": {"eeg": {"standard": LEGACY_EEG_MONTAGE},
                                   "fnirs": {"recording": True}}}


def _legacy_electrode_positions() -> xr.DataArray:
    """
    Every electrode of the legacy template, in MNE's head frame, as loading would
    have placed them.
    """
    montage = mne.channels.make_standard_montage(LEGACY_EEG_MONTAGE)
    info = mne.create_info(montage.ch_names, sfreq=1.0, ch_types="eeg")
    info.set_montage(montage)
    return electrode_positions(info.get_montage(), montage.ch_names)


def _read_csv(path: Path) -> list[dict]:
    """Rows of a CSV, floats exactly as written; none for an empty file."""
    try:
        return pd.read_csv(path, float_precision="round_trip").to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def _recovery(ds: xr.Dataset) -> Recovery:
    back = lambda da: da.rename(time_components="time")
    gain = lambda name, dim: ds[name].rename({dim: "channel"}) if name in ds else None
    stim = pd.DataFrame(json.loads(ds.attrs["stim"]), columns=JOINED_STIM_COLUMNS)
    truth = {name: ds[name] if name in ds else None for name in GROUND_TRUTH_VARIABLES}
    return Recovery(u=back(ds["u"]), v=back(ds["v"]),
                    run_starts=tuple(json.loads(ds.attrs["run_starts"])),
                    runs=tuple(json.loads(ds.attrs["runs"])), stim=stim,
                    pattern_eeg=ds["pattern_eeg"].rename(channel_eeg="channel"),
                    pattern_fnirs=ds["pattern_fnirs"].rename(channel_fnirs="channel"),
                    gain_eeg=gain("gain_eeg", "channel_eeg"),
                    gain_fnirs=gain("gain_fnirs", "channel_fnirs"), **truth)


def _geometry(ds: xr.Dataset, dropped_fnirs: tuple[str, ...] = ()) -> Geometry:
    """
    The run's geometry. `dropped_fnirs` are the fNIRS channels pruning dropped, whose
    optodes a run saved before the optodes were kept can only take from their names.
    """
    geo3d = _points(ds["geo3d"], ds.attrs.get("geo3d_units"))
    if "eeg_geo3d" in ds:
        eeg_geo3d = ds["eeg_geo3d"]
        eeg_geo3d = _points(eeg_geo3d.rename({f"{name}_eeg": name for name in ("label", "type")
                                              if f"{name}_eeg" in eeg_geo3d.coords}),
                            ds.attrs.get("eeg_geo3d_units"))
    else:
        warnings.warn(f"The run's geometry holds no EEG electrode positions (saved before "
                      f"they were kept); using the {LEGACY_EEG_MONTAGE} template instead.")
        eeg_geo3d = _legacy_electrode_positions()
    has_gt = "A_eeg" in ds
    pick = lambda name, dim: ds[name].rename({dim: "channel"}).drop_vars(
        [c for c in ds[name].coords if c not in ds[name].dims])
    return Geometry(eeg_channels=tuple(str(c) for c in ds.eeg_channel.values),
                    fnirs_channels=tuple(str(c) for c in ds.fnirs_channel.values),
                    geo3d=geo3d, eeg_geo3d=eeg_geo3d,
                    fnirs_optodes=_fnirs_optodes(ds, dropped_fnirs),
                    A_eeg=pick("A_eeg", "channel_A_eeg") if has_gt else None,
                    A_fnirs=pick("A_fnirs", "channel_A_fnirs") if has_gt else None)


def _fnirs_optodes(ds: xr.Dataset, dropped_fnirs: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    """
    The optodes of every fNIRS channel as saved, or, for a run saved before they were
    kept, read from the channel names (kept and dropped).
    """
    if "optode_channel" in ds.coords:
        return {str(ch): (str(src), str(det)) for ch, src, det in
                zip(ds.optode_channel.values, ds.optode_source.values,
                    ds.optode_detector.values)}
    warnings.warn("The run's geometry holds no fNIRS channel optodes (saved before they "
                  "were kept); reading them from the S<n>D<m> channel names instead.")
    names = [*(str(c) for c in ds.fnirs_channel.values), *dropped_fnirs]
    return {name: _LEGACY_FNIRS_CHANNEL.fullmatch(name).groups() for name in names}
