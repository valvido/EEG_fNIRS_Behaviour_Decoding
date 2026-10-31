"""
The preprocessed runs a run's models were fit on. They are data, not results, so they
go to data/preprocessed/<dataset>/<started>/, one file per run, with a
preprocessed.json naming the run folder they belong to and the configs that made them.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.config import experiment_snapshot
from src.config.load import PROJECT_ROOT
from src.types import STIM_COLUMNS, PreprocessedRecording, RunResult

PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"


def save_inputs(result: RunResult, run_dir: Path, base_dir: Path = PREPROCESSED_DIR) -> Path:
    """
    Write `result.inputs` to <base_dir>/<dataset>/<started>/.

    Files
    -----
    preprocessed.json
        Dataset, subject, when the run started, the run folder these inputs belong
        to, the runs, and the pruning and preprocessing configs.
    <group>__<run>.nc
        One run: EEG feature, HbO and ground-truth sources as float32, the EEG on its
        own time and channel axes (time_eeg, channel_eeg; fNIRS channel_fnirs), the
        channel stds of z-scoring (eeg_std, fnirs_std) in float64, the stimuli and the
        stimulus window as attributes.
        float32 keeps about 7 significant digits: enough to inspect, but values read
        back are not bit-identical to the ones the models were fit on.

    Parameters
    ----------
    result : RunResult
        With `inputs`, from the run in memory.
    run_dir : Path
        The run folder they belong to.
    base_dir : Path

    Returns
    -------
    Path
        The inputs folder.
    """
    folder = (Path(base_dir) / result.subject.name / result.started.replace(":", "-"))
    folder.mkdir(parents=True, exist_ok=False)
    for rec in result.inputs:
        _recording_dataset(rec).to_netcdf(folder / f"{rec.group}__{rec.run}.nc", engine="scipy")
    snapshot = experiment_snapshot(result.cfg)
    info = {"dataset": result.subject.name, "subject": result.subject.subject_id,
            "started": result.started, "run_folder": relative_to_project(run_dir),
            "runs": [f"{rec.group}/{rec.run}" for rec in result.inputs],
            "pruning": snapshot["pruning"], "preprocessing": snapshot["preprocessing"]}
    (folder / "preprocessed.json").write_text(json.dumps(info, indent=2))
    return folder


def read_inputs(folder: Path, geo3d: xr.DataArray, eeg_geo3d: xr.DataArray,
                fnirs_optodes: dict[str, tuple[str, str]]) -> list[PreprocessedRecording]:
    """
    The runs `save_inputs` wrote to `folder`, in the order of preprocessed.json, as
    float64 again (from float32 values). `ground_truth` is None: the mixing matrices
    are in the run's geometry. `geo3d`, `eeg_geo3d` and `fnirs_optodes` are the run's
    optode and electrode positions and the optodes of each fNIRS channel.
    """
    folder = Path(folder)
    info = json.loads((folder / "preprocessed.json").read_text())
    out = []
    for name in info["runs"]:
        group, run = name.split("/")
        ds = xr.load_dataset(folder / f"{group}__{run}.nc", engine="scipy")
        as64 = lambda da: da.astype(np.float64)
        out.append(PreprocessedRecording(
            group=group, run=run,
            eeg=as64(ds["eeg"].rename(time_eeg="time", channel_eeg="channel")),
            fnirs=as64(ds["fnirs"].rename(channel_fnirs="channel")),
            s_eeg=as64(ds["s_eeg"]) if "s_eeg" in ds else None,
            s_fnirs=as64(ds["s_fnirs"]) if "s_fnirs" in ds else None,
            geo3d=geo3d, eeg_geo3d=eeg_geo3d, fnirs_optodes=fnirs_optodes,
            stim=pd.DataFrame(json.loads(ds.attrs["stim"]), columns=STIM_COLUMNS),
            window=tuple(json.loads(ds.attrs["window"])), ground_truth=None,
            eeg_std=ds["eeg_std"].rename(channel_eeg="channel") if "eeg_std" in ds else None,
            fnirs_std=(ds["fnirs_std"].rename(channel_fnirs="channel")
                       if "fnirs_std" in ds else None)))
    return out


def _recording_dataset(rec: PreprocessedRecording) -> xr.Dataset:
    clean = lambda da: da.drop_vars([c for c in da.coords if c not in da.dims]).astype(np.float32)
    # Each modality keeps its own channel (and EEG its own time) dimension, so one file
    # holds both without merging their axes.
    data = {"eeg": clean(rec.eeg).rename(time="time_eeg", channel="channel_eeg"),
            "fnirs": clean(rec.fnirs).rename(channel="channel_fnirs")}
    if rec.s_eeg is not None:
        data |= {"s_eeg": clean(rec.s_eeg), "s_fnirs": clean(rec.s_fnirs)}
    # The stds stay float64: they are few, and the scores use them.
    if rec.eeg_std is not None:
        data["eeg_std"] = rec.eeg_std.rename(channel="channel_eeg")
    if rec.fnirs_std is not None:
        data["fnirs_std"] = rec.fnirs_std.rename(channel="channel_fnirs")
    ds = xr.Dataset(data)
    ds.attrs = {"stim": json.dumps(rec.stim.reindex(columns=STIM_COLUMNS).to_dict("records"),
                                   default=lambda o: o.item()),
                "window": json.dumps([float(t) for t in rec.window])}
    return ds


def relative_to_project(path: Path) -> str:
    """`path` relative to the project when inside it, so the link survives a move."""
    path = Path(path).resolve()
    return str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path)
