"""
Check that a preprocessed run is cut and aligned in time.

`check_alignment` stops the run on a clear problem; `alignment_report` gives one
summary line per run for the log.
"""
import numpy as np
import pandas as pd
import xarray as xr

from src.config import AlignmentCheckConfig
from src.types import PreprocessedRecording


def check_alignment(recording: PreprocessedRecording, cfg: AlignmentCheckConfig):
    """
    Raise if the run is badly cut or aligned:

    - EEG and fNIRS do not share their time span: the same time points for band
      power; for band-passed EEG, the same start to within one EEG sample and exactly
      k EEG samples per fNIRS sample, k = EEG rate / fNIRS rate.
    - The kept data covers less than `min_window_coverage` of the stimulus window.

    Parameters
    ----------
    recording : PreprocessedRecording
    cfg : AlignmentCheckConfig

    Raises
    ------
    ValueError
        Naming the run and what failed.
    """
    name = f"{recording.group}/{recording.run}"
    eeg, fnirs = recording.eeg, recording.fnirs

    if not _same_axis(recording):
        eeg_dt = float(np.median(np.diff(eeg.time.values)))
        fnirs_dt = float(np.median(np.diff(fnirs.time.values)))
        k = round(fnirs_dt / eeg_dt)
        if (k < 2 or abs(float(eeg.time[0]) - float(fnirs.time[0])) > eeg_dt
                or eeg.sizes["time"] != k * fnirs.sizes["time"]):
            raise ValueError(f"{name}: EEG ({_span(eeg)}) and fNIRS ({_span(fnirs)}) do not "
                             f"cover the same time span in blocks of {k} EEG samples per "
                             f"fNIRS sample.")

    start, end = recording.window
    coverage = (float(fnirs.time[-1]) - float(fnirs.time[0])) / (end - start)
    if coverage < cfg.min_window_coverage:
        raise ValueError(f"{name}: the kept data ({_span(fnirs)}) covers {coverage:.0%} of the "
                         f"stimulus window {start:.1f}-{end:.1f} s, less than "
                         f"{cfg.min_window_coverage:.0%}. The recording may end early or "
                         f"contain non-finite samples.")


def alignment_report(recording: PreprocessedRecording) -> str:
    """
    One line describing how a run was cut and aligned: the stimulus window and the
    time span and sample count of the EEG and fNIRS.
    """
    start, end = recording.window
    parts = [f"{recording.group}/{recording.run}: window {start:.1f}-{end:.1f} s",
             f"EEG {_span(recording.eeg)}", f"fNIRS {_span(recording.fnirs)}"]
    return " · ".join(parts)


def alignment_table(recordings: list[PreprocessedRecording]) -> pd.DataFrame:
    """
    One row per run: the stimulus window and the time span and sample count of the
    EEG and fNIRS.
    """
    rows = []
    for rec in recordings:
        row = {"group": rec.group, "run": rec.run,
               "window_start": rec.window[0], "window_end": rec.window[1]}
        for name, da in (("eeg", rec.eeg), ("fnirs", rec.fnirs)):
            row[f"{name}_start"] = float(da.time[0])
            row[f"{name}_end"] = float(da.time[-1])
            row[f"{name}_samples"] = da.sizes["time"]
        rows.append(row)
    return pd.DataFrame(rows)


def _same_axis(recording: PreprocessedRecording) -> bool:
    return np.array_equal(recording.eeg.time.values, recording.fnirs.time.values)


def _span(da: xr.DataArray) -> str:
    return f"{float(da.time[0]):.1f}-{float(da.time[-1]):.1f} s ({da.sizes['time']})"
