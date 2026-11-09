"""
Reader for the hybrid EEG-fNIRS SMR BCI dataset of Buccino et al. (2016), PLoS One
11(1): e0146610. Fifteen subjects, motor execution of right/left hand and arm.

One MATLAB file per subject, holding the whole session in one piece:

    <paths.mat>     S<n>.mat, one cell array `data`:
        data{1}     EEG: rate, channel count, labels, time stamps, signals
                    (time x channel, µV)
        data{2}     fNIRS: rate, channel count, labels, time stamps, raw intensity
                    at the two wavelengths (time x channel each)
        data{3}     events (event x 2): onset (s, on the shared clock) and code

The session holds 5 blocks of 20 trials. The subject's groups name them block_1 to
block_5; the session is split into them after preprocessing. The dataset has no
ground truth, no separate rest recording and no optode positions. Which optodes each
fNIRS channel uses is not in the file either: every channel gets a placeholder source
and detector label of its own, which the subject's montage replaces
(configs/montages/buccino_fnirs.tsv, reconstructed from Fig 1 of the paper).
"""
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import xarray as xr

from src.config import SubjectConfig
from src.loading.arrays import (build_eeg, build_fnirs_amplitude, build_geo3d,
                                check_regular_time)
from src.loading.files import read_mat
from src.loading.rest import mark_rest_period
from src.types import RawRecording

REQUIRED_PATHS = ("mat",)

# Event codes of data{3}, as the dataset's DataOrganization.txt lists them.
BLOCK_START_CODE = 10
REST_TRIAL_TYPE = "rest"
TRIAL_TYPES = {1: REST_TRIAL_TYPE, 3: "right_arm", 4: "left_arm", 7: "right", 8: "left"}

# The paper gives 760 and 850 nm; the file does not say which of wl1, wl2 is which.
# This order is assumed and is to be checked on the HbO/HbR response to the task.
WAVELENGTHS_NM = (760.0, 850.0)

_GROUP = re.compile(r"block_([1-9]\d*)")


def check(subject: SubjectConfig):
    """
    Check that every group names a block, as block_<k> with k counted from 1.
    Whether the session has that many blocks is checked once it is read.
    """
    bad = [group for group in subject.groups if not _GROUP.fullmatch(group)]
    if bad:
        raise ValueError(f"Subject '{subject.subject_id}' (buccino): groups {bad} do not "
                         f"name a block; expected block_1, block_2, ...")


def read_recordings(subject: SubjectConfig) -> list[RawRecording]:
    """
    The whole session as one recording, its blocks in `blocks`.
    """
    session = _read_session(subject.paths["mat"])
    missing = [group for group in subject.groups if group not in session.blocks]
    if missing:
        raise ValueError(f"Subject '{subject.subject_id}': groups {missing} are not in "
                         f"the session, which has blocks {list(session.blocks)}.")
    return [session]


def read_background(subject: SubjectConfig) -> list[RawRecording]:
    """
    None: the dataset has no recordings without a task.
    """
    return []


def select_pruning_recordings(recordings: list[RawRecording],
                              backgrounds: dict[str, RawRecording]) -> list[RawRecording]:
    """
    The whole session, marked as one rest period. The session has none long enough: it
    starts with a task within seconds, rest trials last 8 s and the longest stretch
    without a task lasts 25 s on S1, 16.4 s on S2. The fNIRS criteria judge the coupling of the optodes
    to the scalp (the cardiac pulse in SCI and PSP, the amplitude level), which a motor
    task barely changes, while a whole session gives them far more windows to judge.

    Parameters
    ----------
    recordings : list[RawRecording]
        The session, as `read_recordings` returns it.
    backgrounds : dict[str, RawRecording]
        Not used: the dataset has none.
    """
    [session] = recordings
    return [mark_rest_period(session, 0.0, float(session.fnirs.time[-1]))]


# -------------------------------------------------------------------------
# The session
# -------------------------------------------------------------------------
def _read_session(path: Path) -> RawRecording:
    eeg_cell, fnirs_cell, events = read_mat(path)["data"]
    eeg = _read_eeg(eeg_cell, path)
    fnirs, geo3d = _read_fnirs(fnirs_cell, path)
    return RawRecording(group="session", run=path.stem, eeg=eeg, fnirs=fnirs, geo3d=geo3d,
                        stim=_stim_table(events, path),
                        blocks=_block_spans(events, float(eeg.times[-1]), path))


def _read_eeg(cell: np.ndarray, path: Path) -> mne.io.RawArray:
    """
    The EEG in volts, channels named as in the file, without positions.
    """
    rate, n_channels, labels, time, signals_uv = cell
    check_regular_time(time, rate, f"{path} EEG")
    _check_shape(signals_uv, len(time), n_channels, f"{path} EEG")
    return build_eeg(signals_uv.T * 1e-6, rate, [str(label) for label in labels])


def _read_fnirs(cell: np.ndarray, path: Path) -> tuple[xr.DataArray, xr.DataArray]:
    """
    The raw intensity, with a placeholder source and detector per channel, and no
    optode positions (an empty geo3d).
    """
    rate, n_channels, labels, time, wl1, wl2 = cell
    check_regular_time(time, rate, f"{path} fNIRS")
    for intensity in (wl1, wl2):
        _check_shape(intensity, len(time), n_channels, f"{path} fNIRS")

    channels = [str(label) for label in labels]
    amp = build_fnirs_amplitude(dict(zip(WAVELENGTHS_NM, (wl1, wl2))), time, channels,
                                [f"{ch}_S" for ch in channels], [f"{ch}_D" for ch in channels])
    return amp, build_geo3d({}, {})


def _stim_table(events: np.ndarray, path: Path) -> pd.DataFrame:
    """
    Every cue as a stimulus, lasting until the next event. The last one, which no
    event follows, lasts as long as the median of its trial type.
    """
    onsets, codes = events[:, 0], [int(code) for code in events[:, 1]]
    if np.any(np.diff(onsets) < 0):
        raise ValueError(f"{path}: events are not in time order.")
    unknown = sorted(set(codes) - set(TRIAL_TYPES) - {BLOCK_START_CODE})
    if unknown:
        raise ValueError(f"{path}: unknown event codes {unknown}; known: "
                         f"{sorted(TRIAL_TYPES)} and {BLOCK_START_CODE} (block start).")

    durations = np.append(np.diff(onsets), np.nan)
    is_cue = np.array([code != BLOCK_START_CODE for code in codes])
    stim = pd.DataFrame({"onset": onsets[is_cue], "duration": durations[is_cue],
                         "value": 1.0,
                         "trial_type": [TRIAL_TYPES[code] for code in codes
                                        if code != BLOCK_START_CODE]})
    median_duration = stim.groupby("trial_type")["duration"].transform("median")
    stim["duration"] = stim["duration"].fillna(median_duration)
    if stim["duration"].isna().any():
        last = stim["trial_type"].iloc[-1]
        raise ValueError(f"{path}: the last cue ({last}) is the only one of its trial "
                         f"type, so its duration cannot be taken from the others.")
    return stim.reset_index(drop=True)


def _block_spans(events: np.ndarray, session_end: float,
                 path: Path) -> dict[str, tuple[float, float]]:
    """
    Each block, block_1 first, from its start event to the next one; the last block
    ends with the session.
    """
    starts = [float(onset) for onset, code in events if int(code) == BLOCK_START_CODE]
    if not starts:
        raise ValueError(f"{path}: no block start events (code {BLOCK_START_CODE}).")
    ends = [*starts[1:], session_end]
    return {f"block_{k}": span for k, span in enumerate(zip(starts, ends), start=1)}


def _check_shape(signals: np.ndarray, n_times: int, n_channels: int, where: str):
    if signals.shape != (n_times, n_channels):
        raise ValueError(f"{where}: signals have shape {signals.shape}, expected "
                         f"({n_times}, {n_channels}) (time x channel).")

