"""
Reader for the open access EEG+NIRS dataset of Shin et al. (2017), IEEE TNSRE 25(10):
1735-1745, dataset A: left versus right hand motor imagery. 29 subjects.

Each subject has an EEG folder and a NIRS folder, each holding BBCI toolbox files with
all six sessions of the subject; cell k is session k. Sessions 1, 3 and 5 are motor
imagery; 2, 4 and 6 (mental arithmetic) are not read.

    <paths.eeg>/cnt.mat     cnt{k}: EEG (time x channel, µV, 200 Hz): 30 EEG
                            channels, VEOG and HEOG
    <paths.eeg>/mrk.mat     mrk{k}: task onsets (ms on the EEG clock), code 16 = left
                            hand, 32 = right hand
    <paths.nirs>/cnt.mat    cnt{k}: raw intensity (time x channel, V, 10 Hz): every
                            channel at the low wavelength, then every channel at the
                            high one
    <paths.nirs>/mrk.mat    mrk{k}: the same task onsets on the NIRS clock, code 1 =
                            left hand, 2 = right hand
    <paths.nirs>/mnt.mat    the source and detector of each NIRS channel, as 10-5 sites

A trial is 2 s of instruction, 10 s of task from the marker on and 15 to 17 s of rest.
A session starts with 1 min of rest, of which the EEG files keep the last 30 s; the NIRS
files keep all of it.

EEG and NIRS were recorded by separate devices, each on its own clock; every trigger
was sent to both at the same time. The NIRS is put on the EEG clock by the median lag
between the two devices' markers of the same trials, and cut to the end of the EEG. What
the NIRS recorded before the EEG started is kept, at negative times, for the rest period;
preprocessing cuts both to the stimulus window.

A session is recorded in one piece and split into blocks of consecutive trials after
preprocessing. The subject's groups name them session<k>_block<b>, e.g. session3_block2.
The trials of a session come in pairs, one per hand, so each block of 10 trials holds
5 per hand. The dataset has no rest recording: pruning judges the fNIRS on the minute of
rest each session starts with.
"""
import dataclasses
import re

import mne
import numpy as np
import pandas as pd
import xarray as xr

from src.config import SubjectConfig
from src.loading.arrays import build_eeg, build_fnirs_amplitude, build_geo3d
from src.loading.files import read_mat
from src.loading.rest import mark_rest_period
from src.types import RawRecording

REQUIRED_PATHS = ("eeg", "nirs")
EEG_FILES = ("cnt.mat", "mrk.mat")
NIRS_FILES = ("cnt.mat", "mrk.mat", "mnt.mat")

MI_SESSIONS = (1, 3, 5)
BLOCKS_PER_SESSION = 2

# Marker codes, as the dataset's documentation lists them.
EEG_TRIAL_TYPES = {16: "left", 32: "right"}
NIRS_TRIAL_TYPES = {1: "left", 2: "right"}
EOG_CHANNELS = ("VEOG", "HEOG")

INSTRUCTION_S = 2.0   # the instruction is shown for this long before the task onset
TASK_S = 10.0
REST_S = 60.0         # the rest a session starts with, up to its first instruction
# The two devices' markers of a trial may disagree by this much: the NIRS was recorded
# at 12.5 Hz, so its markers fall on an 80 ms grid.
SYNC_TOLERANCE_S = 0.1

_GROUP = re.compile(r"session([1-9])_block([1-9]\d*)")


@dataclasses.dataclass(frozen=True)
class _SubjectFiles:
    """
    The subject's files as read, one list entry per session (session k at index k-1).
    """
    eeg_cnt: list[dict]
    eeg_mrk: list[dict]
    nirs_cnt: list[dict]
    nirs_mrk: list[dict]
    nirs_mnt: dict
    where: str


def check(subject: SubjectConfig):
    """
    Check that every group names a block of a motor imagery session, and that both
    folders hold the files the reader needs.
    """
    bad = [group for group in subject.groups if not _is_block(group)]
    if bad:
        raise ValueError(f"Subject '{subject.subject_id}' (shin2017): groups {bad} do not "
                         f"name a block; expected session<k>_block<b> with k in "
                         f"{list(MI_SESSIONS)} and b from 1 to {BLOCKS_PER_SESSION}.")
    absent = [str(subject.paths[key] / name)
              for key, names in (("eeg", EEG_FILES), ("nirs", NIRS_FILES))
              for name in names if not (subject.paths[key] / name).is_file()]
    if absent:
        raise FileNotFoundError(f"Subject '{subject.subject_id}': files do not exist: "
                                f"{absent}.")


def read_recordings(subject: SubjectConfig) -> list[RawRecording]:
    """
    Every session that holds one of the subject's groups, as one recording each, in
    recording order, its blocks in `blocks`.
    """
    files = _read_files(subject)
    sessions = sorted({int(_GROUP.fullmatch(group).group(1)) for group in subject.groups})
    return [_read_session(files, session) for session in sessions]


def read_background(subject: SubjectConfig) -> list[RawRecording]:
    """
    None: the dataset has no recordings without a task.
    """
    return []


def select_pruning_recordings(recordings: list[RawRecording],
                              backgrounds: dict[str, RawRecording]) -> list[RawRecording]:
    """
    Every session, its rest period marked as the REST_S seconds before the instruction
    of its first trial.

    Parameters
    ----------
    recordings : list[RawRecording]
        The sessions, in recording order, as `read_recordings` returns them.
    backgrounds : dict[str, RawRecording]
        Not used: the dataset has none.
    """
    first_instructions = [float(session.stim.onset.min()) - INSTRUCTION_S
                          for session in recordings]
    return [mark_rest_period(session, instruction - REST_S, instruction)
            for session, instruction in zip(recordings, first_instructions)]


def _is_block(group: str) -> bool:
    match = _GROUP.fullmatch(group)
    return (match is not None and int(match.group(1)) in MI_SESSIONS
            and int(match.group(2)) <= BLOCKS_PER_SESSION)


def _read_files(subject: SubjectConfig) -> _SubjectFiles:
    eeg_dir, nirs_dir = subject.paths["eeg"], subject.paths["nirs"]
    read = lambda folder, name: read_mat(folder / f"{name}.mat", structs_as_dicts=True)[name]
    return _SubjectFiles(eeg_cnt=read(eeg_dir, "cnt"), eeg_mrk=read(eeg_dir, "mrk"),
                         nirs_cnt=read(nirs_dir, "cnt"), nirs_mrk=read(nirs_dir, "mrk"),
                         nirs_mnt=read(nirs_dir, "mnt"),
                         where=f"Subject '{subject.subject_id}'")


# -------------------------------------------------------------------------
# A session
# -------------------------------------------------------------------------
def _read_session(files: _SubjectFiles, session: int) -> RawRecording:
    """
    One motor imagery session on the EEG clock, without optode positions (an empty
    geo3d): the files place the optodes on a unit sphere only.
    """
    where = f"{files.where} session {session}"
    eeg_cnt, nirs_cnt = files.eeg_cnt[session - 1], files.nirs_cnt[session - 1]
    eeg_mrk, nirs_mrk = files.eeg_mrk[session - 1], files.nirs_mrk[session - 1]
    titles = {"EEG": eeg_cnt["title"], "NIRS": nirs_cnt["title"]}
    if set(titles.values()) != {"MI"}:
        raise ValueError(f"{where}: expected a motor imagery session (title 'MI'), the "
                         f"files' titles are {titles}.")

    eeg = _read_eeg(eeg_cnt, where)
    onsets, trial_types = _trials(eeg_mrk, nirs_mrk, where)
    lag = _clock_lag(eeg_mrk, nirs_mrk, where)
    session_end = float(eeg.times[-1])
    fnirs = _read_fnirs(nirs_cnt, files.nirs_mnt, lag, session_end, where)
    stim = pd.DataFrame({"onset": onsets, "duration": TASK_S, "value": 1.0,
                         "trial_type": trial_types})
    return RawRecording(group="session", run=f"session{session}", eeg=eeg, fnirs=fnirs,
                        geo3d=build_geo3d({}, {}), stim=stim,
                        blocks=_block_spans(session, onsets, session_end, where))


def _read_eeg(cnt: dict, where: str) -> mne.io.RawArray:
    """
    The EEG in volts, channels named as in the file, VEOG and HEOG typed "eog",
    without positions.
    """
    if cnt["yUnit"] != "uV":
        raise ValueError(f"{where}: EEG is in {cnt['yUnit']!r}, expected 'uV'.")
    labels = [str(label) for label in cnt["clab"]]
    eeg = build_eeg(np.asarray(cnt["x"], dtype=float).T * 1e-6, cnt["fs"], labels)
    eeg.set_channel_types({ch: "eog" for ch in EOG_CHANNELS if ch in labels},
                          verbose=False)
    return eeg


def _read_fnirs(cnt: dict, mnt: dict, lag: float, session_end: float,
                where: str) -> xr.DataArray:
    """
    The raw intensity on the EEG clock: shifted back by the lag of the NIRS clock and
    cut to the end of the EEG, `session_end`. What was recorded before the EEG started
    is kept, at negative times. Each channel has the source and detector site the
    montage file gives it.
    """
    if cnt["yUnit"] != "V":
        raise ValueError(f"{where}: NIRS is in {cnt['yUnit']!r}, expected 'V'.")
    channels = [str(label) for label in mnt["clab"]]
    columns = [f"{ch}lowWL" for ch in channels] + [f"{ch}highWL" for ch in channels]
    if [str(label) for label in cnt["clab"]] != columns:
        raise ValueError(f"{where}: NIRS columns are not the montage's channels at the "
                         f"low wavelength, then at the high one.")
    sd = np.asarray(mnt["sd"], dtype=int) - 1
    sources = [str(mnt["source"]["clab"][s]) for s in sd[:, 0]]
    detectors = [str(mnt["detector"]["clab"][d]) for d in sd[:, 1]]
    miswired = [ch for ch, s, d in zip(channels, sources, detectors) if ch != s + d]
    if miswired:
        raise ValueError(f"{where}: NIRS channels {miswired} are not named after their "
                         f"source and detector.")

    intensity = np.asarray(cnt["x"], dtype=float)
    time = np.arange(len(intensity)) / float(cnt["fs"]) - lag
    keep = time <= session_end
    low_nm, high_nm = (float(wl) for wl in cnt["wavelengths"])
    n = len(channels)
    return build_fnirs_amplitude({low_nm: intensity[keep, :n], high_nm: intensity[keep, n:]},
                                 time[keep], channels, sources, detectors)


def _trials(eeg_mrk: dict, nirs_mrk: dict, where: str) -> tuple[np.ndarray, list[str]]:
    """
    The task onsets (s on the EEG clock) and the hand of each trial. Both devices must
    have marked the same trials.
    """
    eeg_codes = [int(code) for code in np.atleast_1d(eeg_mrk["event"]["desc"])]
    nirs_codes = [int(code) for code in np.atleast_1d(nirs_mrk["event"]["desc"])]
    for device, codes, known in (("EEG", eeg_codes, EEG_TRIAL_TYPES),
                                 ("NIRS", nirs_codes, NIRS_TRIAL_TYPES)):
        unknown = sorted(set(codes) - set(known))
        if unknown:
            raise ValueError(f"{where}: unknown {device} marker codes {unknown}; known: "
                             f"{sorted(known)}.")
    trial_types = [EEG_TRIAL_TYPES[code] for code in eeg_codes]
    if trial_types != [NIRS_TRIAL_TYPES[code] for code in nirs_codes]:
        raise ValueError(f"{where}: the EEG and NIRS markers disagree on the trials.")
    onsets = np.atleast_1d(eeg_mrk["time"]).astype(float) / 1000.0
    if np.any(np.diff(onsets) <= 0):
        raise ValueError(f"{where}: EEG markers are not in time order.")
    return onsets, trial_types


def _clock_lag(eeg_mrk: dict, nirs_mrk: dict, where: str) -> float:
    """
    How far the NIRS clock is ahead of the EEG clock (s): the median over the trials of
    the lag between the two devices' markers of a trial.
    """
    lags = (np.atleast_1d(nirs_mrk["time"]).astype(float)
            - np.atleast_1d(eeg_mrk["time"]).astype(float)) / 1000.0
    lag = float(np.median(lags))
    spread = float(np.abs(lags - lag).max())
    if spread > SYNC_TOLERANCE_S:
        raise ValueError(f"{where}: the NIRS markers lag the EEG markers by {lag:.3f} s, "
                         f"but individual trials deviate by up to {spread:.3f} s (more "
                         f"than {SYNC_TOLERANCE_S} s), so the clocks cannot be aligned.")
    return lag


def _block_spans(session: int, onsets: np.ndarray, session_end: float,
                 where: str) -> dict[str, tuple[float, float]]:
    """
    Each block of BLOCKS_PER_SESSION, from the instruction of its first trial to that
    of the next block's first trial; the first block starts with the session, the last
    ends with it. A block so keeps the rest after its last trial, in which that trial's
    hemodynamic response returns.
    """
    if len(onsets) % BLOCKS_PER_SESSION:
        raise ValueError(f"{where}: {len(onsets)} trials do not split into "
                         f"{BLOCKS_PER_SESSION} blocks of equal size.")
    per_block = len(onsets) // BLOCKS_PER_SESSION
    starts = [0.0] + [float(onsets[b * per_block]) - INSTRUCTION_S
                      for b in range(1, BLOCKS_PER_SESSION)]
    ends = [*starts[1:], session_end]
    return {f"session{session}_block{b}": span
            for b, span in enumerate(zip(starts, ends), start=1)}
