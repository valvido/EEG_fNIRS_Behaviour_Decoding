"""
Dispatch to the reader for the subject's dataset.

One reader per dataset, chosen by `subject.format`. A reader knows its dataset's file
layout and conventions; the file types themselves are read by src/loading/files.py.
Every reader returns the same RawRecording objects, so nothing after loading knows
which dataset the data came from. The subject's montage is applied to what the reader
returned (src/loading/montage.py), so a reader never chooses positions. To add a
dataset, write a reader module that satisfies `Reader` and register it in READERS.
"""
from typing import Protocol

import mne
import numpy as np

from src.config import SubjectConfig
from src.loading import buccino, shin2017, synthetic
from src.loading.montage import apply_montage, check_montage
from src.types import RawRecording


class Reader(Protocol):
    """
    What a reader module provides.

    Attributes
    ----------
    REQUIRED_PATHS : tuple[str, ...]
        Keys of `subject.paths` the reader needs; each must exist on disk.
    """
    REQUIRED_PATHS: tuple[str, ...]

    def check(self, subject: SubjectConfig) -> None:
        """Raise if the subject's data is not laid out as the reader expects, before
        any data is read."""

    def read_recordings(self, subject: SubjectConfig) -> list[RawRecording]:
        """Every run of the subject's groups, unprocessed."""

    def read_background(self, subject: SubjectConfig) -> list[RawRecording]:
        """The background recording of each of the subject's groups, unprocessed, if the
        dataset has data without its signals of interest (the simulator's); otherwise
        none."""

    def select_pruning_recordings(self, recordings: list[RawRecording],
                                  backgrounds: dict[str, RawRecording]) -> list[RawRecording]:
        """The recordings the fNIRS is judged on, chosen from the runs or the
        backgrounds, each with exactly one stim event marking its rest period (edges
        left out, `mark_rest_period`). The first is the earliest; image space takes
        the measurement variance from it."""


READERS: dict[str, Reader] = {"synthetic": synthetic, "buccino": buccino,
                              "shin2017": shin2017}


def check_data_format(subject: SubjectConfig):
    """
    Check, before anything is read, that the subject's data is in a form loading can
    read:
    - its format has a reader
    - the paths that reader needs are set and exist
    - the reader's own layout checks pass (synthetic: every group has a folder with
      run_* folders in it; buccino: every group names a block; shin2017: every group
      names a block of a motor imagery session and the BBCI files are there)
    - the montage can be applied: fNIRS positions set, montage files on disk
    """
    if subject.format not in READERS:
        raise ValueError(f"Subject '{subject.subject_id}': unknown format "
                         f"{subject.format!r}. Known formats: {sorted(READERS)}.")
    reader = READERS[subject.format]
    missing = [key for key in reader.REQUIRED_PATHS if key not in subject.paths]
    if missing:
        raise ValueError(f"Subject '{subject.subject_id}' ({subject.format}) needs "
                         f"paths {list(reader.REQUIRED_PATHS)}; missing: {missing}.")
    absent = [str(subject.paths[key]) for key in reader.REQUIRED_PATHS
              if not subject.paths[key].exists()]
    if absent:
        raise FileNotFoundError(f"Subject '{subject.subject_id}': paths do not exist: {absent}.")
    reader.check(subject)
    check_montage(subject)


def load_recordings(subject: SubjectConfig) -> list[RawRecording]:
    """
    Every run of every group in `subject.groups`, unprocessed, with the subject's
    montage applied.
    """
    recordings = apply_montage(READERS[subject.format].read_recordings(subject),
                               subject.montage)
    _check_eeg_finite(subject, recordings)
    return recordings


def select_pruning_recordings(subject: SubjectConfig, recordings: list[RawRecording],
                              backgrounds: dict[str, RawRecording]) -> list[RawRecording]:
    """
    The recordings pruning judges fNIRS quality on, each with one stim event marking its
    rest period, the earliest first:
    - synthetic: the background recording of the group recorded earliest, its first
      minute
    - shin2017: every session, the minute of rest it starts with
    - buccino: the whole session, which has no rest period long enough

    Parameters
    ----------
    subject : SubjectConfig
    recordings : list[RawRecording]
        From `load_recordings`.
    backgrounds : dict[str, RawRecording]
        From `load_backgrounds`.
    """
    return READERS[subject.format].select_pruning_recordings(recordings, backgrounds)


def load_backgrounds(subject: SubjectConfig) -> dict[str, RawRecording]:
    """
    The background recording of each of the subject's groups, by group, with its montage
    applied. Empty if the dataset has none.
    """
    backgrounds = READERS[subject.format].read_background(subject)
    if not backgrounds:
        return {}
    backgrounds = apply_montage(backgrounds, subject.montage)
    _check_eeg_finite(subject, backgrounds)
    return {background.group: background for background in backgrounds}


def _check_eeg_finite(subject: SubjectConfig, recordings: list[RawRecording]):
    """
    Stop if any recording has NaN or inf in its raw EEG. Such a recording is broken
    and has to be cleaned by hand; skipping it would shrink the dataset unnoticed.
    All broken recordings are listed at once, so they can be fixed in one go.
    """
    broken = []
    for rec in recordings:
        picks = mne.pick_types(rec.eeg.info, eeg=True)
        data = rec.eeg.get_data(picks=picks)
        bad_channels = [rec.eeg.ch_names[i] for i, row in zip(picks, data)
                        if not np.isfinite(row).all()]
        if bad_channels:
            broken.append(f"{rec.group}/{rec.run}: {len(bad_channels)}/{len(data)} "
                          f"EEG channels contain NaN or inf ({', '.join(bad_channels)})")
    if broken:
        raise ValueError(f"Subject '{subject.subject_id}': raw EEG is not finite in "
                         f"{len(broken)} recording(s). Clean them by hand before "
                         f"running the pipeline.\n  " + "\n  ".join(broken))


def loading_report(subject: SubjectConfig, recordings: list[RawRecording]) -> str:
    """
    What was loaded: the runs of each group, the blocks of a session recorded in one
    piece, and the subject's data paths.
    """
    runs_by_group = {}
    for rec in recordings:
        runs_by_group.setdefault(rec.group, []).append(rec.run)
    lines = [f"Loaded {len(recordings)} runs"]
    lines += [f"  {group}: {', '.join(runs)}" for group, runs in runs_by_group.items()]
    for rec in recordings:
        if rec.blocks:
            spans = [f"{name} ({start:.0f}-{end:.0f} s)"
                     for name, (start, end) in rec.blocks.items()]
            lines.append(f"  blocks of {rec.run}: {', '.join(spans)}")
    lines += [f"  {key}: {path}" for key, path in subject.paths.items()]
    return "\n".join(lines)
