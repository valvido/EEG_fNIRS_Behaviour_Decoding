"""
Split a session recorded in one piece into one run per block, after it was preprocessed
as a whole.
"""
import dataclasses

import numpy as np

from src.types import PreprocessedRecording


def split_into_blocks(recordings: list[PreprocessedRecording],
                      groups: tuple[str, ...]) -> list[PreprocessedRecording]:
    """
    Replace every session that has blocks by one run per block in `groups`, in
    recording order; runs without blocks are passed on as they are.

    Filters and ICA ran on the whole session, so no block has edges of its own. The
    blocks keep the session's scaling (`eeg_std`, `fnirs_std`). A block holds the
    fNIRS samples from its start up to, not including, its end, and the EEG that
    belongs to them: on the fNIRS time axis (band power) the same samples, at the
    EEG's own rate (band-passed) the k = EEG samples / fNIRS samples that follow each
    fNIRS sample, so mSPoC's blocks stay on their fNIRS samples.

    Parameters
    ----------
    recordings : list[PreprocessedRecording]
    groups : tuple[str, ...]
        The subject's CV groups; a session's blocks not among them are left out.

    Returns
    -------
    list[PreprocessedRecording]
        Block runs have the block as group, the session's run name and no blocks.
    """
    out = []
    for recording in recordings:
        if recording.blocks is None:
            out.append(recording)
            continue
        out += [cut_block(recording, name, span) for name, span in recording.blocks.items()
                if name in groups]
    return out


def cut_block(recording: PreprocessedRecording, name: str,
              span: tuple[float, float]) -> PreprocessedRecording:
    """
    One block of a preprocessed session, from `span[0]` up to `span[1]` (s).

    Raises
    ------
    ValueError
        If the block holds no fNIRS sample, or the EEG does not hold a whole number of
        samples per fNIRS sample.
    """
    start, end = span
    fnirs_time = recording.fnirs.time.values
    first, stop = np.searchsorted(fnirs_time, [start, end], side="left")
    if stop <= first:
        raise ValueError(f"{recording.group}/{recording.run}: block {name} "
                         f"({start:.1f}-{end:.1f} s) holds no fNIRS sample.")
    k = _eeg_samples_per_fnirs_sample(recording)

    on_fnirs_axis = lambda da: None if da is None else da.isel(time=slice(first, stop))
    in_block = (recording.stim.onset >= start) & (recording.stim.onset < end)
    return dataclasses.replace(
        recording, group=name,
        eeg=recording.eeg.isel(time=slice(first * k, stop * k)),
        fnirs=on_fnirs_axis(recording.fnirs),
        s_eeg=on_fnirs_axis(recording.s_eeg), s_fnirs=on_fnirs_axis(recording.s_fnirs),
        stim=recording.stim[in_block].reset_index(drop=True),
        window=(max(recording.window[0], start), min(recording.window[1], end)),
        blocks=None)


def _eeg_samples_per_fnirs_sample(recording: PreprocessedRecording) -> int:
    """1 for band power on the fNIRS time axis, k for band-passed EEG cut into blocks."""
    n_eeg, n_fnirs = recording.eeg.sizes["time"], recording.fnirs.sizes["time"]
    if n_eeg % n_fnirs:
        raise ValueError(f"{recording.group}/{recording.run}: {n_eeg} EEG samples are not "
                         f"a whole number per fNIRS sample ({n_fnirs}).")
    return n_eeg // n_fnirs
