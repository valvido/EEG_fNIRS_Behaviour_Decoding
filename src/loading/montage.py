"""
Apply the subject's montage (`montage:` in the subject YAML) to what a reader returned.

This is the only place electrode and optode positions are chosen. Afterwards they
travel with the data: the electrode positions in the EEG's mne Raw, the optode
positions as geo3d.
"""
import dataclasses
from pathlib import Path

import cedalion.io
import mne
import numpy as np
import pandas as pd
import xarray as xr

from src.config import EegMontageConfig, FnirsMontageConfig, MontageConfig, SubjectConfig
from src.loading.arrays import build_geo3d
from src.types import RawRecording

# Columns of a montage.fnirs.layout file.
LAYOUT_COLUMNS = ["channel", "source", "detector"]


def check_montage(subject: SubjectConfig):
    """
    Check, before anything is read, that the montage can be applied: the fNIRS
    positions are set, the templates exist and every montage file is on disk.
    """
    montage, where = subject.montage, f"Subject '{subject.subject_id}'"
    if montage.fnirs is None:
        raise ValueError(f"{where}: montage.fnirs is not set, so the fNIRS channels have "
                         f"no optode positions. Set montage.fnirs.recording: true, "
                         f"montage.fnirs.file or montage.fnirs.layout.")
    for key, standard in (("eeg", montage.eeg.standard), ("fnirs", montage.fnirs.standard)):
        if standard is not None and standard not in mne.channels.get_builtin_montages():
            raise ValueError(f"{where}: montage.{key}.standard {standard!r} is not an MNE "
                             f"template; known: {mne.channels.get_builtin_montages()}.")
    files = [f for f in (montage.eeg.file, montage.fnirs.file, montage.fnirs.layout)
             if f is not None]
    absent = [str(f) for f in files if not f.is_file()]
    if absent:
        raise FileNotFoundError(f"{where}: montage files do not exist: {absent}.")


def apply_montage(recordings: list[RawRecording],
                  montage: MontageConfig) -> list[RawRecording]:
    """
    Every recording with the subject's electrode and optode positions.

    Each montage file is read once, for all recordings.

    Parameters
    ----------
    recordings : list[RawRecording]
        As a reader returned them.
    montage : MontageConfig

    Returns
    -------
    list[RawRecording]
        EEG channels spelled as in the montage (e.g. "Fc5" -> "FC5") and placed by it.
        geo3d replaced by the optode positions of `montage.fnirs.file`, or of the
        layout's sites; with a layout, each fNIRS channel's source and detector are
        the layout's.
    """
    if montage.fnirs is None:
        raise ValueError("montage.fnirs is not set, so the fNIRS channels have no optode "
                         "positions.")
    eeg_montage = _eeg_montage(montage.eeg)
    layout = None if montage.fnirs.layout is None else read_layout(montage.fnirs.layout)
    optodes = _optode_positions(montage.fnirs, layout)
    out = []
    for rec in recordings:
        fnirs = rec.fnirs if layout is None else _wired_fnirs(rec, layout)
        geo3d = rec.geo3d if optodes is None else optodes
        _check_optodes_placed(fnirs, geo3d, f"{rec.group}/{rec.run}")
        out.append(dataclasses.replace(rec, eeg=_placed_eeg(rec, eeg_montage), fnirs=fnirs,
                                       geo3d=geo3d))
    return out


def read_layout(path: Path) -> dict[str, tuple[str, str]]:
    """
    The optodes of every channel in a montage.fnirs.layout file.

    Returns
    -------
    dict[str, tuple[str, str]]
        Channel to (source site, detector site), in file order.

    Raises
    ------
    ValueError
        If the columns are not channel, source, detector, a channel is listed twice,
        or a site is both a source and a detector.
    """
    table = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    if list(table.columns) != LAYOUT_COLUMNS:
        raise ValueError(f"{path}: columns are {list(table.columns)}, expected "
                         f"{LAYOUT_COLUMNS}.")
    repeated = sorted(table.channel[table.channel.duplicated()])
    if repeated:
        raise ValueError(f"{path}: channels listed twice: {repeated}.")
    both = sorted(set(table.source) & set(table.detector))
    if both:
        raise ValueError(f"{path}: sites {both} are both a source and a detector.")
    return {row.channel: (row.source, row.detector) for row in table.itertuples()}


def _eeg_montage(cfg: EegMontageConfig) -> mne.channels.DigMontage:
    if cfg.standard is not None:
        return mne.channels.make_standard_montage(cfg.standard)
    return mne.channels.read_custom_montage(cfg.file)


def _optode_positions(cfg: FnirsMontageConfig,
                      layout: dict[str, tuple[str, str]] | None) -> xr.DataArray | None:
    """The optode positions in mm, or None to keep the recording's own."""
    if cfg.recording:
        return None
    if cfg.file is not None:
        return cedalion.io.load_tsv(cfg.file).pint.to("mm")
    return _template_optodes(layout, cfg.standard)


def _template_optodes(layout: dict[str, tuple[str, str]], standard: str) -> xr.DataArray:
    """
    The layout's sites placed by the template, with its nasion and preauricular points
    as landmarks, in MNE's head frame: the frame the EEG electrodes are placed in, so
    electrodes and optodes share one frame.
    """
    sources = list(dict.fromkeys(source for source, _ in layout.values()))
    detectors = list(dict.fromkeys(detector for _, detector in layout.values()))
    template = mne.channels.make_standard_montage(standard)
    unknown = sorted(set(sources + detectors) - set(template.ch_names))
    if unknown:
        raise ValueError(f"Optode sites {unknown} of the fNIRS layout are not in the "
                         f"{standard} template.")
    info = mne.create_info([*sources, *detectors], sfreq=1.0, ch_types="eeg")
    info.set_montage(template)
    positions_m = info.get_montage().get_positions()
    in_mm = lambda names: {name: np.asarray(positions_m["ch_pos"][name]) * 1000.0
                           for name in names}
    landmarks = {"Nz": positions_m["nasion"] * 1000.0, "LPA": positions_m["lpa"] * 1000.0,
                 "RPA": positions_m["rpa"] * 1000.0}
    return build_geo3d(in_mm(sources), in_mm(detectors), landmarks)


def _wired_fnirs(rec: RawRecording, layout: dict[str, tuple[str, str]]) -> xr.DataArray:
    """The fNIRS amplitude with each channel's source and detector from the layout."""
    channels = [str(ch) for ch in rec.fnirs.channel.values]
    missing = [ch for ch in channels if ch not in layout]
    if missing:
        raise ValueError(f"{rec.group}/{rec.run}: fNIRS channels {missing} are not in the "
                         f"fNIRS layout.")
    return rec.fnirs.assign_coords(source=("channel", [layout[ch][0] for ch in channels]),
                                   detector=("channel", [layout[ch][1] for ch in channels]))


def _placed_eeg(rec: RawRecording, montage: mne.channels.DigMontage) -> mne.io.Raw:
    """
    The EEG with its channels spelled as in the montage and placed by it. EOG and EMG
    channels are left as they are.
    """
    spelled = {name.lower(): name for name in montage.ch_names}
    names = [rec.eeg.ch_names[i] for i in mne.pick_types(rec.eeg.info, eeg=True)]
    missing = [name for name in names if name.lower() not in spelled]
    if missing:
        raise ValueError(f"{rec.group}/{rec.run}: EEG channels {missing} are not in the "
                         f"montage.")
    eeg = rec.eeg.copy()
    eeg.rename_channels({name: spelled[name.lower()] for name in names
                         if name != spelled[name.lower()]})
    return eeg.set_montage(montage)


def _check_optodes_placed(fnirs: xr.DataArray, geo3d: xr.DataArray, where: str):
    """Every source and detector of the fNIRS channels must have a position."""
    optodes = {str(o) for o in [*fnirs.source.values, *fnirs.detector.values]}
    labels = {str(label) for label in geo3d.label.values} if "label" in geo3d.dims else set()
    missing = sorted(optodes - labels)
    if missing:
        raise ValueError(f"{where}: optodes {missing} of the fNIRS channels have no "
                         f"position in the fNIRS montage.")
