"""
Reader for synthetic data exported by synthetic_DOT-EEG_pipeline.

Expected layout, with or without the inner <subject_id>/ folder:

    <paths.data>/[<subject_id>/]<subject_id>_<group>/run_<nnn>_seed_<seed>/
        *_raw.fif                   EEG
        *_amp_*.snirf               fNIRS amplitude, stim table, optode geometry
        *_s_eeg_temporal_*.nc       ground truth: EEG source time courses
        *_s_fnirs_temporal_*.nc     ground truth: fNIRS source time courses
        *_A_eeg_*.nc                ground truth: EEG mixing matrix
        *_A_fnirs_*.nc              ground truth: fNIRS mixing matrix

The rest recording pruning judges channels on is the background recording the
simulation was built on, at paths.rest_eeg / paths.rest_fnirs. Outside the pipeline,
read_background_rest finds it from the subject ID alone, in the layout

    <data_root>/background_recordings/<subject_id>/<subject_id>_<recording>-raw.fif
    <data_root>/background_recordings/<subject_id>/<subject_id>_<recording>.snirf
"""
from pathlib import Path

import numpy as np
import xarray as xr

from src.config import SubjectConfig
from src.loading.files import read_fif, read_snirf
from src.types import GroundTruth, RawRecording

REQUIRED_PATHS = ("data", "rest_eeg", "rest_fnirs")

# Ground-truth sources are labelled by the hand they belong to. 
HAND_SOURCES = ("right_hand", "left_hand")


def check(subject: SubjectConfig):
    """
    Check that every group folder exists and holds at least one run.
    """
    for group in subject.groups:
        _run_dirs(subject, group)


def read_recordings(subject: SubjectConfig) -> list[RawRecording]:
    """
    Every run of every group in `subject.groups`, in that order.
    """
    hands = source_hands(subject)
    return [_read_run(run_dir, group, hands)
            for group in subject.groups
            for run_dir in _run_dirs(subject, group)]


def read_rest(subject: SubjectConfig) -> RawRecording:
    """
    The background recording the simulation was built on, without ground truth.
    """
    return _read_rest_files(subject.paths["rest_eeg"], subject.paths["rest_fnirs"])


def read_background_rest(subject_id: str, data_root: Path,
                         recording: str = "sitting_pre_rest_1") -> RawRecording:
    """
    A subject's background rest recording, found from its ID alone. For use outside
    the pipeline (e.g. notebooks), where writing a subject YAML would be overkill;
    the pipeline reads the paths from the subject YAML instead.

    Parameters
    ----------
    subject_id : str
        E.g. "sub01".
    data_root : Path
        The folder holding background_recordings/.
    recording : str
        Which background recording, e.g. "sitting_pre_rest_1".

    Returns
    -------
    RawRecording
        Without ground truth.
    """
    folder = Path(data_root) / "background_recordings" / subject_id
    return _read_rest_files(folder / f"{subject_id}_{recording}-raw.fif",
                            folder / f"{subject_id}_{recording}.snirf")


def _read_rest_files(eeg_path: Path, fnirs_path: Path) -> RawRecording:
    """
    Read a rest recording and check that its EEG and fNIRS belong together.
    """
    eeg = read_fif(eeg_path)
    fnirs, geo3d, stim = read_snirf(fnirs_path)
    _check_clocks_overlap(eeg, fnirs, eeg_path.parent)
    _check_eeg_events_match(eeg, stim, eeg_path)
    return RawRecording(group="rest", run=fnirs_path.stem, eeg=eeg, fnirs=fnirs,
                     geo3d=geo3d, stim=stim)


# -------------------------------------------------------------------------
# One run
# -------------------------------------------------------------------------
def _read_run(run_dir: Path, group: str, hands: dict[str, str]) -> RawRecording:
    eeg = read_fif(_one_file(run_dir, "*_raw.fif"))
    fnirs, geo3d, stim = read_snirf(_one_file(run_dir, "*_amp_*.snirf"))
    _check_clocks_overlap(eeg, fnirs, run_dir)
    # The simulator writes the stimuli into the SNIRF only; the EEG file carries just
    # its "rest" window. Any other EEG annotation must agree with the SNIRF.
    if any(desc != "rest" for desc in eeg.annotations.description):
        _check_eeg_events_match(eeg, stim, run_dir)
    return RawRecording(group=group, run=run_dir.name, eeg=eeg, fnirs=fnirs, geo3d=geo3d,
                     stim=stim, ground_truth=_read_ground_truth(run_dir, hands))


def source_hands(subject: SubjectConfig) -> dict[str, str]:
    """
    The simulator's source names mapped to hand labels, from the trial_source_map of
    its config: {"<source name>": "right_hand" | "left_hand"}.
    """
    try:
        source_map = subject.simulation["source_config"]["trial_source_map"]
    except (KeyError, TypeError):
        raise ValueError(f"Subject '{subject.subject_id}': the simulation config has no "
                         f"source_config.trial_source_map to name the sources by hand.")
    hands = {str(source): f"{hand}_hand" for hand, sources in source_map.items()
             for source in sources}
    unknown = set(hands.values()) - set(HAND_SOURCES)
    if unknown:
        raise ValueError(f"Subject '{subject.subject_id}': trial_source_map names hands "
                         f"{sorted(unknown)}; expected right and left.")
    return hands


def _read_ground_truth(run_dir: Path, hands: dict[str, str]) -> GroundTruth:
    """
    The four ground-truth arrays, read and tidied: the source axis is named "source",
    its labels are the hands (`hands`), right hand first. Nothing else is changed (no
    sign convention, no channel selection, no chromophore selection).
    """
    def load(pattern):
        return _tidy_sources(xr.load_dataarray(_one_file(run_dir, pattern)), hands,
                             run_dir / pattern)

    return GroundTruth(
        s_eeg=load("*_s_eeg_temporal_*.nc").transpose("time", "source"),
        s_fnirs=load("*_s_fnirs_temporal_*.nc").transpose("time", "source", ...),
        A_eeg=load("*_A_eeg_*.nc").transpose("channel", "source"),
        A_fnirs=load("*_A_fnirs_*.nc").transpose("channel", "source", ...),
    )


def _tidy_sources(da: xr.DataArray, hands: dict[str, str], where: Path) -> xr.DataArray:
    """
    Name the source axis "source", relabel the sources by hand (`hands`) and order
    them right hand, left hand.

    Newer exports call the axis "generator", to avoid a clash with the optode "source"
    coordinate on the fNIRS channel axis; that optode coordinate is dropped.
    """
    if "generator" in da.dims:
        if "source" in da.coords:
            da = da.drop_vars("source")
        da = da.rename(generator="source")
    if "source" not in da.dims:
        raise ValueError(f"{where}: no 'source' or 'generator' axis (dims={da.dims}).")

    names = [str(s) for s in da.coords["source"].values]
    if not names or len(names) != len(set(names)) or set(names) - set(hands):
        raise ValueError(f"{where}: source names {names} must be a non-empty subset of "
                         f"{sorted(hands)} (the simulation's trial_source_map) without repeats.")
    da = da.assign_coords(source=[hands[s] for s in names])
    labels = [str(s) for s in da.source.values]
    return da.sel(source=[s for s in HAND_SOURCES if s in labels])


# -------------------------------------------------------------------------
# Files and checks
# -------------------------------------------------------------------------
def _run_dirs(subject: SubjectConfig, group: str) -> list[Path]:
    """
    The run folders of one group. The group folder is <subject_id>_<group>, directly
    in paths.data or inside a <subject_id>/ folder (older exports).
    """
    data_dir, name = subject.paths["data"], f"{subject.subject_id}_{group}"
    candidates = [d for d in (data_dir / name, data_dir / subject.subject_id / name) if d.is_dir()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Group '{group}': expected exactly one folder {name}/ in {data_dir} or "
            f"{data_dir / subject.subject_id}, found {len(candidates)}.")
    runs = sorted(d for d in candidates[0].glob("run_*") if d.is_dir())
    if not runs:
        raise FileNotFoundError(f"Group '{group}': no run_* folders in {candidates[0]}.")
    return runs


def _one_file(run_dir: Path, pattern: str) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"{run_dir}: expected exactly one file matching {pattern}, "
                                f"found {len(matches)}.")
    return matches[0]


def _check_clocks_overlap(eeg, fnirs: xr.DataArray, where: Path):
    eeg_end = eeg.times[-1]
    fnirs_start, fnirs_end = float(fnirs.time[0]), float(fnirs.time[-1])
    if fnirs_start >= eeg_end or fnirs_end <= 0:
        raise ValueError(f"{where}: EEG (0-{eeg_end:.1f} s) and fNIRS "
                         f"({fnirs_start:.1f}-{fnirs_end:.1f} s) do not overlap.")


def _check_eeg_events_match(eeg, stim, where: Path, tolerance_s: float = 0.05):
    """
    The EEG annotations must name the same events as the SNIRF stim table: same
    count, same labels, onsets within `tolerance_s`. EEG onsets are absolute, so
    `first_time` is subtracted to put them on the shared clock.
    """
    ann = eeg.annotations
    order = np.argsort(ann.onset)
    eeg_onsets = ann.onset[order] - eeg.first_time
    eeg_labels = [str(ann.description[i]) for i in order]
    snirf_labels = [str(t) for t in stim["trial_type"]]

    if len(eeg_onsets) != len(stim) or eeg_labels != snirf_labels:
        raise ValueError(f"{where}: EEG events {eeg_labels} do not match the SNIRF stim "
                         f"table {snirf_labels}.")
    gap = np.abs(eeg_onsets - stim["onset"].to_numpy()).max() if len(stim) else 0.0
    if gap > tolerance_s:
        raise ValueError(f"{where}: EEG and SNIRF event onsets differ by up to {gap:.3f} s "
                         f"(tolerance {tolerance_s} s).")
