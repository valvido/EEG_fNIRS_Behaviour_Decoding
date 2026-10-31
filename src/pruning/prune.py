"""
Prune a subject's recordings: detect bad channels once on the rest recording, then
remove them from every run.
"""
from src.config import PruningConfig
from src.positions import channel_optodes
from src.pruning.eeg import detect_bad_eeg_channels
from src.pruning.fnirs import detect_bad_fnirs_channels
from src.pruning.segment import rest_window
from src.types import BadChannels, PrunedRecording, RawRecording

# The electrodes over the hand motor areas. Interpolating one replaces its signal with
# a mix of its neighbours, so a subject with either one bad cannot be used.
MOTOR_EEG_CHANNELS = ("C3", "C4")


def prune_recordings(
        recordings: list[RawRecording], rest: RawRecording,
        cfg: PruningConfig) -> tuple[list[PrunedRecording], BadChannels]:
    """
    Detect the bad channels on the rest recording and remove them from every run.
    Bad EEG channels are interpolated, so every run keeps the same EEG montage; bad
    fNIRS channels are dropped. Every run gets the same lists, so all runs have the
    same channels afterwards.

    Parameters
    ----------
    recordings : list[RawRecording]
        The subject's runs.
    rest : RawRecording
        The recording channel quality is judged on (for synthetic data, the first
        background recording of the session).
    cfg : PruningConfig
        Detection window and thresholds.

    Returns
    -------
    pruned : list[PrunedRecording]
        One per run, in the order given.
    bad_channels : BadChannels
        The lists applied to every run, for the run's result.

    Raises
    ------
    ValueError
        If a motor electrode (MOTOR_EEG_CHANNELS) is bad.
    """
    bad_channels = detect_bad_channels(rest, cfg)
    bad_motor = [ch for ch in MOTOR_EEG_CHANNELS if ch in bad_channels.eeg]
    if bad_motor:
        raise ValueError(f"Motor electrode(s) {bad_motor} are bad on rest recording "
                         f"'{rest.run}'. Interpolation would replace their signal, so "
                         "this subject cannot be used.")
    pruned = [remove_bad_channels(recording, bad_channels) for recording in recordings]
    return pruned, bad_channels


def detect_bad_channels(rest: RawRecording, cfg: PruningConfig) -> BadChannels:
    """
    Detect bad EEG and fNIRS channels on the rest recording. The EEG is judged on the
    whole rest recording. The fNIRS is judged on the pruning window only, and is cut
    to it only after it has been cleaned, since filtering just the window would add
    edge artefacts.
    """
    bad_eeg, eeg_detail = detect_bad_eeg_channels(rest.eeg, cfg.eeg, cfg.random_state)
    start, end = rest_window(rest, cfg.window_sec)
    bad_fnirs, fnirs_detail = detect_bad_fnirs_channels(
        rest.fnirs, rest.geo3d, (start, end), cfg.fnirs)
    return BadChannels(eeg=bad_eeg, fnirs=bad_fnirs,
                       eeg_detail=eeg_detail, fnirs_detail=fnirs_detail)


def remove_bad_channels(recording: RawRecording,
                        bad_channels: BadChannels) -> PrunedRecording:
    """
    Interpolate the bad EEG channels and drop the bad fNIRS channels of one run.
    """
    # Interpolation uses the electrode positions loading set from the subject's montage.
    eeg = recording.eeg.copy()
    eeg.info["bads"] = [ch for ch in bad_channels.eeg if ch in eeg.ch_names]
    eeg.interpolate_bads(reset_bads=True)

    fnirs = recording.fnirs.drop_sel(channel=list(bad_channels.fnirs))

    return PrunedRecording(group=recording.group, run=recording.run, eeg=eeg, fnirs=fnirs,
                           geo3d=recording.geo3d,
                           fnirs_optodes=channel_optodes(recording.fnirs),
                           stim=recording.stim, ground_truth=recording.ground_truth,
                           bad_channels=bad_channels, blocks=recording.blocks)
