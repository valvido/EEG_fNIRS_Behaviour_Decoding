"""
Prune a subject's recordings: detect the bad channels, then remove them from every run.
"""
import dataclasses

from src.config import PruningConfig
from src.positions import channel_optodes
from src.pruning.eeg import detect_bad_eeg_channels_by_recording
from src.pruning.fnirs import detect_bad_fnirs_channels
from src.pruning.segment import rest_window
from src.types import BadChannels, PrunedRecording, RawRecording


def prune_recordings(
        recordings: list[RawRecording], references: list[RawRecording],
        backgrounds: dict[str, RawRecording],
        cfg: PruningConfig) -> tuple[list[PrunedRecording], BadChannels]:
    """
    Detect the bad channels and remove them from every run. Each recording has its own
    bad EEG channels, which are interpolated, so every run keeps the same EEG montage.
    The fNIRS channels bad in any of `references` are dropped from every run, so all
    runs keep the same fNIRS channels.

    Parameters
    ----------
    recordings : list[RawRecording]
        The subject's runs.
    references : list[RawRecording]
        The recordings fNIRS quality is judged on (`select_pruning_recordings`), each
        with one stim event marking its rest period.
    backgrounds : dict[str, RawRecording]
        The background recording of each group (`load_backgrounds`). A run of such a
        group gets the bad EEG channels of its background, any other those of its own
        recording.
    cfg : PruningConfig
        fNIRS window, thresholds and the motor electrodes.

    Returns
    -------
    pruned : list[PrunedRecording]
        One per run, in the order given, each with the EEG channels of its own recording
        as `bad_channels.eeg`.
    bad_channels : BadChannels
        All of the subject's, for the run's result.

    Raises
    ------
    ValueError
        If a motor electrode (cfg.eeg.motor_channels) is not in the EEG of a recording,
        or is bad in any recording.
    """
    judged = [backgrounds.get(recording.group, recording) for recording in recordings]
    motor = cfg.eeg.motor_channels
    absent = {recording.run: [ch for ch in motor if ch not in recording.eeg.ch_names]
              for recording in judged if set(motor) - set(recording.eeg.ch_names)}
    if absent:
        raise ValueError(f"Motor electrodes are not in the EEG of recordings {absent}. Set "
                         f"eeg.motor_channels in the pruning YAML to electrodes of this "
                         f"montage.")
    bad_channels = detect_bad_channels(references, judged, cfg)
    bad_motor = {name: [ch for ch in motor if ch in channels]
                 for name, channels in bad_channels.eeg_by_recording.items()
                 if set(motor) & set(channels)}
    if bad_motor:
        raise ValueError(f"Motor electrodes are bad in recordings {bad_motor}. Interpolation "
                         "would replace their signal, so this subject cannot be used.")
    pruned = [remove_bad_channels(recording, dataclasses.replace(
                  bad_channels, eeg=bad_channels.eeg_by_recording[judged_recording.run]))
              for recording, judged_recording in zip(recordings, judged)]
    return pruned, bad_channels


def detect_bad_channels(references: list[RawRecording], eeg_recordings: list[RawRecording],
                        cfg: PruningConfig) -> BadChannels:
    """
    Detect bad EEG channels on each of `eeg_recordings` on its own, judged whole, and bad
    fNIRS channels on the window of the rest period of each of `references`. A fNIRS
    channel bad in any reference is bad. The fNIRS is cut only after it has been cleaned,
    since filtering just the window would add edge artefacts. A recording listed more
    than once is judged once.
    """
    eeg_by_recording, eeg_detail = detect_bad_eeg_channels_by_recording(
        {recording.run: recording.eeg for recording in eeg_recordings},
        cfg.eeg, cfg.random_state)
    bad_eeg = tuple(sorted({ch for channels in eeg_by_recording.values() for ch in channels}))

    fnirs_by_recording, fnirs_detail = {}, {}
    for reference in references:
        bad, detail = detect_bad_fnirs_channels(
            reference.fnirs, reference.geo3d, rest_window(reference, cfg.window_sec),
            cfg.fnirs)
        fnirs_by_recording[reference.run] = bad
        for criterion, channels in detail.items():
            fnirs_detail[criterion] = tuple(sorted({*fnirs_detail.get(criterion, ()),
                                                    *channels}))
    bad_fnirs = tuple(sorted({ch for channels in fnirs_by_recording.values()
                              for ch in channels}))
    return BadChannels(eeg=bad_eeg, fnirs=bad_fnirs,
                       eeg_detail=eeg_detail, fnirs_detail=fnirs_detail,
                       eeg_by_recording=eeg_by_recording,
                       fnirs_by_recording=fnirs_by_recording)


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
