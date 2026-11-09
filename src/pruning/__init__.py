"""
Pruning: find a subject's bad channels and remove them from every run. The EEG is judged
on each recording on its own (synthetic: on the background recording of each group), the
fNIRS on the rest period of each recording the reader selects (`select_pruning_recordings`);
a fNIRS channel bad in any of them is dropped from every run.

Example
-------
    recordings, bad_channels = prune_recordings(recordings, references, backgrounds,
                                                cfg.pruning)
    # recordings: list[PrunedRecording]; bad_channels: what was removed

Bad EEG channels are interpolated, bad fNIRS channels dropped. To only see which
channels are bad, without removing them:

    bad_channels = detect_bad_channels(references, eeg_recordings, cfg.pruning)
"""
from src.pruning.prune import detect_bad_channels, prune_recordings

__all__ = ["detect_bad_channels", "prune_recordings"]
