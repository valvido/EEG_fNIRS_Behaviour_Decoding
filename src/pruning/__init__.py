"""
Pruning: find a subject's bad EEG and fNIRS channels on the rest recording and remove
them from every run. The EEG is judged on the whole rest recording, the fNIRS on a
window of its rest period.

Example
-------
    recordings, bad_channels = prune_recordings(recordings, rest, cfg.pruning)
    # recordings: list[PrunedRecording]; bad_channels: what was removed

Bad EEG channels are interpolated, bad fNIRS channels dropped. To only see which
channels are bad, without removing them:

    bad_channels = detect_bad_channels(rest, cfg.pruning)
"""
from src.pruning.prune import detect_bad_channels, prune_recordings

__all__ = ["detect_bad_channels", "prune_recordings"]
