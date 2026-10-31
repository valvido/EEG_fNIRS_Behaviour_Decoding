"""
Select the part of the rest recording that fNIRS channel quality is judged on.
"""
from src.types import RawRecording


def rest_window(rest: RawRecording, window_sec: float) -> tuple[float, float]:
    """
    The window fNIRS channel quality is judged on: the first `window_sec` seconds of
    the rest period. The window ends earlier if the rest period or the recorded data
    ends first; nothing before or after the rest period is used. The EEG is not cut
    to this window, it is judged on the whole rest recording.

    Parameters
    ----------
    rest : RawRecording
        The rest recording, with exactly one stim event marking the rest period.
        Its onset is on the shared clock (`eeg.times`).
    window_sec : float
        Seconds of rest to keep, from the rest onset.

    Returns
    -------
    start : float
        Window start (s), the rest onset.
    end : float
        Window end (s).
    """
    if len(rest.stim) != 1:
        raise ValueError(f"Rest recording '{rest.run}' must have exactly one stim event "
                         f"marking the rest period, found {len(rest.stim)}.")
    rest_onset = float(rest.stim["onset"].iloc[0])
    rest_end = rest_onset + float(rest.stim["duration"].iloc[0])

    end = min(rest_onset + window_sec, rest_end, rest.eeg.times[-1])
    return rest_onset, end
