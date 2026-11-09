"""
Mark the period of a recording that pruning judges the fNIRS on, for the readers.
"""
import dataclasses

import pandas as pd

from src.types import RawRecording

# Seconds left out at each end of the marked period. The period often starts or ends at
# an edge of the recording, where filtering and motion correction leave artefacts.
EDGE_S = 3.0


def mark_rest_period(recording: RawRecording, start: float, end: float) -> RawRecording:
    """
    The recording with its stim table replaced by one event marking the period from
    `start` + EDGE_S to `end` - EDGE_S, and without blocks.

    Parameters
    ----------
    recording : RawRecording
    start : float
        Start of the rest period (s), on the recording's clock.
    end : float
        End of the rest period (s).

    Returns
    -------
    RawRecording

    Raises
    ------
    ValueError
        If nothing is left of the period once its edges are left out, or the period
        starts before the fNIRS does.
    """
    start, end = start + EDGE_S, end - EDGE_S
    if end <= start:
        raise ValueError(f"Recording '{recording.run}' has no rest period: without its "
                         f"{EDGE_S} s edges it would run from {start} s to {end} s.")
    fnirs_start = float(recording.fnirs.time[0])
    if start < fnirs_start:
        raise ValueError(f"Recording '{recording.run}': the rest period starts at "
                         f"{start:.1f} s, before the fNIRS does ({fnirs_start:.1f} s).")
    rest = pd.DataFrame({"onset": [start], "duration": [end - start], "value": [1.0],
                         "trial_type": ["rest"]})
    return dataclasses.replace(recording, stim=rest, blocks=None)
