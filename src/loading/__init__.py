"""
Loading: a subject's recordings, exactly as they are on disk.

Example
-------
    check_data_format(subject)
    recordings = load_recordings(subject)   # list[RawRecording], one per run
    rest = load_rest(subject)               # the recording pruning judges channels on
    logger.info(loading_report(subject, recordings))

One reader per dataset (src/loading/synthetic.py, src/loading/buccino.py), chosen by
`subject.format`; see src/loading/load.py for how a dataset is added.
"""
from src.loading.load import (READERS, check_data_format, load_recordings, load_rest,
                             loading_report)

__all__ = ["READERS", "check_data_format", "load_recordings", "load_rest", "loading_report"]
