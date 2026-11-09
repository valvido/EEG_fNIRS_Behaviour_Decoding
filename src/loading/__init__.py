"""
Loading: a subject's recordings, exactly as they are on disk.

Example
-------
    check_data_format(subject)
    recordings = load_recordings(subject)   # list[RawRecording], one per run
    backgrounds = load_backgrounds(subject)   # synthetic: EEG quality judged on these
    references = select_pruning_recordings(subject, recordings, backgrounds)   # and fNIRS here
    logger.info(loading_report(subject, recordings))

One reader per dataset (src/loading/synthetic.py, src/loading/buccino.py,
src/loading/shin2017.py), chosen by
`subject.format`; see src/loading/load.py for how a dataset is added.
"""
from src.loading.load import (READERS, check_data_format, load_backgrounds, load_recordings,
                             loading_report, select_pruning_recordings)

__all__ = ["READERS", "check_data_format", "load_backgrounds", "load_recordings",
           "loading_report", "select_pruning_recordings"]
