"""
Preprocessing: turn each pruned run into the signals the models are fit on.

Example
-------
    preprocessed = preprocess_recordings(recordings, cfg.preprocessing)
    # list[PreprocessedRecording], one per run

Per run: cleaned EEG feature and band-passed HbO, cut to the stimulus window, on one
time axis, z-scored. The ground-truth sources go through the same transforms.
The models are always fit on these full recordings; the input is never cut
into trials. A run that is badly cut or aligned stops the
pipeline (cfg.preprocessing.checks).

    for recording in preprocessed:
        print(alignment_report(recording))   # window, spans

A session recorded in one piece (`blocks` set by its reader) is preprocessed as a
whole and then split into one run per block:

    preprocessed = split_into_blocks(preprocessed, subject.groups)
"""
from src.preprocessing.check import alignment_report, alignment_table
from src.preprocessing.preprocess import preprocess_recordings
from src.preprocessing.split import split_into_blocks

__all__ = ["alignment_report", "alignment_table", "preprocess_recordings",
           "split_into_blocks"]
