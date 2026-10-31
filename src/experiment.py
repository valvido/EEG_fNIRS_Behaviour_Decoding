"""
The composition root: the only place that knows the order of the pipeline steps.

Each step is a module with a public API. `run` calls the steps in order, gives each
one its own config section, and hands each step's output to the next; no step calls
another. Steps are added here one at a time, as the modules are refactored.
"""
import logging
from datetime import datetime

from src.classification import classify
from src.config import ExperimentConfig, SubjectConfig, check_subject_fits
from src.loading import check_data_format, load_recordings, load_rest, loading_report
from src.preprocessing import (alignment_report, alignment_table, preprocess_recordings,
                               split_into_blocks)
from src.pruning import prune_recordings
from src.results import save_run
from src.types import Geometry, RunResult
from src.unmixing import check_config as check_unmixing
from src.unmixing import find_hands, unmix
from src.visualization import save_figures


def run(cfg: ExperimentConfig, subject: SubjectConfig,
        logger: logging.Logger) -> RunResult:
    """
    Run one experiment on one subject.

    Parameters
    ----------
    cfg : ExperimentConfig
        What to run.
    subject : SubjectConfig
        Which data to run it on.
    logger : logging.Logger

    Returns
    -------
    RunResult
        Everything the run produced, built from the steps' return values. It is also
        written to results/runs/<experiment>/<dataset>/<started>/: the data to
        results/, the figures to figures/.
    """
    started = datetime.now().isoformat(timespec="seconds")
    # Checks that need no data, before reading any of it: the subject's data can be
    # read, the unmixing methods exist and fit the preprocessing, and the subject
    # fits the experiment
    check_data_format(subject)
    check_unmixing(cfg.unmixing, cfg.preprocessing.eeg_feature)
    check_subject_fits(cfg, subject)
    logger.info(f"Experiment '{cfg.name}' on subject '{subject.subject_id}' "
                f"({subject.format}, groups={list(subject.groups)}, "
                f"test_group={subject.test_group})")

    # 1. Loading: every run of the subject's groups, plus the rest recording that
    #    pruning judges channel quality on. Nothing is processed yet.
    recordings = load_recordings(subject)
    rest = load_rest(subject)
    logger.info(loading_report(subject, recordings))

    # 2. Pruning: bad channels found once on the rest recording, then interpolated
    #    (EEG) or dropped (fNIRS) in every run. The rest recording is not used after.
    #    A bad C3 or C4 stops the run: the subject cannot be used.
    notch_hz = ", ".join(f"{f:g}" for f in cfg.pruning.eeg.line_freqs)
    logger.warning(f"Pruning notches the EEG at {notch_hz} Hz before bad-channel detection "
                   f"(eeg.line_freqs in the pruning YAML that the experiment's 'pruning:' "
                   f"names). It must match the mains frequency of the recording.")
    recordings, bad_channels = prune_recordings(recordings, rest, cfg.pruning)
    logger.info(f"Pruning: EEG interpolated {list(bad_channels.eeg)}; "
                f"fNIRS dropped {len(bad_channels.fnirs)} channels")

    # 3. Preprocessing: per run, the EEG feature and HbO on one time axis, cut to the
    #    stimulus window and z-scored, with the ground truth transformed alike. A run
    #    that is badly cut or aligned stops here; the others get one line each.
    recordings = preprocess_recordings(recordings, cfg.preprocessing)
    logger.info(f"Preprocessing: {len(recordings)} runs, EEG feature "
                f"'{cfg.preprocessing.eeg_feature}'")
    for recording in recordings:
        logger.info(alignment_report(recording))
    # A session recorded in one piece becomes one run per block (its CV groups), cut
    # only now so that no filter has an edge at a block boundary.
    if any(recording.blocks is not None for recording in recordings):
        recordings = split_into_blocks(recordings, subject.groups)
        logger.info(f"Split into blocks: {[f'{r.group}/{r.run}' for r in recordings]}")

    # 4. Unmixing: each method's settings selected on all groups but the test group,
    #    the selected model scored on the test group (and, with nested: true, the same
    #    with every group held out once). A pair is assigned to each hand whose side
    #    has trials in the stim tables.
    hands = find_hands(recordings, subject.trial_classes, cfg.hand)
    logger.info(f"Hands: {hands} (trial_classes {dict(subject.trial_classes)})")
    unmixing = unmix(recordings, cfg.unmixing, subject.test_group, hands, logger,
                     subject.trial_classes)

    # 5. Classification, if the experiment has a classification section: the trials
    #    of each held-out group classified with rLDA from every selected model's
    #    components, the classifier fit on the trials the model was fit on. Trial
    #    types are merged into sides by the subject's trial_classes, as for the hands.
    classification = None
    if cfg.classification is not None:
        classification = classify(recordings, unmixing, cfg.classification, logger,
                                  subject.trial_classes)

    result = RunResult(cfg=cfg, subject=subject, bad_channels=bad_channels,
                       alignment=alignment_table(recordings),
                       geometry=Geometry.of(recordings[0]), unmixing=unmixing,
                       inputs=recordings, started=started,
                       finished=datetime.now().isoformat(timespec="seconds"),
                       classification=classification)

    # 6. Results: everything the run produced, as data, to <run>/results/. With
    #    save_inputs, the preprocessed runs also go to data/preprocessed/.
    run_dir = save_run(result)
    logger.info(f"Results written to {run_dir / 'results'}")

    # 7. Figures: every standard figure to <run>/figures/, drawn from the result in
    #    memory, preprocessed runs included, so they never depend on save_inputs.
    save_figures(result, run_dir / "figures", logger)
    logger.info(f"Figures written to {run_dir / 'figures'}")
    return result
