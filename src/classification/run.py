"""
Classification for one subject: the trials of each held-out group, classified from
the components of every selected model.
"""
import logging

import numpy as np

from src.classification.features import TrialFeatures, epoch_span, trial_features
from src.classification.report import classification_report
from src.classification.rlda import confusion_counts, predict_rlda
from src.config import ClassificationConfig
from src.types import (ClassificationResult, ClassificationScore, PreprocessedRecording,
                       Selection, UnmixingResult)
from src.unmixing import split


def classify(recordings: list[PreprocessedRecording], unmixing: UnmixingResult,
             cfg: ClassificationConfig, logger: logging.Logger | None = None,
             trial_classes: dict[str, tuple[str, ...]] | None = None) -> ClassificationResult:
    """
    Classify the trials of every selection's held-out group with rLDA, once per
    feature set.

    A selection's model unmixes every run, and its pairs keep the labels they got on
    the model's training runs. The classifier is fit on the trials of those training
    runs and scored on the trials of the held-out group, so neither the model nor
    the classifier ever saw the trials it is scored on. Nested selections are
    classified the same way; for the test group the non-nested scores are reused.

    Parameters
    ----------
    recordings : list[PreprocessedRecording]
        All runs of the subject, as unmixing got them.
    unmixing : UnmixingResult
    cfg : ClassificationConfig
    logger : logging.Logger | None
    trial_classes : dict[str, tuple[str, ...]] | None
        Side to the stim trial types that make up its class
        (`SubjectConfig.trial_classes`), e.g. hand and arm trials together as one
        side. None keeps every trial type as its own class.

    Returns
    -------
    ClassificationResult

    Raises
    ------
    ValueError
        If a class has no training or no test trials.
    """
    logger = logger or logging.getLogger(__name__)
    warn_if_epochs_overlap(recordings, cfg, logger)
    logger.info(f"=== Classification: test group {unmixing.test_group} ===")
    scores = classify_selections(recordings, unmixing.selections, cfg, trial_classes, logger)
    nested = None
    if unmixing.nested:
        nested = {}
        for group, selections in unmixing.nested.items():
            if group == unmixing.test_group:
                nested[group] = scores
                continue
            logger.info(f"=== Classification, nested: test group {group} ===")
            nested[group] = classify_selections(recordings, selections, cfg, trial_classes,
                                                logger)
    return ClassificationResult(test_group=unmixing.test_group, scores=scores, nested=nested)


def classify_selections(recordings: list[PreprocessedRecording], selections: list[Selection],
                        cfg: ClassificationConfig,
                        trial_classes: dict[str, tuple[str, ...]] | None,
                        logger: logging.Logger) -> list[ClassificationScore]:
    """One score per selection and feature set, in that order."""
    features = {}
    scores = []
    for selection in selections:
        # Metrics that selected the same setting share one model, and so its features.
        key = id(selection.fitted)
        if key not in features:
            dev, test = split(recordings, selection.test_group)
            pairs = selection.scores.labels.pairs
            features[key] = (trial_features(selection.fitted, pairs, dev, cfg, trial_classes),
                             trial_features(selection.fitted, pairs, test, cfg, trial_classes))
        train, test = features[key]
        check_every_class_has_trials(train, test, selection, cfg)
        chosen = [score_feature_set(selection, feature_set, train, test, cfg)
                  for feature_set in cfg.feature_sets]
        logger.info(classification_report(chosen))
        scores += chosen
    return scores


def score_feature_set(selection: Selection, feature_set: str, train: TrialFeatures,
                      test: TrialFeatures, cfg: ClassificationConfig) -> ClassificationScore:
    predicted = predict_rlda(train.of(feature_set), train.labels, test.of(feature_set))
    return ClassificationScore(
        method=selection.method, metric=selection.metric, test_group=selection.test_group,
        feature_set=feature_set, classes=cfg.classes,
        confusion=confusion_counts(test.labels, predicted, cfg.classes),
        n_train=len(train.labels), dropped_train=train.n_dropped, dropped_test=test.n_dropped)


def check_every_class_has_trials(train: TrialFeatures, test: TrialFeatures,
                                 selection: Selection, cfg: ClassificationConfig):
    """
    Raises
    ------
    ValueError
        If a class has no training or no test trials, naming the class.
    """
    start, end = epoch_span(cfg)
    for name, trials in (("training", train), ("test", test)):
        missing = [c for c in cfg.classes if c not in set(trials.labels)]
        if not missing:
            continue
        hint = ""
        if cfg.rest_class in missing:
            hint = (f" Rest trials come from the stim table, or else from gaps between "
                    f"trials at least one epoch ({end - start:g} s) long.")
        raise ValueError(f"Classification of {selection.method} by {selection.metric}, test "
                         f"group {selection.test_group}: no {name} trials of class(es) "
                         f"{missing}.{hint}")


def warn_if_epochs_overlap(recordings: list[PreprocessedRecording], cfg: ClassificationConfig,
                           logger: logging.Logger):
    """
    Warn once if the epochs reach past the shortest gap between two onsets of a run:
    then a trial's last features include the start of the next trial.
    """
    gaps = np.concatenate([np.diff(np.sort(rec.stim.onset.values)) for rec in recordings])
    if gaps.size == 0:
        return
    _, end = epoch_span(cfg)
    shortest = float(gaps.min())
    if end > shortest:
        logger.warning(f"Classification: the epochs reach {end:g} s after onset, past the "
                       f"shortest gap between two onsets ({shortest:.1f} s); a trial's last "
                       f"features then include the start of the next trial.")
