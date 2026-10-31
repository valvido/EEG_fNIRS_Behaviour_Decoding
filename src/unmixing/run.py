"""
Unmixing for one subject: non-nested evaluation, and optionally nested evaluation.
"""
import logging

from src.config import UnmixingConfig
from src.types import PreprocessedRecording, UnmixingResult
from src.unmixing.assignment import STRATEGIES
from src.unmixing.folds import groups_of
from src.unmixing.laplacian import build_laplacians
from src.unmixing.metrics import METRICS
from src.unmixing.models import METHODS
from src.unmixing.report import selection_report
from src.unmixing.select import select_and_score


def unmix(recordings: list[PreprocessedRecording], cfg: UnmixingConfig, test_group: str,
          hands: list[str], logger: logging.Logger | None = None,
          trial_classes: dict[str, tuple[str, ...]] | None = None) -> UnmixingResult:
    """
    Select, fit and score every method of `cfg` on one subject's runs.

    Non-nested evaluation always runs: `test_group` is held out, settings are
    selected on the other groups, and the selected models are scored on it. These are
    the models classification uses; they also keep their validation-fold components,
    for plots. With `cfg.nested`, the same is repeated with every other group held
    out once, for analysis only; for `test_group` the non-nested result is reused.

    Parameters
    ----------
    recordings : list[PreprocessedRecording]
        All runs of the subject, all with ground truth or all without.
    cfg : UnmixingConfig
    test_group : str
        The subject's test group.
    hands : list[str]
        The hands a pair is assigned to (`find_hands`).
    logger : logging.Logger | None
    trial_classes : dict[str, tuple[str, ...]] | None
        Side to its stim trial types (`SubjectConfig.trial_classes`), for the
        "task_regressors" assignment.

    Returns
    -------
    UnmixingResult
    """
    logger = logger or logging.getLogger(__name__)
    check_ground_truth(recordings, hands, cfg)
    laplacians = None
    if any(METHODS[m].structured for m in cfg.methods):
        first = recordings[0]
        Lx, Ly, _ = build_laplacians(cfg.laplacian, first.eeg.channel.values,
                                     first.fnirs.channel.values, first.geo3d,
                                     first.eeg_geo3d, first.fnirs_optodes)
        laplacians = (Lx, Ly)

    landscape, combinations = [], []

    def evaluate(held_out, keep_folds=False):
        kind = "non-nested" if keep_folds else "nested"
        logger.info(f"=== Unmixing, {kind}: test group {held_out} ===")
        selections = []
        for method in cfg.methods:
            chosen, points, means = select_and_score(recordings, held_out, method, cfg, hands,
                                                     laplacians, logger, keep_folds,
                                                     trial_classes)
            for selection in chosen:
                logger.info(selection_report(selection))
            selections += chosen
            landscape.extend(points)
            combinations.extend(means)
        return selections

    selections = evaluate(test_group, keep_folds=True)
    nested = None
    if cfg.nested:
        nested = {group: selections if group == test_group else evaluate(group)
                  for group in groups_of(recordings)}
    return UnmixingResult(test_group=test_group, selections=selections, nested=nested,
                          landscape=landscape, combinations=combinations)



def find_hands(recordings: list[PreprocessedRecording], trial_classes: dict[str, tuple[str, ...]],
               hand: str) -> list[str]:
    """
    The hands a pair is assigned to: "<side>_hand" for each side of `trial_classes`
    with at least one trial in the runs, right first, narrowed to `hand`.

    Parameters
    ----------
    recordings : list[PreprocessedRecording]
    trial_classes : dict[str, tuple[str, ...]]
        Side to its stim trial types (`SubjectConfig.trial_classes`).
    hand : str
        "left" | "right" | "both" (`ExperimentConfig.hand`).

    Returns
    -------
    list[str]
        E.g. ["right_hand", "left_hand"].

    Raises
    ------
    ValueError
        If no side the experiment asks for has a trial.
    """
    trial_types = {t for rec in recordings for t in rec.stim["trial_type"]}
    wanted = ("right", "left") if hand == "both" else (hand,)
    hands = [f"{side}_hand" for side in wanted
             if trial_types & set(trial_classes.get(side, ()))]
    if not hands:
        raise ValueError(f"No trial of the sides {list(wanted)} in the runs: trial_classes "
                         f"{dict(trial_classes)}, trial types in the runs {sorted(trial_types)}.")
    return hands


def check_ground_truth(recordings: list[PreprocessedRecording], hands: list[str],
                       cfg: UnmixingConfig):
    """
    Check that the runs either all have ground truth holding every hand as a
    source, or all have none and the metrics and the assignment need none.

    Raises
    ------
    ValueError
        Naming what is missing.
    """
    with_truth = [rec.s_eeg is not None for rec in recordings]
    if any(with_truth) and not all(with_truth):
        raise ValueError("Some runs have ground truth and others do not.")
    if all(with_truth):
        missing = [h for h in hands if h not in recordings[0].s_eeg.source.values]
        if missing:
            raise ValueError(f"The ground truth has no source for {missing}; its sources "
                             f"are {list(recordings[0].s_eeg.source.values)}.")
        return
    needing = [m for m in cfg.metrics if METRICS[m].needs_ground_truth]
    if STRATEGIES[cfg.assignment.method].needs_ground_truth:
        needing.append(f"assignment {cfg.assignment.method}")
    if needing:
        raise ValueError(f"The runs have no ground truth, which {needing} need. Without it, "
                         f"select by canonical_correlation_avg and assign by motor_channels or "
                         f"task_regressors.")
