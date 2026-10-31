"""
Select one method's hyperparameters and score the selected model: the core of both
non-nested and nested evaluation.
"""
import json
import logging
import time

import numpy as np
from joblib import Parallel, delayed, parallel_config

from src.config import UnmixingConfig
from src.types import (CombinationScore, FoldRecovery, FoldScore, JoinedRuns,
                       PreprocessedRecording, Scores, Selection)
from src.unmixing.concat import concat_runs
from src.unmixing.fit import DegenerateFit, check_blocks, fit_model
from src.unmixing.folds import leave_one_group_out, split
from src.unmixing.metrics import METRICS, fold_mean, fold_value
from src.unmixing.models import param_grid
from src.unmixing.scores import TooFewSamples, score


def select_and_score(recordings: list[PreprocessedRecording], test_group: str, method: str,
                     cfg: UnmixingConfig, hands: list[str], laplacians: tuple | None = None,
                     logger: logging.Logger | None = None, keep_folds: bool = False,
                     trial_classes: dict[str, tuple[str, ...]] | None = None
                     ) -> tuple[list[Selection], list[FoldScore], list[CombinationScore]]:
    """
    Hold out `test_group`; on the other groups, try every hyperparameter combination
    in leave-one-group-out folds; for each metric take the combination with the best
    mean over the folds, refit it on all other groups and score it on `test_group`.

    In every fold, as in the final fit, the pairs are labelled on the training runs
    and scored on the held-out runs, so a combination is selected under the same rule
    its model is judged by. For mSPoC, the EEG blocks of every data set are checked
    before any fit; a mismatch stops the run.

    Parameters
    ----------
    recordings : list[PreprocessedRecording]
        All runs of the subject.
    test_group : str
    method : str
    cfg : UnmixingConfig
    hands : list[str]
        The hands a pair is assigned to.
    laplacians : tuple | None
        (Lx, Ly), for the structured methods.
    logger : logging.Logger | None
    keep_folds : bool
        Also fit each selected combination again in every validation fold, with the
        fold's seed, and keep its components (`Selection.folds`). The selection loop
        keeps only scores, since which combination wins is known only at its end.
    trial_classes : dict[str, tuple[str, ...]] | None
        Side to its stim trial types (`SubjectConfig.trial_classes`), for the
        "task_regressors" assignment.

    Returns
    -------
    list[Selection]
        One per metric that selected a combination. Metrics that pick the same
        combination share one refit.
    list[FoldScore]
        Every combination in every validation fold.
    list[CombinationScore]
        Every combination's fold mean per metric, as compared for the selection.
    """
    logger = logger or logging.getLogger(__name__)
    dev, test = split(recordings, test_group)
    splits = leave_one_group_out(dev)
    folds = [(concat_runs(train), concat_runs(val)) for train, val in splits]
    val_groups = [val[0].group for _, val in splits]
    dev_joined, test_joined = concat_runs(dev), concat_runs(test)
    if method == "mSPoC":
        # A block mismatch is in the data, so it would fail every fit: stop before any.
        for data in [d for fold in folds for d in fold] + [dev_joined, test_joined]:
            check_blocks(data.eeg, data.fnirs)
    combos = param_grid(method, cfg)

    logger.info(f"{method}: grid search, {len(combos)} combinations x {len(folds)} folds "
                f"(validation {', '.join(val_groups)})")
    started = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        fold_scores = Parallel(n_jobs=cfg.n_jobs)(
            delayed(_fit_and_score)(method, params, train, val, cfg, hands,
                                    laplacians, task_seed(cfg.seed, 0, j, p), trial_classes)
            for j, (train, val) in enumerate(folds) for p, params in enumerate(combos))
    logger.info(f"{method}: grid search done in {time.perf_counter() - started:.0f} s")
    errors = np.array([error for _, error in fold_scores], dtype=object)
    fold_scores = np.array([scores for scores, _ in fold_scores], dtype=object)
    errors = errors.reshape(len(folds), len(combos))
    fold_scores = fold_scores.reshape(len(folds), len(combos))
    for p, params in enumerate(combos):
        failed = [e for e in errors[:, p] if e is not None]
        if failed:
            logger.warning(f"{method} {json.dumps(params, default=str)}: {len(failed)} of "
                           f"{len(folds)} folds failed, excluded from selection. First: "
                           f"{failed[0]}")
    combinations = [CombinationScore(method=method, test_group=test_group, params=params,
                                     means={m: fold_mean(m, list(fold_scores[:, p]))
                                            for m in METRICS})
                    for p, params in enumerate(combos)]

    chosen = {}
    for metric in cfg.metrics:
        means = [c.means[metric] for c in combinations]
        if np.all(np.isnan(means)):
            logger.warning(f"{method} {metric}: no combination could be scored")
            continue
        p = int(np.nanargmax(means))
        chosen.setdefault(p, []).append((metric, means[p]))

    fold_recoveries = {p: () for p in chosen}
    if keep_folds:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            refits = Parallel(n_jobs=cfg.n_jobs)(
                delayed(_refit_fold)(method, combos[p], train, val, val_groups[j], cfg,
                                     hands, laplacians, task_seed(cfg.seed, 0, j, p),
                                     trial_classes)
                for p in chosen for j, (train, val) in enumerate(folds))
        refits = iter(refits)
        fold_recoveries = {p: tuple(f for f in (next(refits) for _ in folds) if f is not None)
                           for p in chosen}
        # A refit repeats its grid fit with the same seed, and adds the pattern_r
        # diagnostics the grid search leaves out.
        for p, recovered in fold_recoveries.items():
            for f in recovered:
                fold_scores[val_groups.index(f.val_group), p] = f.scores
    landscape = [FoldScore(method=method, test_group=test_group, val_group=val_groups[j],
                           params=params, scores=fold_scores[j, p],
                           metrics={m: fold_value(m, fold_scores[j, p]) for m in METRICS},
                           error=errors[j, p])
                 for j in range(len(folds)) for p, params in enumerate(combos)]

    selections = []
    for p, metrics in chosen.items():
        params = combos[p]
        try:
            fitted = fit_model(method, params, dev_joined, cfg.n_components, laplacians,
                               seed=task_seed(cfg.seed, 1, p))
        except DegenerateFit as e:
            logger.warning(f"{e} On all groups but {test_group}, after it worked in every "
                           f"fold: {', '.join(m for m, _ in metrics)} select nothing for "
                           f"{method}.")
            continue
        try:
            scores, recovery = score(fitted, dev_joined, test_joined, cfg.assignment, hands,
                                     trial_classes=trial_classes)
        except TooFewSamples as e:
            raise TooFewSamples(f"{method} {params}, test {list(test_joined.runs)}: {e}") from e
        for metric, inner in metrics:
            selections.append(Selection(method=method, metric=metric, params=params,
                                        inner_score=inner, fitted=fitted, scores=scores,
                                        recovery=recovery, test_group=test_group,
                                        folds=fold_recoveries[p]))
    return selections, landscape, combinations


def task_seed(base: int, *index: int) -> int:
    """
    Deterministic 32-bit seed from a base seed and task indices, the same for any
    n_jobs. Python's hash() is salted per process, so it is not used.
    """
    seed = int(base) & 0xFFFFFFFF
    for i in index:
        seed = (seed * 1_000_003 + int(i) + 1) & 0xFFFFFFFF
    return seed


def _fit_and_score(method: str, params: dict, train: JoinedRuns, val: JoinedRuns,
                   cfg: UnmixingConfig, hands: list[str], laplacians: tuple | None,
                   seed: int, trial_classes: dict[str, tuple[str, ...]] | None
                   ) -> tuple[Scores | None, str | None]:
    """
    One fold of one combination: (scores, None), or (None, the error) if the fit
    fails. The error is returned, not logged, because this runs in a worker process
    whose log does not reach the run's log. TooFewSamples is raised instead: it
    stops the run.
    """
    try:
        fitted = fit_model(method, params, train, cfg.n_components, laplacians, seed)
        return score(fitted, train, val, cfg.assignment, hands, with_patterns=False,
                     trial_classes=trial_classes)[0], None
    except TooFewSamples as e:
        raise TooFewSamples(f"{method} {params}, validation {list(val.runs)}: {e}") from e
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _refit_fold(method: str, params: dict, train: JoinedRuns, val: JoinedRuns, val_group: str,
                cfg: UnmixingConfig, hands: list[str], laplacians: tuple | None,
                seed: int, trial_classes: dict[str, tuple[str, ...]] | None
                ) -> FoldRecovery | None:
    """
    One fold of one selected combination with its components; None if the fit or
    its scoring fails, as the same fold then failed in the selection loop.
    """
    try:
        fitted = fit_model(method, params, train, cfg.n_components, laplacians, seed)
        scores, recovery = score(fitted, train, val, cfg.assignment, hands,
                                 trial_classes=trial_classes)
    except TooFewSamples:
        raise
    except Exception:
        return None
    return FoldRecovery(val_group=val_group, scores=scores, recovery=recovery)
