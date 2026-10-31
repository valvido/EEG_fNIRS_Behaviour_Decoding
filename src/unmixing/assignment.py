"""
Assign the unlabelled canonical pairs to the hands.

A fitted model gives pairs, each an EEG and an fNIRS component found together, with
no label: the model does not know which pair is which hand. A strategy decides which
pair is the right hand and which the left hand. Pairs are never split: a hand always
gets the EEG and the fNIRS component of one pair. Every strategy looks at the
training runs only.

"ground_truth" matches the pairs to the simulated sources. "motor_channels" needs no
ground truth: it gives each hand the pair whose estimated mixing lies most on that
hand's motor channels (e.g. C3 for the right hand, C4 for the left).
"task_regressors" needs no ground truth either: it gives each hand the pair whose
components follow that hand's trials best.

To add a strategy, write a function with the signature of `ground_truth` and register
it in STRATEGIES; `assignment.method` in the experiment config chooses it.
"""
import itertools
from dataclasses import dataclass
from typing import Callable

import cedalion.models.glm as glm
import numpy as np
import xarray as xr
from cedalion import units

from src.config import AssignmentConfig
from src.types import FittedModel, JoinedRuns, PairLabels
from src.unmixing.correlation import correlations
from src.unmixing.orient import haufe_pattern, spatial_filter


@dataclass(frozen=True)
class TrainingPairs:
    """
    What a strategy may look at, all on the training runs.

    Attributes
    ----------
    fitted : FittedModel
        The fitted model.
    data : JoinedRuns
        The training runs: EEG, fNIRS, stimuli, and the ground truth if there is one.
    u, v : xr.DataArray
        EEG and fNIRS components of every pair, dims (time, component).
    r_eeg, r_fnirs : xr.DataArray | None
        Correlation of every pair with every ground-truth source, dims (component,
        source). None without ground truth.
    trial_classes : dict[str, tuple[str, ...]] | None
        Side ("right", "left") to its stim trial types (`SubjectConfig.trial_classes`).
        Needed by "task_regressors" only.
    """
    fitted: FittedModel
    data: JoinedRuns
    u: xr.DataArray
    v: xr.DataArray
    r_eeg: xr.DataArray | None
    r_fnirs: xr.DataArray | None
    trial_classes: dict[str, tuple[str, ...]] | None = None


@dataclass(frozen=True)
class Strategy:
    """
    Attributes
    ----------
    assign : Callable[[TrainingPairs, list[str], AssignmentConfig], PairLabels]
        Gives each hand in the list its own pair.
    needs_ground_truth : bool
        Synthetic data only.
    """
    assign: Callable[[TrainingPairs, list[str], AssignmentConfig], PairLabels]
    needs_ground_truth: bool


def assign_pairs(pairs: TrainingPairs, hands: list[str], cfg: AssignmentConfig) -> PairLabels:
    """
    Give each hand its own pair with the strategy `cfg.method`.

    Parameters
    ----------
    pairs : TrainingPairs
    hands : list[str]
        The hands to assign, e.g. ["right_hand", "left_hand"].
    cfg : AssignmentConfig

    Returns
    -------
    PairLabels

    Raises
    ------
    ValueError
        If the strategy needs ground truth and the data has none.
    """
    strategy = STRATEGIES[cfg.method]
    if strategy.needs_ground_truth and pairs.r_eeg is None:
        raise ValueError(f"Assignment '{cfg.method}' needs ground truth; the data has none.")
    return strategy.assign(pairs, hands, cfg)


def ground_truth(pairs: TrainingPairs, hands: list[str], cfg: AssignmentConfig) -> PairLabels:
    """
    Give each hand its own pair among the first `cfg.n_top` pairs, choosing the
    assignment with the highest score

        sum over hands of eeg_weight * |r_eeg| + fnirs_weight * |r_fnirs|,

    where r is the correlation of the pair's component with that hand's ground-truth
    source. With two hands and n_top 2 this decides which of the two pairs is the
    right hand and which the left hand.

    Parameters
    ----------
    pairs : TrainingPairs
        With ground truth.
    hands : list[str]
    cfg : AssignmentConfig

    Returns
    -------
    PairLabels
    """
    return best_assignment(pairs.r_eeg, pairs.r_fnirs, hands, cfg)


def assign_by_motor_channels(pairs: TrainingPairs, hands: list[str],
                             cfg: AssignmentConfig) -> PairLabels:
    """
    Give each hand its own pair among the first `cfg.n_top` pairs, choosing the
    assignment with the highest score

        sum over hands of eeg_weight * share_eeg + fnirs_weight * share_fnirs,

    where share is the mean over the hand's motor channels (`cfg.motor_channels`) of
    |a_c| / ||a||, and a is the column of the estimated mixing matrix (Haufe pattern)
    of the pair's component in that modality, on the training runs. A pair is scored
    as a whole, from both of its components. A modality with weight 0 is not used.
    Needs no ground truth.

    Parameters
    ----------
    pairs : TrainingPairs
    hands : list[str]
    cfg : AssignmentConfig

    Returns
    -------
    PairLabels
    """
    weights = {"eeg": cfg.eeg_weight, "fnirs": cfg.fnirs_weight}
    score = sum(weight * measure_motor_share(pairs, modality, hands, cfg.motor_channels[modality])
                for modality, weight in weights.items() if weight > 0)
    return choose_best_permutation(score, hands, cfg.n_top)


def measure_motor_share(pairs: TrainingPairs, modality: str, hands: list[str],
                        channels: dict[str, tuple[str, ...]]) -> np.ndarray:
    """
    Share of each pair's estimated mixing column on each hand's motor channels.

    The column a of a component is its Haufe pattern on the training runs; its share
    on a hand is the mean over the hand's channels of |a_c| / ||a||. The sign of a
    component is arbitrary and does not count, and a component that is strong on
    every channel does not get a high share.

    Parameters
    ----------
    pairs : TrainingPairs
    modality : str
        "eeg" or "fnirs".
    hands : list[str]
    channels : dict[str, tuple[str, ...]]
        Hand to its motor channels in this modality. Channels missing from the data
        (e.g. pruned) are skipped.

    Returns
    -------
    np.ndarray
        Dims (component, hand), the hands in the order of `hands`. NaN for a
        component whose filter has no variance on the data.

    Raises
    ------
    ValueError
        If none of a hand's channels is in the data.
    """
    data = pairs.data.eeg if modality == "eeg" else pairs.data.fnirs
    pattern = haufe_pattern(spatial_filter(pairs.fitted.model, modality), data)
    share = np.abs(pattern) / np.sqrt((pattern ** 2).sum("channel"))
    columns = []
    for hand in hands:
        present = [c for c in channels.get(hand, ()) if c in share.channel.values]
        if not present:
            raise ValueError(f"None of the {modality} motor channels "
                             f"{list(channels.get(hand, ()))} of {hand} is in the data.")
        columns.append(share.sel(channel=present).mean("channel").values)
    return np.stack(columns, axis=1)


def assign_by_task_regressors(pairs: TrainingPairs, hands: list[str],
                              cfg: AssignmentConfig) -> PairLabels:
    """
    Give each hand its own pair among the first `cfg.n_top` pairs, choosing the
    assignment with the highest score

        sum over hands of eeg_weight * |r(EEG component, EEG reference)|
                        + fnirs_weight * |r(fNIRS component, fNIRS reference)|,

    where a hand's references are its trials on the training runs as a boxcar (EEG)
    and as the boxcar convolved with a gamma HRF (`cfg.fnirs_hrf`, fNIRS). A hand's
    trials are the trial types of its side in `pairs.trial_classes`. Needs no ground
    truth.

    Parameters
    ----------
    pairs : TrainingPairs
        With `trial_classes`.
    hands : list[str]
        "<side>_hand" names, e.g. ["right_hand", "left_hand"].
    cfg : AssignmentConfig

    Returns
    -------
    PairLabels

    Raises
    ------
    ValueError
        If `pairs` has no `trial_classes`.
    """
    if pairs.trial_classes is None:
        raise ValueError("Assignment 'task_regressors' needs the subject's trial_classes.")
    hand_trials = {hand: pairs.trial_classes.get(hand.removesuffix("_hand"), ())
                   for hand in hands}
    hrf = glm.Gamma(**{name: value * units.s for name, value in cfg.fnirs_hrf.items()})
    reference_eeg = build_task_references(pairs.data, hand_trials, glm.DiracDelta())
    reference_fnirs = build_task_references(pairs.data, hand_trials, hrf)
    return best_assignment(correlations(pairs.u, reference_eeg),
                           correlations(pairs.v, reference_fnirs), hands, cfg)


def build_task_references(data: JoinedRuns, hand_trials: dict[str, tuple[str, ...]],
                          basis: glm.TemporalBasisFunction) -> xr.DataArray:
    """
    Task time course of each hand on the fNIRS time axis of `data`: the trials of its
    trial types convolved with `basis` (cedalion's `hrf_regressors`), every trial with
    amplitude 1, so only the cue timing goes in.

    Parameters
    ----------
    data : JoinedRuns
    hand_trials : dict[str, tuple[str, ...]]
        Hand to its stim trial types.
    basis : glm.TemporalBasisFunction
        E.g. glm.DiracDelta() for a boxcar over each trial.

    Returns
    -------
    xr.DataArray
        Dims (time, source), one source per hand. All zeros for a hand without trials
        in `data`; it then correlates as NaN and scores 0.
    """
    n_time = data.fnirs.sizes["time"]
    # cedalion builds regressors for a time series with a channel and a chromophore
    # dimension and a "samples" coordinate; this one only lends them its time axis.
    grid = xr.DataArray(np.zeros((1, 1, n_time)), dims=("channel", "chromo", "time"),
                        coords={"channel": ["task"], "chromo": ["HbO"],
                                "time": data.fnirs.time.values,
                                "samples": ("time", np.arange(n_time))})
    grid = grid.assign_coords(time=grid.time.assign_attrs(units="s"))
    stim = data.stim.assign(value=1.0)
    regressors = glm.design_matrix.hrf_regressors(grid, stim, basis).common.sel(chromo="HbO")
    available = set(regressors.regressor.values)
    columns = []
    for trial_types in hand_trials.values():
        names = [f"HRF {t}" for t in trial_types if f"HRF {t}" in available]
        columns.append(regressors.sel(regressor=names).sum("regressor"))
    return (xr.concat(columns, dim=xr.DataArray(list(hand_trials), dims="source"))
            .transpose("time", "source"))


def best_assignment(r_eeg: xr.DataArray, r_fnirs: xr.DataArray, hands: list[str],
                    cfg: AssignmentConfig) -> PairLabels:
    """
    The assignment of hands to distinct pairs among the first `cfg.n_top` with the
    highest summed `match_weights`. On training correlations this is the
    "ground_truth" strategy; on held-out correlations it is the assignment that
    would have been best there.

    Parameters
    ----------
    r_eeg, r_fnirs : xr.DataArray
        Correlation of every pair with every source, dims (component, source).
    hands : list[str]
    cfg : AssignmentConfig

    Returns
    -------
    PairLabels
    """
    weight = match_weights(r_eeg.sel(source=hands), r_fnirs.sel(source=hands), cfg)
    return choose_best_permutation(weight, hands, cfg.n_top)


def choose_best_permutation(score: np.ndarray, hands: list[str], n_top: int) -> PairLabels:
    """
    The assignment of hands to distinct pairs among the first `n_top` with the
    highest summed score.

    Parameters
    ----------
    score : np.ndarray
        Dims (component, hand), the hands in the order of `hands`. NaN counts 0.
    hands : list[str]
    n_top : int

    Returns
    -------
    PairLabels
    """
    n_top = min(n_top, score.shape[0])
    if n_top < len(hands):
        raise ValueError(f"{len(hands)} hands need at least as many pairs, got {n_top}.")
    score = np.nan_to_num(score)
    best, best_score = None, -np.inf
    for chosen in itertools.permutations(range(n_top), len(hands)):
        total = sum(score[p, j] for j, p in enumerate(chosen))
        if total > best_score:
            best, best_score = chosen, total
    return PairLabels(pairs={h: int(p) for h, p in zip(hands, best)}, score=float(best_score))


STRATEGIES = {
    "ground_truth": Strategy(ground_truth, needs_ground_truth=True),
    "motor_channels": Strategy(assign_by_motor_channels, needs_ground_truth=False),
    "task_regressors": Strategy(assign_by_task_regressors, needs_ground_truth=False),
}


def best_pairs(r_eeg: xr.DataArray, r_fnirs: xr.DataArray, cfg: AssignmentConfig) -> dict[str, int]:
    """
    Diagnostic only: per ground-truth source, the pair out of all fitted pairs with the
    highest weighted |r| on the training data, each source on its own.
    """
    weight = match_weights(r_eeg, r_fnirs, cfg)
    return {str(s): int(np.argmax(np.nan_to_num(weight[:, j], nan=-np.inf)))
            for j, s in enumerate(r_eeg.source.values)}


def match_weights(r_eeg: xr.DataArray, r_fnirs: xr.DataArray, cfg: AssignmentConfig) -> np.ndarray:
    """(component, source): eeg_weight |r_eeg| + fnirs_weight |r_fnirs|; NaN counts 0."""
    r_eeg = np.nan_to_num(np.abs(r_eeg.transpose("component", "source").values))
    r_fnirs = np.nan_to_num(np.abs(r_fnirs.transpose("component", "source").values))
    return cfg.eeg_weight * r_eeg + cfg.fnirs_weight * r_fnirs
