"""
Score a fitted model: against the simulated ground truth where there is one, and
by the canonical correlation of its pairs always.
"""
import numpy as np
import pandas as pd
import xarray as xr

from src.config import AssignmentConfig
from src.types import (FittedModel, JoinedRuns, PairLabels, PreprocessedRecording, Recovery,
                       Scores)
from src.unmixing.concat import concat_runs
from src.unmixing.correlation import MIN_SAMPLES, correlate, correlations
from src.unmixing.fit import shift_samples, shift_to_eeg_time, transform
from src.unmixing.models import METHODS
from src.unmixing.orient import haufe_pattern, spatial_filter
from src.unmixing.assignment import TrainingPairs, assign_pairs, best_pairs

class TooFewSamples(ValueError):
    """
    A labelled pair has fewer than MIN_SAMPLES time points to be correlated on: the
    fit or the data is broken. Stops the run instead of being skipped like a failed
    fit, since skipping would let the combination be judged on its other folds.
    """


def components(fitted: FittedModel,
               data: JoinedRuns) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    The components of `data`, the EEG ones both on the EEG's time and as the model
    pairs them with the fNIRS ones.

    For temporal models the EEG components are set to NaN where the time embedding
    does not see the EEG of one run: cedalion embeds the EEG with past copies
    x(t - tau), so the first max(tau) after the start b of every run reaches into
    the previous run, or into the zero padding before the first. That window is
    [b, b + max(tau)) in `u_paired`, and [b - tau_k, b - tau_k + max(tau)) in `u`
    for component k, which is shifted back by tau_k.

    Returns
    -------
    u : xr.DataArray
        EEG components on the time of the EEG they came from (`shift_to_eeg_time`):
        what is compared with the EEG ground truth, plotted and classified.
    u_paired : xr.DataArray
        EEG components as the model pairs them with `v` (`transform`): what is
        compared with `v`. The same as `u` for non-temporal models.
    v : xr.DataArray
        fNIRS components.
    All dims (time, component), on the fNIRS time points of `data`.
    """
    u_paired, v = transform(fitted, data)
    u = shift_to_eeg_time(fitted, u_paired)
    if not METHODS[fitted.method].temporal:
        return u, u_paired, v

    time = data.fnirs.time.values
    reach = shift_samples(time, float(np.max(fitted.params["time_shifts"])))
    run_start_samples = [int(np.searchsorted(time, start)) for start in data.run_starts]
    lags = [shift_samples(time, shift) for shift in np.atleast_1d(fitted.model.optimal_shift)]
    paired_valid = ~_embedding_reaches_out(len(time), run_start_samples, reach, shift_back=0)
    u_valid = np.stack([~_embedding_reaches_out(len(time), run_start_samples, reach, lag)
                        for lag in lags], axis=1)
    return u.where(u_valid), u_paired.where(paired_valid[:, None]), v


def _embedding_reaches_out(n_time: int, run_start_samples: list[int], reach: int,
                           shift_back: int) -> np.ndarray:
    """
    Boolean (n_time,): True within `reach` samples after each run start b, moved
    back by `shift_back` samples, i.e. on [b - shift_back, b - shift_back + reach).
    """
    sample = np.arange(n_time)
    out = np.zeros(n_time, dtype=bool)
    for b in run_start_samples:
        out |= (sample >= b - shift_back) & (sample < b - shift_back + reach)
    return out


def unmix_run(fitted: FittedModel,
              recording: PreprocessedRecording) -> tuple[xr.DataArray, xr.DataArray, pd.DataFrame]:
    """
    `components` of one run on its own, so no time shift reaches into another run.

    The run is placed on the time axis `concat_runs` gives it, so the components and
    the stimuli are returned together, on that axis.

    Returns
    -------
    u, v : xr.DataArray
        Dims (time, component).
    stim : pd.DataFrame
        The run's stimuli, columns `JOINED_STIM_COLUMNS`.
    """
    data = concat_runs([recording])
    u, _, v = components(fitted, data)
    return u, v, data.stim


def score(fitted: FittedModel, train: JoinedRuns, test: JoinedRuns,
          assignment: AssignmentConfig, hands: list[str],
          with_patterns: bool = True,
          trial_classes: dict[str, tuple[str, ...]] | None = None
          ) -> tuple[Scores, Recovery | None]:
    """
    Label the pairs on the training runs, then score those pairs on the test runs.

    Without ground truth, only the labels, `canonical_r` and the patterns can be
    computed; the scores against the ground truth are NaN and `best_pairs` is empty.

    Parameters
    ----------
    fitted : FittedModel
        Fitted on `train`.
    train, test : JoinedRuns
        Both with ground truth, or both without.
    assignment : AssignmentConfig
    hands : list[str]
        The hands to give a pair each, e.g. ["right_hand", "left_hand"].
    with_patterns : bool
        Compute the Haufe patterns on `test`, for the plots and the pattern_r
        diagnostics. The grid search leaves them out: selection never uses them, and
        only the selected models are plotted.
    trial_classes : dict[str, tuple[str, ...]] | None
        Side to its stim trial types (`SubjectConfig.trial_classes`), for the
        "task_regressors" assignment.

    Returns
    -------
    Scores
        pattern_r_eeg and pattern_r_fnirs are NaN without `with_patterns`.
    Recovery | None
        The test components and patterns, and the ground truth the scores were
        computed from. None without `with_patterns`.

    Raises
    ------
    TooFewSamples
        If a labelled pair has fewer than MIN_SAMPLES time points on `test` to be
        correlated between its EEG and fNIRS component, or with its source.
    """
    u_train, _, v_train = components(fitted, train)
    r_train_eeg, r_train_fnirs = correlations_with_truth(u_train, v_train, train)
    labels = assign_pairs(TrainingPairs(fitted=fitted, data=train, u=u_train, v=v_train,
                                        r_eeg=r_train_eeg, r_fnirs=r_train_fnirs,
                                        trial_classes=trial_classes),
                          hands, assignment)

    u, u_paired, v = components(fitted, test)
    _check_samples(u, u_paired, v, test, labels)
    r_eeg, r_fnirs = correlations_with_truth(u, v, test)
    if with_patterns:
        pattern_eeg, pattern_fnirs = patterns(fitted, test)
        eeg_feature = METHODS[fitted.method].eeg_feature
        pattern_r = lambda pattern, modality: {
            s: pattern_correlation(pattern.sel(component=p), s, test, modality, eeg_feature)
            for s, p in labels.pairs.items()}
        pattern_r_eeg, pattern_r_fnirs = pattern_r(pattern_eeg, "eeg"), pattern_r(pattern_fnirs, "fnirs")
        recovery = Recovery(u=u, v=v, s_eeg=test.s_eeg, s_fnirs=test.s_fnirs,
                            run_starts=test.run_starts, runs=test.runs, stim=test.stim,
                            pattern_eeg=pattern_eeg, pattern_fnirs=pattern_fnirs,
                            r_train_eeg=r_train_eeg, r_train_fnirs=r_train_fnirs,
                            r_test_eeg=r_eeg, r_test_fnirs=r_fnirs,
                            gain_eeg=test.eeg_gain, gain_fnirs=test.fnirs_gain)
    else:
        pattern_r_eeg = pattern_r_fnirs = {s: np.nan for s in labels.pairs}
        recovery = None
    scores = Scores(
        labels=labels,
        isi_eeg=isi(fitted, labels, test, "eeg"),
        isi_fnirs=isi(fitted, labels, test, "fnirs"),
        canonical_r={s: canonical_r(u_paired, v, p) for s, p in labels.pairs.items()},
        pattern_r_eeg=pattern_r_eeg,
        pattern_r_fnirs=pattern_r_fnirs,
        **recovery_scores(labels, r_eeg, r_fnirs, r_train_eeg, r_train_fnirs, assignment),
    )
    return scores, recovery


def correlations_with_truth(u: xr.DataArray, v: xr.DataArray, data: JoinedRuns
                            ) -> tuple[xr.DataArray | None, xr.DataArray | None]:
    """
    `correlations` of the EEG components `u` with the EEG ground truth and of the
    fNIRS components `v` with the fNIRS ground truth; (None, None) without ground
    truth.
    """
    if data.s_eeg is None:
        return None, None
    return correlations(u, data.s_eeg), correlations(v, data.s_fnirs)


def recovery_scores(labels: PairLabels, r_eeg: xr.DataArray | None,
                    r_fnirs: xr.DataArray | None, r_train_eeg: xr.DataArray | None,
                    r_train_fnirs: xr.DataArray | None, assignment: AssignmentConfig) -> dict:
    """
    The fields of `Scores` that compare the labelled pairs with the ground truth:
    r_eeg, r_fnirs, best_pairs, leak_eeg, leak_fnirs. Without ground truth (r_eeg
    None) they are NaN, and best_pairs is empty.
    """
    if r_eeg is None:
        nan = {s: np.nan for s in labels.pairs}
        return {"r_eeg": nan, "r_fnirs": dict(nan), "best_pairs": {},
                "leak_eeg": dict(nan), "leak_fnirs": dict(nan)}
    pick = lambda r, source, pair: abs(float(r.sel(component=pair, source=source)))
    other = _other_pairs(labels)
    leak = lambda r, s: pick(r, s, other[s]) if s in other else np.nan
    best = best_pairs(r_train_eeg, r_train_fnirs, assignment)
    return {"r_eeg": {s: pick(r_eeg, s, p) for s, p in labels.pairs.items()},
            "r_fnirs": {s: pick(r_fnirs, s, p) for s, p in labels.pairs.items()},
            "best_pairs": {s: (p, pick(r_eeg, s, p), pick(r_fnirs, s, p))
                           for s, p in best.items()},
            "leak_eeg": {s: leak(r_eeg, s) for s in labels.pairs},
            "leak_fnirs": {s: leak(r_fnirs, s) for s in labels.pairs}}


def _other_pairs(labels: PairLabels) -> dict[str, int]:
    """With two sources, each source's other pair: the pair assigned to the other
    source. Empty with one source."""
    if len(labels.pairs) != 2:
        return {}
    (a, pair_a), (b, pair_b) = labels.pairs.items()
    return {a: pair_b, b: pair_a}


def canonical_r(u_paired: xr.DataArray, v: xr.DataArray, pair: int) -> float:
    """
    Signed correlation between a pair's EEG and fNIRS component: the canonical
    correlation as the model found it on this data. The model fits the pair to move
    together, so a pair that comes out negative on held-out data moved against
    itself there; that must count against it.

    Parameters
    ----------
    u_paired : xr.DataArray
        EEG components as the model pairs them with `v` (`components`), not shifted
        to the EEG's time.
    v : xr.DataArray
        fNIRS components.
    pair : int
    """
    return correlate(u_paired.sel(component=pair).values, v.sel(component=pair).values)


def patterns(fitted: FittedModel, data: JoinedRuns) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Haufe patterns of the fitted filters on `data`, the held-out runs: the filters
    are fixed, only the data covariance comes from `data`. Each pattern has the sign
    the model gave its filter.

    Returns
    -------
    pattern_eeg, pattern_fnirs : xr.DataArray
        Dims (channel, component).
    """
    return (haufe_pattern(spatial_filter(fitted.model, "eeg"), data.eeg),
            haufe_pattern(spatial_filter(fitted.model, "fnirs"), data.fnirs))


def pattern_correlation(pattern: xr.DataArray, source: str, data: JoinedRuns,
                        modality: str, eeg_feature: str) -> float:
    """
    |r| of one pair's pattern (from `patterns`) with a source's ground-truth mixing,
    over the channels both have. The mixing is taken as in `isi`, for the model's
    EEG feature `eeg_feature` ("bandpower" or "bandpassed"), scaled like the data the
    pattern was computed on.

    A diagnostic only: it is reported but never used to select a model.

    Returns
    -------
    float
        NaN without ground truth.
    """
    if data.ground_truth is None:
        return np.nan
    A = ground_truth_mixing(data.ground_truth, modality, _gain(data, modality), eeg_feature)
    A = A.sel(source=source)
    pattern, A = xr.align(pattern.drop_vars("component"), A.drop_vars("source"), join="inner")
    return abs(correlate(pattern.values, A.values, min_samples=3))


def isi(fitted: FittedModel, labels: PairLabels, data: JoinedRuns, modality: str) -> float:
    """
    Amari ISI of the labelled pairs' filters W against the ground-truth mixing A of
    the features the model is fit on (`ground_truth_mixing`): G = |W' A|, rows the
    pairs and columns the sources in label order. 0 means each filter sees exactly
    one source, 1 that every filter sees all sources equally.
    Each row of G is divided by its maximum first: a filter's scale is arbitrary, and
    without this the column terms compare filters of different scale, so a weak
    filter seeing the same source as a strong one would look separated. The mixing is
    taken for the data's channels by name, scaled by the data's gain: the filters act
    on z-scored data, whose mixing is A / std per channel. For EEG band power, A is
    the square of the EEG field. For fNIRS it is the HbO mixing at the first
    wavelength; the wavelengths differ by one constant factor, which the ISI does not
    depend on.

    Returns
    -------
    float
        NaN with fewer than two sources or without ground truth.
    """
    gt = data.ground_truth
    if gt is None or len(labels.pairs) < 2:
        return np.nan
    W = spatial_filter(fitted.model, modality)
    A = ground_truth_mixing(gt, modality, _gain(data, modality),
                            METHODS[fitted.method].eeg_feature)
    sources = list(labels.pairs)
    A = A.sel(channel=W.channel, source=sources).transpose("channel", "source").values
    W = W.sel(component=[labels.pairs[s] for s in sources]).values
    G = np.abs(W.T @ A)
    if not np.isfinite(G).all() or not (G.max(axis=1) > 0).all():
        return np.nan
    G = G / G.max(axis=1, keepdims=True)
    n = len(sources)
    rows = (G.sum(axis=1) / G.max(axis=1) - 1.0).sum()
    cols = (G.sum(axis=0) / G.max(axis=0) - 1.0).sum()
    return float((rows + cols) / (2 * n * (n - 1)))


def ground_truth_mixing(gt, modality: str, gain: xr.DataArray | None = None,
                        eeg_feature: str | None = None) -> xr.DataArray:
    """
    Ground-truth mixing of the features the models are fit on, dims (channel, source).

    - EEG band power: A_eeg ** 2. A source adds band power to a channel in proportion
      to the square of its field there, whatever the sign of the field.
    - Band-passed EEG (mSPoC): A_eeg.
    - fNIRS: A_fnirs of HbO at the first wavelength.

    Parameters
    ----------
    gt : GroundTruth | Geometry
        Anything with A_eeg and A_fnirs.
    modality : str
        "eeg" or "fnirs".
    gain : xr.DataArray | None
        Per-channel factor the data was scaled by (`JoinedRuns.eeg_gain`); the mixing
        is multiplied by it and kept for its channels only. None leaves it as is.
    eeg_feature : str | None
        The EEG feature of the model, "bandpower" or "bandpassed"
        (`Method.eeg_feature`). Needed for EEG, not used for fNIRS.

    Raises
    ------
    ValueError
        For EEG without `eeg_feature`.
    """
    if modality == "eeg" and eeg_feature is None:
        raise ValueError("The EEG mixing depends on the EEG feature: pass eeg_feature.")
    A = gt.A_eeg if modality == "eeg" else gt.A_fnirs.sel(chromo="HbO").isel(wavelength=0)
    if getattr(A.pint, "units", None) is not None:
        A = A.pint.dequantify()
    A = A.drop_vars([c for c in A.coords if c not in ("channel", "source")])
    if modality == "eeg" and eeg_feature == "bandpower":
        A = A ** 2
    return A if gain is None else A * gain


def _gain(data: JoinedRuns, modality: str) -> xr.DataArray | None:
    return data.eeg_gain if modality == "eeg" else data.fnirs_gain


def _check_samples(u: xr.DataArray, u_paired: xr.DataArray, v: xr.DataArray, test: JoinedRuns,
                   labels: PairLabels):
    """
    Raise TooFewSamples for a labelled pair without enough finite time points, for
    each correlation `score` computes: `u_paired` with `v`, and with ground truth
    `u` with the EEG ground truth and `v` with the fNIRS ground truth.
    """
    for source, pair in labels.pairs.items():
        compared = []
        if test.s_eeg is not None:
            compared += [("EEG", u.sel(component=pair), test.s_eeg.sel(source=source)),
                         ("fNIRS", v.sel(component=pair), test.s_fnirs.sel(source=source))]
        compared.append(("EEG-fNIRS", u_paired.sel(component=pair), v.sel(component=pair)))
        for name, a, b in compared:
            a, b = xr.align(a, b)
            n = int((np.isfinite(a.values) & np.isfinite(b.values)).sum())
            if n < MIN_SAMPLES:
                raise TooFewSamples(f"pair {pair} ({source}), {name}: {n} finite time points "
                                    f"in common on {list(test.runs)}, fewer than "
                                    f"{MIN_SAMPLES}.")
