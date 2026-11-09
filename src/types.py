"""
The data objects pipeline modules hand each other. No logic.

Each type is added together with the module that produces it.
"""
from dataclasses import dataclass, field

import mne
import pandas as pd
import xarray as xr

from src.config import ExperimentConfig, SubjectConfig

# Columns of a run's stim table (PreprocessedRecording.stim), as saved.
STIM_COLUMNS = ["onset", "duration", "value", "trial_type"]
# Columns of JoinedRuns.stim and Recovery.stim: the same, plus the run of each stimulus.
JOINED_STIM_COLUMNS = [*STIM_COLUMNS, "run"]


@dataclass(frozen=True)
class GroundTruth:
    """
    What the simulator put into a synthetic run. Real data has none.

    Sources are labelled by hand, "right_hand" before "left_hand" when both are
    present, so source position means the same physical source in every array.

    Attributes
    ----------
    s_eeg : xr.DataArray
        EEG source time courses, dims (time, source), at the EEG rate.
    s_fnirs : xr.DataArray
        fNIRS source time courses, dims (time, source, chromo), at the fNIRS rate.
    A_eeg : xr.DataArray
        EEG mixing matrix, dims (channel, source).
    A_fnirs : xr.DataArray
        fNIRS mixing matrix, dims (channel, source, wavelength, chromo).
    """
    s_eeg: xr.DataArray
    s_fnirs: xr.DataArray
    A_eeg: xr.DataArray
    A_fnirs: xr.DataArray


@dataclass(frozen=True)
class RawRecording:
    """
    One run as it is on disk, before any processing. Preprocessing returns a
    different type, so the stage of the data is always visible from its class.

    Both modalities share one clock: `eeg.times` and `fnirs.time` are in seconds from
    the start of the EEG recording, and `stim` onsets are on that clock. The fNIRS may
    start before the EEG (negative times) when the device recorded the rest before it;
    preprocessing cuts both to the stimulus window.

    Attributes
    ----------
    group : str
        CV group the run belongs to, e.g. "sitting_post_rest_1".
    run : str
        Run name within the group, e.g. "run_001_seed_2418716002".
    eeg : mne.io.Raw
        EEG as recorded, all channels (EOG and EMG included). Once loading has applied
        the subject's montage, the EEG channels are spelled and placed as in it.
    fnirs : xr.DataArray
        fNIRS amplitude, dims (channel, wavelength, time).
    geo3d : xr.DataArray
        Optode positions (mm), dims (label, pos): the subject's fNIRS montage once
        loading has applied it. A reader returns the positions its files hold, empty
        if they hold none.
    stim : pd.DataFrame
        Stimuli: onset, duration, trial_type (+ any other columns the file has).
    ground_truth : GroundTruth | None
        None for real data.
    blocks : dict[str, tuple[float, float]] | None
        For a session recorded in one piece: the CV group of each block, mapped to its
        start and end (s) on the shared clock, in recording order. The session is
        split into these groups after preprocessing. None when every run is its own
        recording.
    """
    group: str
    run: str
    eeg: mne.io.Raw
    fnirs: xr.DataArray
    geo3d: xr.DataArray
    stim: pd.DataFrame
    ground_truth: GroundTruth | None = None
    blocks: dict[str, tuple[float, float]] | None = None


@dataclass(frozen=True)
class BadChannels:
    """
    The channels pruning found bad for the subject, and why.

    Attributes
    ----------
    eeg : tuple[str, ...]
        Bad EEG channels. They are interpolated, so the EEG montage stays fixed. On a
        PrunedRecording, those of its own recording; on a RunResult, those bad in any of
        the subject's recordings.
    fnirs : tuple[str, ...]
        Bad fNIRS channels: those bad in any recording the fNIRS was judged on. They are
        dropped from every run.
    eeg_detail : dict[str, dict[str, tuple[str, ...]]]
        Channels flagged by each pyprep criterion in any recording judged, per pass:
        {"pass1": {"bad_by_flat": (...), ...}, "pass2": {"bad_by_correlation": (...), ...}}.
    fnirs_detail : dict[str, tuple[str, ...]]
        Channels failing each quality criterion in any recording judged: "sd_dist",
        "mean_amp", "snr", "sci_psp".
    eeg_by_recording : dict[str, tuple[str, ...]]
        Bad EEG channels of each recording judged, by recording name.
    fnirs_by_recording : dict[str, tuple[str, ...]]
        Bad fNIRS channels of each recording judged, by recording name.
    """
    eeg: tuple[str, ...]
    fnirs: tuple[str, ...]
    eeg_detail: dict[str, dict[str, tuple[str, ...]]]
    fnirs_detail: dict[str, tuple[str, ...]]
    eeg_by_recording: dict[str, tuple[str, ...]] = field(default_factory=dict)
    fnirs_by_recording: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class PrunedRecording:
    """
    One run after channel pruning: bad EEG channels interpolated, bad fNIRS channels
    dropped. Everything else is as in the RawRecording it came from, on the same clock.

    Attributes
    ----------
    group : str
        CV group the run belongs to.
    run : str
        Run name within the group.
    eeg : mne.io.Raw
        EEG with the bad EEG channels interpolated, placed by the subject's montage.
        All channels are kept (EOG and EMG included).
    fnirs : xr.DataArray
        fNIRS amplitude without the bad channels, dims (channel, wavelength, time).
    geo3d : xr.DataArray
        Optode positions (mm).
    fnirs_optodes : dict[str, tuple[str, str]]
        Source and detector label of every fNIRS channel of the recording, also those
        pruning dropped: {channel: (source, detector)}, labels as in geo3d.
    stim : pd.DataFrame
        Stimuli: onset, duration, trial_type.
    ground_truth : GroundTruth | None
        As loaded, over all channels. Matching it to the kept channels is up to the
        evaluation.
    bad_channels : BadChannels
        What was interpolated in this run and dropped.
    blocks : dict[str, tuple[float, float]] | None
        As in the RawRecording: the CV group of each block of a session recorded in
        one piece, mapped to its start and end (s). None when every run is its own
        recording.
    """
    group: str
    run: str
    eeg: mne.io.Raw
    fnirs: xr.DataArray
    geo3d: xr.DataArray
    fnirs_optodes: dict[str, tuple[str, str]]
    stim: pd.DataFrame
    ground_truth: GroundTruth | None
    bad_channels: BadChannels
    blocks: dict[str, tuple[float, float]] | None = None


@dataclass(frozen=True)
class PreprocessedRecording:
    """
    One run after preprocessing: the EEG feature and the fNIRS HbO, cut to the
    stimulus window, on a shared time axis and z-scored per channel. The ground-truth
    sources went through the same transforms as the data and sit on the fNIRS time
    axis, so they can be compared with what the models recover.

    Attributes
    ----------
    group : str
        CV group the run belongs to.
    run : str
        Run name within the group.
    eeg : xr.DataArray
        EEG feature, dims (time, channel). "bandpower": the band power envelope on the
        fNIRS time axis. "bandpassed": the band-filtered EEG at its own rate, over the
        same time span as the fNIRS, centred but not scaled.
    fnirs : xr.DataArray
        HbO, dims (time, channel).
    s_eeg : xr.DataArray | None
        EEG ground-truth sources as band power, dims (time, source), on the fNIRS time
        axis. None for real data.
    s_fnirs : xr.DataArray | None
        fNIRS ground-truth sources (HbO), band-passed like the data, dims (time,
        source), on the fNIRS time axis. None for real data.
    geo3d : xr.DataArray
        Optode positions (mm).
    eeg_geo3d : xr.DataArray
        Electrode positions (mm) of every EEG channel of the recording (also those
        dropped from `eeg`) and the head landmarks, dims (label, pos), typed
        ELECTRODE and LANDMARK, from the subject's montage.
    fnirs_optodes : dict[str, tuple[str, str]]
        Source and detector label of every fNIRS channel of the recording, also those
        pruning dropped: {channel: (source, detector)}, labels as in geo3d.
    stim : pd.DataFrame
        Stimuli: onset, duration, trial_type, on the same clock as `time`.
    window : tuple[float, float]
        The stimulus window (s) every signal was cut to. The signals can start later
        or end earlier than it where the data does not reach that far.
    ground_truth : GroundTruth | None
        As loaded, for the mixing matrices. None for real data.
    eeg_std, fnirs_std : xr.DataArray | None
        The standard deviation each channel was divided by when it was z-scored, dims
        (channel,). Z-scoring scales the mixing of channel c by 1 / std_c, so the
        ground-truth mixing is scaled the same way before it is compared with what the
        models recover. None where the signal was not scaled (band-passed EEG).
    blocks : dict[str, tuple[float, float]] | None
        For a session recorded in one piece, before it is split: the CV group of each
        block, mapped to its start and end (s). None for a run that is its own
        recording, and for the block runs the split returns.
    """
    group: str
    run: str
    eeg: xr.DataArray
    fnirs: xr.DataArray
    s_eeg: xr.DataArray | None
    s_fnirs: xr.DataArray | None
    geo3d: xr.DataArray
    eeg_geo3d: xr.DataArray
    fnirs_optodes: dict[str, tuple[str, str]]
    stim: pd.DataFrame
    window: tuple[float, float]
    ground_truth: GroundTruth | None
    eeg_std: xr.DataArray | None = None
    fnirs_std: xr.DataArray | None = None
    blocks: dict[str, tuple[float, float]] | None = None


@dataclass(frozen=True)
class JoinedRuns:
    """
    Several preprocessed runs joined end to end into one recording: what an unmixing
    model is fit on or applied to.

    The time axes are rebuilt: each run continues one fNIRS sample after the previous
    one ends, so the axes are regular. Time shifts of the temporal models therefore
    reach across run boundaries; `run_starts` says where those are.

    Attributes
    ----------
    eeg : xr.DataArray
        Dims (time, channel). Band power on the fNIRS time axis, or band-passed EEG at
        its own rate with k samples per fNIRS sample (mSPoC).
    fnirs : xr.DataArray
        HbO, dims (time, channel).
    s_eeg, s_fnirs : xr.DataArray | None
        Ground-truth sources, dims (time, source), on the fNIRS time axis. None for
        real data.
    run_starts : tuple[float, ...]
        Time (s) at which each run starts on the joined axes, in order.
    runs : tuple[str, ...]
        "group/run" of each run, in the same order.
    stim : pd.DataFrame
        The stimuli of every run on the joined time axis: onset, duration, value,
        trial_type, and run ("group/run"). Only stimuli whose onset lies within the
        run's data are kept.
    ground_truth : GroundTruth | None
        Of the first run, for the mixing matrices, which every run shares. None for
        real data.
    eeg_gain, fnirs_gain : xr.DataArray | None
        The factor z-scoring multiplied each channel by, dims (channel,): 1 / std of
        each run, averaged over the runs weighted by their number of fNIRS samples.
        Multiplying the ground-truth mixing by it gives the mixing of the joined
        data. None where the runs were not scaled (band-passed EEG).
    """
    eeg: xr.DataArray
    fnirs: xr.DataArray
    s_eeg: xr.DataArray | None
    s_fnirs: xr.DataArray | None
    run_starts: tuple[float, ...]
    runs: tuple[str, ...]
    stim: pd.DataFrame
    ground_truth: GroundTruth | None
    eeg_gain: xr.DataArray | None = None
    fnirs_gain: xr.DataArray | None = None


@dataclass(frozen=True)
class FittedModel:
    """
    One unmixing model fitted on one set of runs.

    Attributes
    ----------
    method : str
        Method name.
    params : dict
        The hyperparameters it was built with.
    model : object
        The fitted cedalion model.
    """
    method: str
    params: dict
    model: object


@dataclass(frozen=True)
class PairLabels:
    """
    Which canonical pair carries which hand, chosen on training data.

    Attributes
    ----------
    pairs : dict[str, int]
        Hand (the ground-truth source of that name, on synthetic data) to pair
        (component index), e.g. {"right_hand": 0, "left_hand": 1}. The pair's EEG and
        fNIRS components both go with that hand.
    score : float
        The labelling's score on the training data, as the assignment strategy
        defines it.
    """
    pairs: dict[str, int]
    score: float


@dataclass(frozen=True)
class Scores:
    """
    How well a fitted model recovers the ground truth on one split. Without ground
    truth only `labels` and `canonical_r` carry values: every other per-source score
    is NaN, `best_pairs` is empty.

    Attributes
    ----------
    labels : PairLabels
        The pairs scored, labelled on the training data.
    r_eeg, r_fnirs : dict[str, float]
        Per source: |r| of its pair's EEG (fNIRS) component with the EEG (fNIRS)
        ground truth. The component's sign is arbitrary, so only |r| is kept.
    isi_eeg, isi_fnirs : float
        Amari ISI of the labelled pairs' filters against the ground-truth mixing: 0 is
        perfect separation, 1 none. NaN with a single source.
    canonical_r : dict[str, float]
        Per source: signed correlation between its pair's EEG and fNIRS component on
        this split, the pair's canonical correlation as the model found it; negative
        means the two modalities moved against each other here. Uses no ground truth.
    best_pairs : dict[str, tuple[int, float, float]]
        Diagnostic only: per source, the pair (out of all fitted pairs) that matched
        it best on the training data, and its (|r_eeg|, |r_fnirs|) on this split.
    pattern_r_eeg, pattern_r_fnirs : dict[str, float]
        Diagnostic only, never used for selection: per source, |r| of its pair's Haufe
        pattern on this split with the ground-truth mixing over the channels. NaN
        without ground truth, and for combinations the grid search did not select, as
        it computes no patterns.
    leak_eeg, leak_fnirs : dict[str, float]
        Diagnostic only, never used for selection: per source, |r| of the EEG (fNIRS)
        component of the pair assigned to the other source with this source's EEG
        (fNIRS) ground truth: how much of this source leaks into the other pair. NaN
        with a single source.
    """
    labels: PairLabels
    r_eeg: dict[str, float]
    r_fnirs: dict[str, float]
    isi_eeg: float
    isi_fnirs: float
    canonical_r: dict[str, float]
    best_pairs: dict[str, tuple[int, float, float]]
    pattern_r_eeg: dict[str, float]
    pattern_r_fnirs: dict[str, float]
    leak_eeg: dict[str, float]
    leak_fnirs: dict[str, float]


@dataclass(frozen=True)
class Recovery:
    """
    The components of a fitted model on held-out runs, next to the ground truth if
    there is one: what time-course and pair-diagnostic plots are drawn from. The held-out runs are the
    test group for a selected model, or a validation group for a fold refit by the
    rebuild.

    Attributes
    ----------
    u, v : xr.DataArray
        EEG and fNIRS components of every pair on the held-out runs, with the sign
        the model gave them, dims (time, component). EEG components of temporal models are NaN near run
        boundaries and at the end, where cedalion padded them.
    s_eeg, s_fnirs : xr.DataArray | None
        Ground truth on the held-out runs, dims (time, source). None for real data.
    run_starts : tuple[float, ...]
        Time (s) at which each held-out run starts on that axis.
    runs : tuple[str, ...]
        "group/run" of each held-out run.
    stim : pd.DataFrame
        The stimuli of the held-out runs on that axis, as in `JoinedRuns.stim`.
    pattern_eeg, pattern_fnirs : xr.DataArray
        Haufe patterns of every pair on the held-out runs, dims (channel, component):
        the fitted filters with the held-out data covariance, with the sign the model
        gave the filters. The plots orient a component and its pattern together.
    r_train_eeg, r_train_fnirs : xr.DataArray | None
        Correlation of every pair with every source on the training runs (what the
        "ground_truth" labelling uses), dims (component, source). None for real data.
    r_test_eeg, r_test_fnirs : xr.DataArray | None
        The same on the held-out runs.
    gain_eeg, gain_fnirs : xr.DataArray | None
        `JoinedRuns.eeg_gain` and `fnirs_gain` of the held-out runs, so the plots can
        scale the ground-truth mixing as the scores did.
    """
    u: xr.DataArray
    v: xr.DataArray
    s_eeg: xr.DataArray | None
    s_fnirs: xr.DataArray | None
    run_starts: tuple[float, ...]
    runs: tuple[str, ...]
    stim: pd.DataFrame
    pattern_eeg: xr.DataArray
    pattern_fnirs: xr.DataArray
    r_train_eeg: xr.DataArray | None
    r_train_fnirs: xr.DataArray | None
    r_test_eeg: xr.DataArray | None
    r_test_fnirs: xr.DataArray | None
    gain_eeg: xr.DataArray | None = None
    gain_fnirs: xr.DataArray | None = None


@dataclass(frozen=True)
class FoldScore:
    """
    One hyperparameter combination scored in one validation fold: one point of the
    selection landscape.

    Attributes
    ----------
    method : str
    test_group : str
        The group held out of the whole selection.
    val_group : str
        The group this fold validated on.
    params : dict
    scores : Scores | None
        None if the fit failed.
    metrics : dict[str, float]
        Each selection metric's value in this fold alone (the plain mean of its
        quantities, see `src.unmixing.metrics`); NaN if the fit failed. For display:
        selection averages each quantity over the folds first.
    error : str | None
        Why the fit failed.
    """
    method: str
    test_group: str
    val_group: str
    params: dict
    scores: Scores | None
    metrics: dict[str, float]
    error: str | None


@dataclass(frozen=True)
class CombinationScore:
    """
    One hyperparameter combination's mean over the validation folds, per selection
    metric: the numbers selection compares, the best one per metric winning.

    Attributes
    ----------
    method : str
    test_group : str
        The group held out of the whole selection.
    params : dict
    means : dict[str, float]
        Per metric, the value selection compares (`fold_mean`): each of its quantities
        averaged over the folds, then the plain mean of those. NaN if a quantity has
        no finite value in any fold.
    """
    method: str
    test_group: str
    params: dict
    means: dict[str, float]


@dataclass(frozen=True)
class FoldRecovery:
    """
    A selected model's hyperparameters fit in one validation fold of the selection,
    as the selection fit them there (same runs, same seed): what the fold's time-
    course plots are drawn from.

    Attributes
    ----------
    val_group : str
        The group the fold validated on.
    scores : Scores
        On the validation group, pairs labelled on the fold's training runs; the same
        as the fold's scores in the landscape.
    recovery : Recovery
        The fold model's components on the validation group.
    """
    val_group: str
    scores: Scores
    recovery: Recovery


@dataclass(frozen=True)
class Selection:
    """
    One method's model selected by one metric and scored on the held-out group.

    Attributes
    ----------
    method : str
    metric : str
        The selection metric that picked `params`.
    params : dict
        The selected hyperparameters.
    inner_score : float
        The metric's mean over the validation folds for `params`.
    fitted : FittedModel
        Refit with `params` on all runs except the test group.
    scores : Scores
        On the test group, pairs labelled on the training runs.
    recovery : Recovery
        The components on the test group with the ground truth.
    test_group : str
    folds : tuple[FoldRecovery, ...]
        `params` fit in each validation fold, for non-nested selections; empty for
        nested ones. A fold whose fit failed has none.
    """
    method: str
    metric: str
    params: dict
    inner_score: float
    fitted: FittedModel
    scores: Scores
    recovery: Recovery
    test_group: str
    folds: tuple[FoldRecovery, ...]


@dataclass(frozen=True)
class UnmixingResult:
    """
    What unmixing produced for one subject.

    Attributes
    ----------
    test_group : str
        The group held out in non-nested evaluation.
    selections : list[Selection]
        Non-nested: one per method and metric. These models go on to classification.
    nested : dict[str, list[Selection]] | None
        Nested evaluation, for analysis only: the selections with each group held out
        once, keyed by that group. None when not run.
    landscape : list[FoldScore]
        Every combination in every validation fold, of every method, for the
        non-nested selection and every nested one.
    combinations : list[CombinationScore]
        Every combination's fold means, for the same selections.
    """
    test_group: str
    selections: list[Selection]
    nested: dict[str, list[Selection]] | None
    landscape: list[FoldScore]
    combinations: list[CombinationScore]


@dataclass(frozen=True)
class ClassificationScore:
    """
    rLDA classification of a held-out group's trials from the components of one
    selected model, with one feature set. The classifier is fit on the trials of all
    other groups, the ones the model was fit on.

    Attributes
    ----------
    method, metric : str
        The selection whose model gave the components.
    test_group : str
        The held-out group whose trials are classified.
    feature_set : str
        "eeg_fnirs", "eeg" or "fnirs".
    classes : tuple[str, ...]
        The classes, in the order of `confusion`.
    confusion : dict[str, dict[str, int]]
        True class to predicted class to number of test trials.
    n_train : int
        Training trials.
    dropped_train, dropped_test : int
        Trials left out because their baseline window or a whole interval had no
        finite sample, e.g. in the NaN end of a temporal model's EEG component. An
        epoch cut by the end of the run is kept, its windows averaged over the
        part the run has.
    """
    method: str
    metric: str
    test_group: str
    feature_set: str
    classes: tuple[str, ...]
    confusion: dict[str, dict[str, int]]
    n_train: int
    dropped_train: int
    dropped_test: int

    @property
    def n_test(self) -> int:
        return sum(sum(row.values()) for row in self.confusion.values())

    @property
    def accuracy(self) -> float:
        """Correctly classified test trials / test trials."""
        return sum(self.confusion[c][c] for c in self.classes) / self.n_test

    @property
    def recall(self) -> dict[str, float]:
        """Per class: its test trials classified as it / its test trials."""
        return {c: self.confusion[c][c] / sum(self.confusion[c].values()) for c in self.classes}


@dataclass(frozen=True)
class ClassificationResult:
    """
    What classification produced for one subject, laid out like `UnmixingResult`.

    Attributes
    ----------
    test_group : str
        The group held out in non-nested evaluation.
    scores : list[ClassificationScore]
        Non-nested: one per selection and feature set.
    nested : dict[str, list[ClassificationScore]] | None
        The same for the nested selections, keyed by the held-out group. None when
        unmixing ran no nested evaluation.
    """
    test_group: str
    scores: list[ClassificationScore]
    nested: dict[str, list[ClassificationScore]] | None


@dataclass(frozen=True)
class Geometry:
    """
    Where the channels are and how the simulated sources reach them: what spatial
    plots (topomaps of patterns and of the ground-truth mixing) are drawn from.

    Attributes
    ----------
    eeg_channels : tuple[str, ...]
        EEG channels in data order.
    fnirs_channels : tuple[str, ...]
        fNIRS channels kept after pruning, in data order.
    geo3d : xr.DataArray
        Optode positions, dims (label, pos).
    eeg_geo3d : xr.DataArray
        Electrode positions of every EEG channel and the head landmarks, dims
        (label, pos).
    fnirs_optodes : dict[str, tuple[str, str]]
        Source and detector label of every fNIRS channel, also those pruning dropped:
        {channel: (source, detector)}.
    A_eeg : xr.DataArray | None
        Ground-truth EEG mixing, dims (channel, source). None for real data.
    A_fnirs : xr.DataArray | None
        Ground-truth fNIRS mixing, dims (channel, source, wavelength, chromo), over
        all channels. None for real data.
    """
    eeg_channels: tuple[str, ...]
    fnirs_channels: tuple[str, ...]
    geo3d: xr.DataArray
    eeg_geo3d: xr.DataArray
    fnirs_optodes: dict[str, tuple[str, str]]
    A_eeg: xr.DataArray | None
    A_fnirs: xr.DataArray | None

    @classmethod
    def of(cls, recording: "PreprocessedRecording") -> "Geometry":
        gt = recording.ground_truth
        return cls(eeg_channels=tuple(str(c) for c in recording.eeg.channel.values),
                   fnirs_channels=tuple(str(c) for c in recording.fnirs.channel.values),
                   geo3d=recording.geo3d, eeg_geo3d=recording.eeg_geo3d,
                   fnirs_optodes=recording.fnirs_optodes,
                   A_eeg=None if gt is None else gt.A_eeg,
                   A_fnirs=None if gt is None else gt.A_fnirs)


@dataclass(frozen=True)
class RunResult:
    """
    Everything one run produced. `experiment.run` builds it from the steps' return
    values at the end of the run; the results module writes it to disk. Fields are
    added as the pipeline steps are refactored.

    Attributes
    ----------
    cfg : ExperimentConfig
        The experiment that ran, for the experiment.yaml snapshot.
    subject : SubjectConfig
        The subject it ran on, for the subject.yaml snapshot.
    bad_channels : BadChannels
        The channels pruning interpolated (EEG) and dropped (fNIRS).
    alignment : pd.DataFrame
        One row per run: how it was cut and aligned (window, spans, sample counts).
    geometry : Geometry
        Channel positions and ground-truth mixing.
    unmixing : UnmixingResult
        The selected models and their scores.
    inputs : list[PreprocessedRecording] | None
        The preprocessed runs every model was fit on, in memory after a run, for
        the figures. Written to disk only if the experiment sets save_inputs; None
        when the result is read back with `load_run`.
    started, finished : str
        When the run started and finished, ISO 8601.
    classification : ClassificationResult | None
        The trials classified from the selected models' components. None when the
        experiment has no classification section.
    """
    cfg: ExperimentConfig
    subject: SubjectConfig
    bad_channels: BadChannels
    alignment: pd.DataFrame
    geometry: Geometry
    unmixing: UnmixingResult
    inputs: list[PreprocessedRecording] | None
    started: str
    finished: str
    classification: ClassificationResult | None = None
