"""
Config dataclasses: one per pipeline section, each read only by its own module.

Every dataclass checks its own fields on construction, so a config built directly in
Python (e.g. in a notebook) passes the same checks as one loaded from YAML.

Names that another module owns are not checked here; each of those modules checks
its own:
- methods, their grids and the Laplacian spec (unmixing)
- metrics (unmixing)
- the subject format and its paths, the montage name and files (loading)
"""
from dataclasses import dataclass, field
from pathlib import Path

HANDS = ("left", "right", "both")
SIDES = ("right", "left")
DOWNSAMPLE_METHODS = ("resample", "epoch_mean")
EEG_FEATURES = ("bandpower", "bandpassed")
FEATURE_SETS = ("eeg_fnirs", "eeg", "fnirs")


def _check_choice(value, choices: tuple, name: str):
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}.")


def _check_window(window: tuple, name: str):
    lo, hi = window
    if lo >= hi:
        raise ValueError(f"{name} must be [lo, hi] with lo < hi, got {list(window)}.")


# -------------------------------------------------------------------------
# Pruning
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class EegPruningPass1:
    """
    Thresholds for the first of the two EEG bad-channel detection passes.

    EEG pruning runs in two passes with an average re-reference in between (see
    `detect_bad_eeg_channels` in src/pruning/eeg.py):
    - pass 1 judges each channel on its own, on the data as recorded
    - the data is re-referenced to the average of the channels pass 1 kept
    - pass 2 compares the channels against each other (EegPruningPass2)
    A channel flagged in either pass is pruned.

    Pass 1 criteria do not depend on the reference, so they are valid on the
    recording reference. Defaults are pyprep's own. 

    Attributes
    ----------
    flat_threshold : float
        SD or MAD (V) below which a channel counts as flat.
    deviation_threshold : float
        Robust z-score of the channel amplitude.
    hf_noise_threshold : float
        Robust z-score of the >50 Hz noise-to-signal ratio.
    """
    flat_threshold: float = 1.0e-15
    deviation_threshold: float = 5.0
    hf_noise_threshold: float = 5.0


@dataclass(frozen=True)
class EegPruningPass2:
    """
    Thresholds for the second of the two EEG bad-channel detection passes.

    Pass 2 compares the channels against each other (correlation, SNR, RANSAC). It
    runs after the data is re-referenced to the average of the channels pass 1 kept,
    because these criteria assume the shared reference signal has been removed (see
    EegPruningPass1 for the full sequence, and `detect_bad_eeg_channels` in
    src/pruning/eeg.py). Defaults are pyprep's own.

    Attributes
    ----------
    correlation_secs : float
        Window length (s) of the correlation criterion.
    correlation_threshold : float
        Lowest acceptable best-neighbour correlation.
    correlation_frac_bad : float
        Fraction of bad windows that condemns a channel.
    ransac_n_samples, ransac_sample_prop, ransac_corr_thresh, ransac_frac_bad,
    ransac_window_secs
        RANSAC parameters, passed to pyprep unchanged.
    """
    correlation_secs: float = 1.0
    correlation_threshold: float = 0.4
    correlation_frac_bad: float = 0.1
    ransac_n_samples: int = 50
    ransac_sample_prop: float = 0.25
    ransac_corr_thresh: float = 0.75
    ransac_frac_bad: float = 0.4
    ransac_window_secs: float = 5.0


@dataclass(frozen=True)
class EegPruningConfig:
    """
    EEG bad-channel detection: two pyprep passes separated by an average re-reference.

    Attributes
    ----------
    line_freqs : tuple[float, ...]
        Line-noise frequencies notched before detection.
    ransac : bool
        If True, run the RANSAC criterion in pass 2.
    pass1 : EegPruningPass1
    pass2 : EegPruningPass2
    """
    line_freqs: tuple[float, ...] = (50, 100)
    ransac: bool = True
    pass1: EegPruningPass1 = field(default_factory=EegPruningPass1)
    pass2: EegPruningPass2 = field(default_factory=EegPruningPass2)

    def __post_init__(self):
        if not self.line_freqs:
            raise ValueError("line_freqs must list at least one frequency.")


@dataclass(frozen=True)
class FnirsPruningConfig:
    """
    fNIRS bad-channel detection. The criteria are judged on a cleaned amplitude
    (motion corrected and high-pass filtered in optical density), and a channel is
    kept only if it passes every criterion (see cedalion.sigproc.quality).

    Attributes
    ----------
    highpass_hz : float
        High-pass cutoff (Hz) applied in optical density before the criteria.
    butter_order : int
        Order of the Butterworth high-pass filter.
    sd_range_mm : tuple[float, float]
        Allowed source-detector distance (mm).
    amp_range_v : tuple[float, float]
        Allowed mean amplitude (V).
    snr_threshold : float
        Lowest signal-to-noise ratio (mean over standard deviation).
    sci_threshold : float
        Lowest scalp coupling index, per window.
    psp_threshold : float
        Lowest peak spectral power, per window.
    window_length_s : float
        Window length (s) for SCI and PSP.
    time_pass_fraction : float
        Fraction of windows in which a channel must pass SCI and PSP together.
    """
    highpass_hz: float = 0.01
    butter_order: int = 4
    sd_range_mm: tuple[float, float] = (10.0, 45.0)
    amp_range_v: tuple[float, float] = (1e-3, 0.85)
    snr_threshold: float = 3.0
    sci_threshold: float = 0.5
    psp_threshold: float = 0.05
    window_length_s: float = 5.0
    time_pass_fraction: float = 0.5

    def __post_init__(self):
        if self.highpass_hz <= 0:
            raise ValueError(f"highpass_hz must be > 0, got {self.highpass_hz}.")
        if self.butter_order < 1:
            raise ValueError(f"butter_order must be >= 1, got {self.butter_order}.")
        _check_window(self.sd_range_mm, "sd_range_mm")
        _check_window(self.amp_range_v, "amp_range_v")
        if self.window_length_s <= 0:
            raise ValueError(f"window_length_s must be > 0, got {self.window_length_s}.")
        if not 0 <= self.time_pass_fraction <= 1:
            raise ValueError(f"time_pass_fraction must be in [0, 1], got {self.time_pass_fraction}.")


@dataclass(frozen=True)
class PruningConfig:
    """
    Bad-channel detection for both modalities. Which recording it runs on is subject
    information (SubjectConfig.paths).

    Attributes
    ----------
    random_state : int
        Seed for pyprep's RANSAC.
    window_sec : float
        fNIRS detection window: seconds of rest judged, from the rest onset. The
        window never runs past the end of the rest period. The EEG is judged on the
        whole rest recording.
    eeg : EegPruningConfig
    fnirs : FnirsPruningConfig
    """
    random_state: int = 42
    window_sec: float = 25.0
    eeg: EegPruningConfig = field(default_factory=EegPruningConfig)
    fnirs: FnirsPruningConfig = field(default_factory=FnirsPruningConfig)

    def __post_init__(self):
        if self.window_sec <= 0:
            raise ValueError(f"window_sec must be positive, got {self.window_sec}.")


# -------------------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class EegPreprocessingConfig:
    """
    EEG cleaning and the frequency band the EEG feature is taken from.

    Attributes
    ----------
    bandpass : tuple[float, float]
        Broadband filter (Hz) applied before ICA.
    ica_n_components : int
        ICA components to fit. Capped at the number of channels that were not
        interpolated, since an interpolated channel adds no independent signal.
    ica_method : str
        ICA algorithm, as mne.preprocessing.ICA names it (e.g. "picard", "fastica").
    random_state : int
        Seed for ICA.
    feature_band : tuple[float, float]
        Band (Hz) of the EEG feature, e.g. alpha. The EEG ground truth is taken from
        the same band, so recovery is scored on the same signal.
    drop_interpolated : bool
        Drop the channels pruning interpolated from the EEG feature. An interpolated
        channel is a linear combination of its neighbours, so band-passed EEG with
        them has a singular channel covariance, which mSPoC cannot use.
    """
    bandpass: tuple[float, float] = (1.0, 45.0)
    ica_n_components: int = 15
    ica_method: str = "picard"
    random_state: int = 42
    feature_band: tuple[float, float] = (8.0, 12.0)
    drop_interpolated: bool = False

    def __post_init__(self):
        _check_window(self.bandpass, "bandpass")
        _check_window(self.feature_band, "feature_band")
        if self.ica_n_components < 1:
            raise ValueError(f"ica_n_components must be >= 1, got {self.ica_n_components}.")


@dataclass(frozen=True)
class FnirsPreprocessingConfig:
    """
    fNIRS conversion to haemoglobin and filtering.

    Attributes
    ----------
    dpf : float
        Differential pathlength factor, used for both wavelengths.
    bandpass : tuple[float, float]
        Band (Hz) kept of the haemoglobin signal. The fNIRS ground truth is filtered
        the same way.
    """
    dpf: float = 6.0
    bandpass: tuple[float, float] = (0.01, 0.6)

    def __post_init__(self):
        if self.dpf <= 0:
            raise ValueError(f"dpf must be > 0, got {self.dpf}.")
        _check_window(self.bandpass, "bandpass")


@dataclass(frozen=True)
class AlignmentCheckConfig:
    """
    When a preprocessed run counts as badly cut or aligned. A run that fails stops
    the pipeline.

    Attributes
    ----------
    min_window_coverage : float
        Fraction of the stimulus window the kept data must cover.
    """
    min_window_coverage: float = 0.9

    def __post_init__(self):
        if not 0 < self.min_window_coverage <= 1:
            raise ValueError(f"min_window_coverage must be in (0, 1], got {self.min_window_coverage}.")


@dataclass(frozen=True)
class PreprocessingConfig:
    """
    Signal preprocessing, usually fixed once set up. The models are always fit on the
    full recording (cut to the stimulus window); the input is never epoched.

    Attributes
    ----------
    eeg_feature : str
        "bandpower" (CCA family): the power envelope of the feature band, on the fNIRS
        time grid. "bandpassed" (mSPoC): the band-filtered EEG at its own rate.
    fnirs_srate : float
        Rate (Hz) the EEG bandpower is brought to before it is aligned to the fNIRS.
    downsample_method : str
        How the EEG bandpower is brought to fnirs_srate: "resample" | "epoch_mean".
    trim_pre_sec, trim_post_sec : float
        Recording kept from the first onset - trim_pre_sec until the last stimulus
        end + trim_post_sec.
    fnirs_mspoc_srate : float | None
        mSPoC only: fNIRS rate (Hz) that sets mSPoC's epoch length. None = native rate.
    eeg : EegPreprocessingConfig
    fnirs : FnirsPreprocessingConfig
    checks : AlignmentCheckConfig
    """
    eeg_feature: str = "bandpower"
    fnirs_srate: float = 7.4
    downsample_method: str = "resample"
    trim_pre_sec: float = 5.0
    trim_post_sec: float = 20.0
    fnirs_mspoc_srate: float | None = None
    eeg: EegPreprocessingConfig = field(default_factory=EegPreprocessingConfig)
    fnirs: FnirsPreprocessingConfig = field(default_factory=FnirsPreprocessingConfig)
    checks: AlignmentCheckConfig = field(default_factory=AlignmentCheckConfig)

    def __post_init__(self):
        _check_choice(self.downsample_method, DOWNSAMPLE_METHODS, "downsample_method")
        _check_choice(self.eeg_feature, EEG_FEATURES, "eeg_feature")
        if self.trim_pre_sec < 0 or self.trim_post_sec < 0:
            raise ValueError(f"trim_pre_sec and trim_post_sec must be >= 0, got "
                             f"{self.trim_pre_sec}/{self.trim_post_sec}.")
        if self.fnirs_mspoc_srate is not None:
            if self.eeg_feature != "bandpassed":
                raise ValueError("fnirs_mspoc_srate only applies to eeg_feature: bandpassed.")
            if not 0 < self.fnirs_mspoc_srate < self.fnirs_srate:
                raise ValueError(f"fnirs_mspoc_srate must be in (0, fnirs_srate={self.fnirs_srate}), "
                                 f"got {self.fnirs_mspoc_srate}; resampling never upsamples.")


# -------------------------------------------------------------------------
# Unmixing and classification
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class AssignmentConfig:
    """
    How the unlabelled canonical pairs are assigned to the hands (right_hand,
    left_hand). Each pair keeps its EEG and fNIRS component together.

    Attributes
    ----------
    method : str
        The assignment strategy (src/unmixing/assignment.py, STRATEGIES).
        "ground_truth": the assignment that matches the ground truth best on the
        training split.
        "motor_channels": the assignment whose estimated mixing lies most on each
        hand's `motor_channels`; needs no ground truth.
        "task_regressors": the assignment whose components follow each hand's trials
        best; needs no ground truth.
    n_top : int
        The sources are assigned to pairs among the first `n_top` (by canonical
        correlation). With 2 sources and n_top 2, the two pairs are labelled
        right_hand/left_hand or the other way round.
    eeg_weight, fnirs_weight : float
        How much each modality counts. "ground_truth": an assignment scores
        eeg_weight * |r(EEG component, EEG source)| + fnirs_weight * |r(fNIRS
        component, fNIRS source)|, summed over the hands. "motor_channels": the same
        weights on the shares of the mixing on the motor channels
        (src/unmixing/assignment.py, assign_by_motor_channels). "task_regressors": the
        same weights on |r| with each hand's task references.
    motor_channels : dict[str, dict[str, tuple[str, ...]]]
        "motor_channels" only: modality ("eeg", "fnirs") to hand to the channels over
        that hand's motor cortex, e.g. {"eeg": {"right_hand": ("C3",), "left_hand":
        ("C4",)}}. Every modality with a weight above 0 needs an entry.
    fnirs_hrf : dict[str, float]
        "task_regressors" only: the gamma HRF of the fNIRS reference, the keyword
        arguments of cedalion's glm.Gamma in seconds (tau, sigma, T).
    """
    method: str = "ground_truth"
    n_top: int = 2
    eeg_weight: float = 1.0
    fnirs_weight: float = 1.0
    motor_channels: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)
    fnirs_hrf: dict[str, float] = field(
        default_factory=lambda: {"tau": 0.0, "sigma": 3.0, "T": 0.0})

    def __post_init__(self):
        if self.n_top < 1:
            raise ValueError(f"n_top must be >= 1, got {self.n_top}.")
        if self.eeg_weight < 0 or self.fnirs_weight < 0:
            raise ValueError(f"eeg_weight and fnirs_weight must be >= 0, got "
                             f"{self.eeg_weight} and {self.fnirs_weight}.")
        if self.eeg_weight + self.fnirs_weight == 0:
            raise ValueError("eeg_weight and fnirs_weight cannot both be 0.")
        unknown = set(self.motor_channels) - {"eeg", "fnirs"}
        if unknown:
            raise ValueError(f"motor_channels takes 'eeg' and 'fnirs', got {sorted(unknown)}.")
        unknown = set(self.fnirs_hrf) - {"tau", "sigma", "T"}
        if unknown:
            raise ValueError(f"fnirs_hrf takes 'tau', 'sigma' and 'T', got {sorted(unknown)}.")
        if self.method != "motor_channels":
            return
        needed = [modality for modality, weight in (("eeg", self.eeg_weight),
                                                    ("fnirs", self.fnirs_weight))
                  if weight > 0 and modality not in self.motor_channels]
        if needed:
            raise ValueError(f"assignment.method motor_channels needs motor_channels for "
                             f"{needed}.")


@dataclass(frozen=True)
class UnmixingConfig:
    """
    Hyperparameter selection, fitting and scoring of the unmixing models.

    Non-nested evaluation always runs: the subject's test group is held out, the
    settings are selected on the other groups, and the selected model is scored on
    the test group. Nested evaluation repeats this with every group held out once,
    for analysis only.

    Attributes
    ----------
    methods : tuple[str, ...]
        Method names.
    grids : dict
        {method: {param: values}}, read from the grids file the experiment names.
    metrics : tuple[str, ...]
        Selection metrics; each picks one hyperparameter combination.
    nested : bool
        Also run nested evaluation.
    n_components : int | None
        Canonical pairs to fit. None = all.
    assignment : AssignmentConfig
        How the pairs are assigned to the hands.
    grid_overrides : dict
        {method: {param: values}} replacing parts of a method's grid.
    laplacian : dict
        Graph spec for the StructuredSparse* methods. Empty = default graph.
    seed : int
        Base seed; every fit derives its own seed from it.
    n_jobs : int
        Parallel fits. 1 = serial, -1 = all cores.
    """
    methods: tuple[str, ...]
    grids: dict
    metrics: tuple[str, ...] = ("joint_avg",)
    nested: bool = False
    n_components: int | None = None
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    grid_overrides: dict = field(default_factory=dict)
    laplacian: dict = field(default_factory=dict)
    seed: int = 0
    n_jobs: int = 1

    def __post_init__(self):
        if not self.methods:
            raise ValueError("methods must list at least one method.")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError(f"methods lists a method twice: {list(self.methods)}.")
        if not self.metrics:
            raise ValueError("metrics must list at least one metric.")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError(f"metrics lists a metric twice: {list(self.metrics)}.")
        if self.n_components is not None and self.n_components < self.assignment.n_top:
            raise ValueError(f"n_components ({self.n_components}) must be null or at least "
                             f"assignment.n_top ({self.assignment.n_top}).")
        if self.n_jobs < 1 and self.n_jobs != -1:
            raise ValueError(f"n_jobs must be -1 or a positive int, got {self.n_jobs}.")
        stray = set(self.grid_overrides) - set(self.methods)
        if stray:
            raise ValueError(f"grid_overrides name method(s) not in methods: {sorted(stray)}.")


@dataclass(frozen=True)
class ClassificationConfig:
    """
    rLDA classification of the trials from the recovered sources. Absent = skipped.

    Attributes
    ----------
    classes : tuple[str, ...]
        Trial types to tell apart, at least two. `rest_class` among them is taken
        from the stim table where a run has such trials, and otherwise cut from the
        gaps between the trials (see `src.classification.features.rest_trials`).
    rest_class : str
        The trial type that means rest.
    feature_sets : tuple[str, ...]
        Which components each classifier sees: "eeg_fnirs" (both), "eeg", "fnirs".
        Every feature set is classified on the same trials.
    baseline_window : tuple[float, float]
        [lo, hi] seconds relative to onset subtracted from each epoch.
    eeg_intervals, fnirs_intervals : tuple[tuple[float, float], ...]
        [t0, t1] seconds after onset; the mean of each interval is one feature.
    """
    classes: tuple[str, ...] = ("left", "right")
    rest_class: str = "rest"
    feature_sets: tuple[str, ...] = FEATURE_SETS
    baseline_window: tuple[float, float] = (-2.0, 0.0)
    eeg_intervals: tuple[tuple[float, float], ...] = ((0.5, 4.5), (5.5, 8.0))
    fnirs_intervals: tuple[tuple[float, float], ...] = ((2.0, 5.0), (5.0, 8.0), (8.0, 12.0))

    def __post_init__(self):
        if len(self.classes) < 2:
            raise ValueError(f"classes must list at least two classes, got {list(self.classes)}.")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError(f"classes lists a class twice: {list(self.classes)}.")
        if not self.feature_sets:
            raise ValueError("feature_sets must list at least one feature set.")
        if len(set(self.feature_sets)) != len(self.feature_sets):
            raise ValueError(f"feature_sets lists a set twice: {list(self.feature_sets)}.")
        for feature_set in self.feature_sets:
            _check_choice(feature_set, FEATURE_SETS, "feature_sets")
        _check_window(self.baseline_window, "baseline_window")
        for name in ("eeg_intervals", "fnirs_intervals"):
            intervals = getattr(self, name)
            if not intervals:
                raise ValueError(f"{name} must list at least one interval.")
            for interval in intervals:
                _check_window(interval, name)


# -------------------------------------------------------------------------
# Experiment and subject
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    """
    One experiment, independent of the subject it runs on.

    Attributes
    ----------
    name : str
        Experiment name; part of the results path.
    pruning : PruningConfig
    preprocessing : PreprocessingConfig
    unmixing : UnmixingConfig
    hand : str
        Which hand(s) the experiment is about: "left" | "right" | "both".
    classification : ClassificationConfig | None
        None = no classification.
    save_inputs : bool
        Also write the preprocessed runs every model was fit on to disk, to inspect
        them later. They go to data/preprocessed/, not to the results. The figures
        of a run never need them on disk: they use the runs in memory.
    """
    name: str
    pruning: PruningConfig
    preprocessing: PreprocessingConfig
    unmixing: UnmixingConfig
    hand: str = "both"
    classification: ClassificationConfig | None = None
    save_inputs: bool = True

    def __post_init__(self):
        _check_choice(self.hand, HANDS, "hand")
        n_hands = 2 if self.hand == "both" else 1
        if self.unmixing.assignment.n_top < n_hands:
            raise ValueError(f"unmixing.assignment.n_top ({self.unmixing.assignment.n_top}) "
                             f"must be at least the number of hands ({n_hands} for hand: "
                             f"{self.hand}).")


@dataclass(frozen=True)
class EegMontageConfig:
    """
    Where the EEG electrode positions come from: exactly one of `standard` and `file`.

    Attributes
    ----------
    standard : str | None
        Name of an MNE template montage, e.g. "standard_1020".
    file : Path | None
        A montage file that comes with the dataset, in a format
        `mne.channels.read_custom_montage` reads (.elc, .sfp, .bvef, ...).
    """
    standard: str | None = None
    file: Path | None = None

    def __post_init__(self):
        if (self.standard is None) == (self.file is None):
            raise ValueError("montage.eeg must set exactly one of 'standard' and 'file'.")


@dataclass(frozen=True)
class FnirsMontageConfig:
    """
    Where the fNIRS optode positions come from: exactly one of `recording`, `file` and
    `layout`; `layout` together with `standard`.

    Attributes
    ----------
    recording : bool
        True: the positions stored in the recording's own files (e.g. the SNIRF).
    file : Path | None
        An optode position file, as `cedalion.io.load_tsv` reads it.
    layout : Path | None
        A tab-separated file with the columns channel, source, detector: the optodes
        of every channel, named by the template sites they sit on (e.g. "FCC5h").
        Lines starting with "#" are comments.
    standard : str | None
        With `layout`: the MNE template montage that places the sites, e.g.
        "standard_1005".
    """
    recording: bool = False
    file: Path | None = None
    layout: Path | None = None
    standard: str | None = None

    def __post_init__(self):
        if sum([self.recording, self.file is not None, self.layout is not None]) != 1:
            raise ValueError("montage.fnirs must set exactly one of 'recording: true', "
                             "'file' and 'layout'.")
        if (self.layout is None) != (self.standard is None):
            raise ValueError("montage.fnirs.layout and montage.fnirs.standard go together: "
                             "the layout names template sites, the standard places them.")


@dataclass(frozen=True)
class MontageConfig:
    """
    Electrode and optode positions: the only place they are chosen. Loading applies
    them once; after that they travel with the data.

    Attributes
    ----------
    eeg : EegMontageConfig
    fnirs : FnirsMontageConfig | None
        None while the dataset has no known optode positions; loading then refuses
        the subject.
    """
    eeg: EegMontageConfig
    fnirs: FnirsMontageConfig | None = None


@dataclass(frozen=True)
class SubjectConfig:
    """
    One subject's data, synthetic or real. The pipeline treats both alike; only the
    loader chosen by `format` differs.

    Attributes
    ----------
    subject_id : str
    name : str
        Name of this dataset, e.g. "sub01_snr_fnirs-5_eeg5": the subject file's name
        unless the file sets one. Part of the results path, so runs of one subject at
        different SNRs are kept apart.
    format : str
        Which reader loads the data, e.g. "synthetic" | "shin2017" | "buccino".
    paths : dict[str, Path]
        Data locations; which keys exist depends on the format.
    groups : tuple[str, ...]
        CV groups to use, e.g. background recordings or sessions.
    montage : MontageConfig
        Electrode and optode positions.
    test_group : str | None
        Group held out as the test set in non-nested evaluation; needed to run unmixing.
    trial_classes : dict[str, tuple[str, ...]]
        Side ("right", "left") to the stim trial types that make up its class, e.g.
        {"right": ("right", "right_arm"), "left": ("left", "left_arm")}. Unmixing
        assigns a pair to the hand of each side that has trials. Trial types in no
        class (e.g. rest) belong to no hand.
    simulation : dict | None
        Synthetic only: the simulator's config, as it saved it next to the data.
    """
    subject_id: str
    name: str
    format: str
    paths: dict[str, Path]
    groups: tuple[str, ...]
    montage: MontageConfig
    test_group: str | None = None
    trial_classes: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"right": ("right",), "left": ("left",)})
    simulation: dict | None = None

    def __post_init__(self):
        if not self.groups:
            raise ValueError("groups must list at least one group.")
        if len(set(self.groups)) != len(self.groups):
            raise ValueError(f"groups lists a group twice: {list(self.groups)}.")
        if self.test_group is not None and self.test_group not in self.groups:
            raise ValueError(f"test_group {self.test_group!r} is not in groups {list(self.groups)}.")
        unknown = set(self.trial_classes) - set(SIDES)
        if unknown:
            raise ValueError(f"trial_classes takes the sides {list(SIDES)}, got {sorted(unknown)}.")
        trial_types = [t for types in self.trial_classes.values() for t in types]
        if len(set(trial_types)) != len(trial_types):
            raise ValueError(f"trial_classes puts a trial type in two classes: "
                             f"{dict(self.trial_classes)}.")


def check_subject_fits(cfg: ExperimentConfig, subject: SubjectConfig):
    """
    Checks that need both configs. Each config is checked on its own when it is
    loaded; these only fail for a particular combination of experiment and subject.

    Raises
    ------
    ValueError
        If the subject cannot run the experiment.
    """
    if subject.test_group is None:
        raise ValueError(f"Experiment '{cfg.name}' holds out the subject's test_group in "
                         f"non-nested evaluation, but subject '{subject.subject_id}' sets none.")
