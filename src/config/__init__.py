"""
Experiment and subject configuration.

Example
-------
    cfg = load_experiment("configs/experiments/default.yaml")
    subject = load_subject("configs/subjects/synthetic/sub01/sub01_snr_fnirs-5_eeg5.yaml")

`cfg` has one section per module (cfg.pruning, cfg.preprocessing, cfg.unmixing,
cfg.classification); each module reads only its own. The two snapshots are what a
run saves into its results folder.
"""
from src.config.load import (experiment_from_snapshot, experiment_snapshot,
                             load_experiment, load_pruning, load_subject,
                             subject_from_snapshot, subject_snapshot)
from src.config.schema import (AlignmentCheckConfig, ClassificationConfig,
                               EegPreprocessingConfig,
                               EegMontageConfig, EegPruningConfig, EegPruningPass1,
                               EegPruningPass2, ExperimentConfig, FnirsMontageConfig,
                               FnirsPreprocessingConfig, FnirsPruningConfig,
                               AssignmentConfig, MontageConfig, PreprocessingConfig,
                               PruningConfig, SubjectConfig, UnmixingConfig,
                               check_subject_fits)

__all__ = [
    "load_experiment", "load_pruning", "load_subject", "experiment_snapshot", "subject_snapshot",
    "experiment_from_snapshot", "subject_from_snapshot", "check_subject_fits",
    "ExperimentConfig", "SubjectConfig",
    "MontageConfig", "EegMontageConfig", "FnirsMontageConfig",
    "PruningConfig", "EegPruningConfig", "EegPruningPass1", "EegPruningPass2",
    "FnirsPruningConfig",
    "PreprocessingConfig", "EegPreprocessingConfig", "FnirsPreprocessingConfig",
    "AlignmentCheckConfig",
    "UnmixingConfig", "AssignmentConfig", "ClassificationConfig",
]
