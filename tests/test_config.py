"""
Tests for the unmixing part of the config: the pairs section, the grids file the
experiment points to, and the checks that tie them to the rest of the experiment.
Also the subject's trial classes, which name the hands unmixing assigns pairs to.
"""
import dataclasses
import glob
from pathlib import Path

import pytest
import yaml

from src.config import AssignmentConfig, UnmixingConfig, load_experiment, load_subject

# Fixture configs, so no test depends on the real experiments.
CONFIGS = Path(__file__).parent / "configs"
EXPERIMENT = CONFIGS / "experiment.yaml"


def unmixing(**changes) -> UnmixingConfig:
    return UnmixingConfig(**{"methods": ("CCA",), "grids": {"CCA": {}}, **changes})


@pytest.mark.parametrize("path", sorted(glob.glob("configs/experiments/**/*.yaml",
                                                  recursive=True)))
def test_every_experiment_loads(path):
    cfg = load_experiment(path)
    assert set(cfg.unmixing.methods) <= set(cfg.unmixing.grids)


def test_grids_are_read_from_the_grids_file():
    cfg = load_experiment(EXPERIMENT)
    with open(CONFIGS / "grids.yaml") as f:
        assert cfg.unmixing.grids == yaml.safe_load(f)


def test_grids_must_name_a_file(tmp_path):
    with open(EXPERIMENT) as f:
        raw = yaml.safe_load(f)
    raw["pruning"] = str(CONFIGS / "pruning.yaml")
    raw["preprocessing"] = str(CONFIGS / "preprocessing.yaml")
    raw["unmixing"]["grids"] = {"CCA": {}}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unmixing.grids"):
        load_experiment(path)


def write_motor_channel_experiment(tmp_path, motor_channels) -> Path:
    """The test experiment with the motor_channels assignment, written to tmp_path."""
    with open(EXPERIMENT) as f:
        raw = yaml.safe_load(f)
    raw["pruning"] = str(CONFIGS / "pruning.yaml")
    raw["preprocessing"] = str(CONFIGS / "preprocessing.yaml")
    raw["unmixing"]["grids"] = str(CONFIGS / "grids.yaml")
    raw["unmixing"]["assignment"] = {"method": "motor_channels",
                                     "motor_channels": motor_channels}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_motor_channels_are_read_from_their_file(tmp_path):
    channels = {"eeg": {"right_hand": ["C3"], "left_hand": ["C4"]},
                "fnirs": {"right_hand": ["S1D1"], "left_hand": ["S2D2"]}}
    (tmp_path / "motor.yaml").write_text(yaml.safe_dump(channels))
    cfg = load_experiment(write_motor_channel_experiment(tmp_path, "motor.yaml"))
    assert cfg.unmixing.assignment.motor_channels == {
        "eeg": {"right_hand": ("C3",), "left_hand": ("C4",)},
        "fnirs": {"right_hand": ("S1D1",), "left_hand": ("S2D2",)}}


def test_motor_channels_must_name_a_file(tmp_path):
    path = write_motor_channel_experiment(tmp_path, {"eeg": {"right_hand": ["C3"]}})
    with pytest.raises(ValueError, match="unmixing.assignment.motor_channels"):
        load_experiment(path)


def test_assignment_defaults():
    assignment = unmixing().assignment
    assert (assignment.method, assignment.n_top, assignment.eeg_weight,
            assignment.fnirs_weight) == ("ground_truth", 2, 1.0, 1.0)


@pytest.mark.parametrize("kwargs", [{"n_top": 0}, {"eeg_weight": -1.0},
                                    {"eeg_weight": 0.0, "fnirs_weight": 0.0}])
def test_assignment_rejects(kwargs):
    with pytest.raises(ValueError):
        AssignmentConfig(**kwargs)


def test_n_components_must_cover_n_top():
    unmixing(n_components=None)
    unmixing(n_components=2)
    with pytest.raises(ValueError, match="n_components"):
        unmixing(n_components=1)


def test_metric_listed_twice_is_rejected():
    with pytest.raises(ValueError, match="twice"):
        unmixing(metrics=("joint_avg", "joint_avg"))


def test_n_top_must_cover_the_sources_of_the_hand():
    cfg = load_experiment(EXPERIMENT)
    one_pair = dataclasses.replace(cfg.unmixing, assignment=AssignmentConfig(n_top=1))
    dataclasses.replace(cfg, hand="left", unmixing=one_pair)
    with pytest.raises(ValueError, match="n_top"):
        dataclasses.replace(cfg, hand="both", unmixing=one_pair)


def test_trial_classes_default_to_the_hand_trials():
    subject = load_subject(CONFIGS / "subject.yaml")
    assert subject.trial_classes == {"right": ("right",), "left": ("left",)}


@pytest.mark.parametrize("trial_classes, message", [
    ({"right": ("right",), "up": ("up",)}, "sides"),
    ({"right": ("right", "arm"), "left": ("left", "arm")}, "two classes"),
])
def test_trial_classes_rejects(trial_classes, message):
    subject = load_subject(CONFIGS / "subject.yaml")
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(subject, trial_classes=trial_classes)


@pytest.mark.parametrize("path", sorted(glob.glob("configs/subjects/buccino/*/*.yaml")))
def test_buccino_subjects_merge_hand_and_arm_trials_per_side(path):
    assert load_subject(path).trial_classes == {"right": ("right", "right_arm"),
                                                "left": ("left", "left_arm")}
