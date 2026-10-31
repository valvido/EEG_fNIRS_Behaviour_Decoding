"""
Read the experiment and subject YAML files into the config dataclasses.

An experiment is four files:
- the main file: tuning knobs (unmixing, classification), changed every run
- a pruning file, a preprocessing file and a grids file, which rarely change

The main file points to the other three (the grids file from `unmixing.grids`). Every
key lives in exactly one of the files, so a key in the wrong file is an unknown key
there and is rejected. The "motor_channels" assignment also points to a motor channels
file (`unmixing.assignment.motor_channels`), one per montage.

Relative paths to the referenced files resolve against the main file's directory;
data paths in a subject file resolve against the project root.
"""
import dataclasses
import types
import typing
from pathlib import Path

import yaml

from src.config.schema import (ExperimentConfig, PreprocessingConfig, PruningConfig,
                               SubjectConfig)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Written by the simulator next to the data it exports.
SIMULATION_FILE = "simulation_config.yaml"

# Sections of the main experiment file that name another YAML file.
_REFERENCED_SECTIONS = {"pruning": PruningConfig, "preprocessing": PreprocessingConfig}


def load_experiment(path: str | Path) -> ExperimentConfig:
    """
    Load the main experiment YAML and the pruning, preprocessing, grids and motor
    channels files it points to.

    Parameters
    ----------
    path : str | Path
        Main experiment YAML, with `pruning:`, `preprocessing:` and `unmixing.grids:`
        naming the other three files relative to it, and optionally
        `unmixing.assignment.motor_channels:` naming a motor channels file.

    Returns
    -------
    ExperimentConfig
    """
    path = Path(path)
    raw = _read_yaml(path)
    for section, cls in _REFERENCED_SECTIONS.items():
        ref = raw.get(section)
        if not isinstance(ref, str):
            raise ValueError(f"{path}: '{section}' must be the path of a {section} YAML "
                             f"file, got {ref!r}.")
        ref_path = (path.parent / ref).resolve()
        raw[section] = _build(cls, _read_yaml(ref_path), str(ref_path))

    unmixing = raw.get("unmixing")
    if isinstance(unmixing, dict):
        ref = unmixing.get("grids")
        if not isinstance(ref, str):
            raise ValueError(f"{path}: 'unmixing.grids' must be the path of a grids YAML "
                             f"file, got {ref!r}.")
        raw["unmixing"] = {**unmixing, "grids": _read_yaml((path.parent / ref).resolve())}

        assignment = unmixing.get("assignment")
        if isinstance(assignment, dict) and "motor_channels" in assignment:
            ref = assignment["motor_channels"]
            if not isinstance(ref, str):
                raise ValueError(f"{path}: 'unmixing.assignment.motor_channels' must be the "
                                 f"path of a motor channels YAML file, got {ref!r}.")
            channels = _read_yaml((path.parent / ref).resolve())
            raw["unmixing"]["assignment"] = {**assignment, "motor_channels": channels}
    return _build(ExperimentConfig, raw, str(path))


def load_pruning(path: str | Path) -> PruningConfig:
    """
    Load a pruning YAML on its own, without an experiment file pointing to it.

    Parameters
    ----------
    path : str | Path
        Pruning YAML, e.g. configs/pruning/default.yaml.

    Returns
    -------
    PruningConfig
    """
    path = Path(path)
    return _build(PruningConfig, _read_yaml(path), str(path))


def load_subject(path: str | Path) -> SubjectConfig:
    """
    Load a subject YAML.

    `name` is the file's name without extension, unless the file sets `name`. For
    synthetic data, the simulator's config found next to the data
    ({paths.data}/simulation_config.yaml) is attached as `simulation`.

    Parameters
    ----------
    path : str | Path
        Subject YAML.

    Returns
    -------
    SubjectConfig
    """
    path = Path(path)
    raw = _read_yaml(path)
    if "simulation" in raw:
        raise ValueError(f"{path}: 'simulation' is read from {{paths.data}}/{SIMULATION_FILE}, "
                         "not written in the subject file.")
    subject = _build(SubjectConfig, {"name": path.stem, **raw}, str(path))

    data_dir = subject.paths.get("data")
    if data_dir is not None and (data_dir / SIMULATION_FILE).is_file():
        subject = dataclasses.replace(subject, simulation=_read_yaml(data_dir / SIMULATION_FILE))
    return subject


def experiment_from_snapshot(raw: dict) -> ExperimentConfig:
    """The ExperimentConfig an `experiment_snapshot` was taken of."""
    return _build(ExperimentConfig, raw, "experiment snapshot")


def subject_from_snapshot(raw: dict) -> SubjectConfig:
    """The SubjectConfig a `subject_snapshot` was taken of."""
    return _build(SubjectConfig, raw, "subject snapshot")


def experiment_snapshot(cfg: ExperimentConfig) -> dict:
    """
    Every resolved experiment value, defaults included, as plain YAML-ready data.
    """
    return _plain(dataclasses.asdict(cfg))


def subject_snapshot(subject: SubjectConfig) -> dict:
    """
    The subject's values and its simulation config, as plain YAML-ready data.
    """
    return _plain(dataclasses.asdict(subject))


# -------------------------------------------------------------------------
# YAML -> dataclass
# -------------------------------------------------------------------------
def _read_yaml(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a YAML mapping, got {type(raw).__name__}.")
    return raw


def _build(cls, raw: dict, where: str):
    """
    `cls(**raw)`, with unknown keys, missing keys and wrong types reported together
    with `where` (the file, and the key path inside it).
    """
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping, got {type(raw).__name__}.")
    fields = {f.name: f for f in dataclasses.fields(cls)}

    unknown = set(raw) - set(fields)
    if unknown:
        raise ValueError(f"Unknown key(s) in {where}: {sorted(unknown)}. "
                         f"Known keys: {sorted(fields)}.")
    missing = [name for name, f in fields.items() if name not in raw
               and f.default is dataclasses.MISSING
               and f.default_factory is dataclasses.MISSING]
    if missing:
        raise ValueError(f"Missing required key(s) in {where}: {missing}.")

    hints = typing.get_type_hints(cls)
    kwargs = {name: _coerce(value, hints[name], f"{where}: {name}")
              for name, value in raw.items()}
    try:
        return cls(**kwargs)
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from None


def _coerce(value, hint, where: str):
    """
    Convert a YAML value to the type `hint` names.

    - nested mappings -> their dataclass
    - lists -> tuples
    - strings -> Paths, resolved against the project root
    - ints are accepted where a float is expected
    """
    if dataclasses.is_dataclass(hint):
        return value if isinstance(value, hint) else _build(hint, value, where)

    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        (inner,) = [a for a in args if a is not type(None)]
        return _coerce(value, inner, where)

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where} must be a list, got {value!r}.")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], f"{where}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ValueError(f"{where} must have {len(args)} entries, got {value!r}.")
        return tuple(_coerce(v, a, f"{where}[{i}]") for i, (v, a) in enumerate(zip(value, args)))

    if origin is dict or hint is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{where} must be a mapping, got {value!r}.")
        if not args:
            return value
        key_type, value_type = args
        return {_coerce(k, key_type, where): _coerce(v, value_type, f"{where}.{k}")
                for k, v in value.items()}

    if hint is Path:
        if not isinstance(value, str):
            raise ValueError(f"{where} must be a path, got {value!r}.")
        path = Path(value).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where} must be a number, got {value!r}.")
        return float(value)
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where} must be an integer, got {value!r}.")
        return value
    if hint in (bool, str):
        if not isinstance(value, hint):
            raise ValueError(f"{where} must be a {hint.__name__}, got {value!r}.")
        return value
    raise TypeError(f"{where}: no YAML conversion for type {hint!r}.")


def _plain(value):
    """
    Tuples -> lists and Paths -> strings, recursively, so yaml.safe_dump accepts it.
    Paths inside the project are written relative to it, so a snapshot still finds
    the data after the project moves to another folder or machine.
    """
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value.relative_to(PROJECT_ROOT) if value.is_relative_to(PROJECT_ROOT)
                   else value)
    return value
