"""
The unmixing methods: which cedalion class each name builds, what input it needs,
and its hyperparameter grid.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import ParameterGrid

from cedalion.sigdecomp.multimodal.cca import (CCA, ElasticNetCCA, RidgeCCA, SparseCCA,
                                               StructuredSparseCCA)
from cedalion.sigdecomp.multimodal.mspoc import mSPoC
from cedalion.sigdecomp.multimodal.tcca import (ElasticNetTCCA, StructuredSparseTCCA,
                                                tCCA)

from src.config import UnmixingConfig
from src.unmixing.assignment import STRATEGIES
from src.unmixing.laplacian import GRAPH_BUILDERS
from src.unmixing.metrics import check_metrics


@dataclass(frozen=True)
class Method:
    """
    Attributes
    ----------
    cls : type
        The cedalion class.
    hyperparameters : tuple[str, ...]
        The keys its grid may contain; anything else is rejected. Fixed settings
        (shift_source, scale, N_components, the Laplacians, the seed) are never grid
        keys.
    eeg_feature : str
        The preprocessing's `eeg_feature` the method needs.
    temporal : bool
        Embeds the EEG in time (`time_shifts`) and finds an optimal shift per
        component; `shift_to_eeg_time` shifts the EEG component back by it.
    structured : bool
        Takes the graph Laplacians Lx and Ly.
    """
    cls: type
    hyperparameters: tuple[str, ...] = ()
    eeg_feature: str = "bandpower"
    temporal: bool = False
    structured: bool = False


METHODS = {
    "CCA": Method(CCA),
    "SparseCCA": Method(SparseCCA, ("l1_reg",)),
    "RidgeCCA": Method(RidgeCCA, ("l2_reg",)),
    "ElasticNetCCA": Method(ElasticNetCCA, ("l1_reg", "l2_reg")),
    "StructuredSparseCCA": Method(StructuredSparseCCA, ("l1_reg", "l2_reg"), structured=True),
    "tCCA": Method(tCCA, ("time_shifts",), temporal=True),
    "ElasticNetTCCA": Method(ElasticNetTCCA, ("l1_reg", "l2_reg", "time_shifts"), temporal=True),
    "StructuredSparseTCCA": Method(StructuredSparseTCCA, ("l1_reg", "l2_reg", "time_shifts"),
                                   temporal=True, structured=True),
    "mSPoC": Method(mSPoC, ("time_shifts", "max_iter", "N_restarts", "tol"),
                    eeg_feature="bandpassed", temporal=True),
}

# Grid keys given per modality ({eeg: [...], fnirs: [...]}); every pair is tried.
PER_MODALITY = ("l1_reg", "l2_reg")


def check_config(cfg: UnmixingConfig, eeg_feature: str):
    """
    Check the parts of the unmixing config that name things this module owns: the
    methods, their grids, the input they need, the metrics, the assignment strategy
    and the Laplacian spec.
    Run before any data is read.

    Parameters
    ----------
    cfg : UnmixingConfig
    eeg_feature : str
        The preprocessing's EEG feature, "bandpower" or "bandpassed".

    Raises
    ------
    ValueError
        Naming the method or setting at fault.
    """
    check_metrics(cfg.metrics)
    unknown = set(cfg.methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown method(s) {sorted(unknown)}; known: {sorted(METHODS)}.")
    for name in cfg.methods:
        method = METHODS[name]
        if method.eeg_feature != eeg_feature:
            raise ValueError(f"{name} needs eeg_feature: {method.eeg_feature} in the "
                             f"preprocessing file, got {eeg_feature}.")
        if name not in cfg.grids:
            raise ValueError(f"The grids file has no grid for {name}.")
        param_grid(name, cfg)
    if "mSPoC" in cfg.methods and cfg.n_components is None:
        raise ValueError("mSPoC needs n_components as a number: cedalion's mSPoC cannot "
                         "fit all components (n_components: null).")
    if cfg.assignment.method not in STRATEGIES:
        raise ValueError(f"assignment.method must be one of {sorted(STRATEGIES)}, got "
                         f"{cfg.assignment.method!r}.")
    unknown = set(cfg.laplacian) - {"eeg", "fnirs"}
    if unknown:
        raise ValueError(f"laplacian takes 'eeg' and 'fnirs' blocks, got {sorted(unknown)}.")
    for modality, block in cfg.laplacian.items():
        if block.get("method") not in GRAPH_BUILDERS:
            raise ValueError(f"laplacian.{modality}.method must be one of "
                             f"{sorted(GRAPH_BUILDERS)}, got {block.get('method')!r}.")


def param_grid(name: str, cfg: UnmixingConfig) -> list[dict]:
    """
    Every hyperparameter combination of a method: its grid from the grids file, with
    the experiment's `grid_overrides` replacing whole keys. Only the method's
    `hyperparameters` are accepted as keys.

    Parameters
    ----------
    name : str
        Method name.
    cfg : UnmixingConfig

    Returns
    -------
    list[dict]
        One dict of model arguments per combination; [{}] for a method without
        hyperparameters. `l1_reg`/`l2_reg` become [eeg, fnirs] and `time_shifts`
        arrays, as cedalion takes them.
    """
    grid = {**cfg.grids.get(name, {}), **cfg.grid_overrides.get(name, {})}
    allowed = METHODS[name].hyperparameters
    unknown = [key for key in grid if key not in allowed]
    if unknown:
        raise ValueError(f"{name} has no hyperparameter {unknown[0]!r}; its grid takes: "
                         f"{', '.join(allowed) or 'nothing'}.")
    values = {}
    for key, value in grid.items():
        if key in PER_MODALITY:
            if not isinstance(value, dict) or set(value) != {"eeg", "fnirs"}:
                raise ValueError(f"{name}.{key} must give 'eeg' and 'fnirs' lists, got {value!r}.")
            values[key] = [[x, y] for x in value["eeg"] for y in value["fnirs"]]
        elif key == "time_shifts":
            values[key] = [np.asarray(shifts, dtype=float) for shifts in value]
        else:
            if not isinstance(value, list):
                raise ValueError(f"{name}.{key} must be a list of values, got {value!r}.")
            values[key] = value
    if METHODS[name].temporal and "time_shifts" not in values:
        raise ValueError(f"{name} needs time_shifts in its grid.")
    return list(ParameterGrid(values)) if values else [{}]


def build_model(name: str, params: dict, n_components: int | None,
                laplacians: tuple | None = None):
    """
    An unfitted cedalion model.

    Parameters
    ----------
    name : str
        Method name.
    params : dict
        One combination from `param_grid`.
    n_components : int | None
        Pairs to fit; None = all.
    laplacians : tuple | None
        (Lx, Ly) for the structured methods.

    Returns
    -------
    The cedalion model. Temporal models do not shift their EEG components
    (`shift_source=False`), so `transform` gives each pair as it was fitted.
    """
    method = METHODS[name]
    kwargs = dict(N_components=n_components, scale=True, **params)
    if method.structured:
        if laplacians is None:
            raise ValueError(f"{name} needs the graph Laplacians.")
        kwargs["Lx"], kwargs["Ly"] = laplacians
    if method.temporal:
        kwargs["shift_source"] = False
    return method.cls(**kwargs)
