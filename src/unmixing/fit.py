"""
Fit the unmixing models and apply them.
"""
import warnings

import numpy as np
import xarray as xr

from src.types import FittedModel, JoinedRuns
from src.unmixing.models import METHODS, build_model
from src.unmixing.orient import spatial_filter


class DegenerateFit(ValueError):
    """
    The fitted model has NaN filters, so its components are NaN on any data. Seen
    with sparse CCA when the L1 soft threshold removes every weight of a filter:
    cedalion then normalises a zero vector and the NaN carries through the deflation
    into every later pair. A setting that gives this is excluded from selection.
    """


def fit_model(name: str, params: dict, data: JoinedRuns, n_components: int | None,
              laplacians: tuple | None = None, seed: int | None = None) -> FittedModel:
    """
    Fit one method on the joined training runs. The components keep the sign the
    model gives them: every score is sign-free, and only the plots orient them.

    Parameters
    ----------
    name : str
        Method name.
    params : dict
        One combination from `param_grid`.
    data : JoinedRuns
        Training runs.
    n_components : int | None
        Pairs to fit; None = all.
    laplacians : tuple | None
        (Lx, Ly), for the structured methods.
    seed : int | None
        Seeds numpy's global generator, which cedalion's mSPoC restarts draw from.

    Returns
    -------
    FittedModel

    Raises
    ------
    DegenerateFit
        If any pair's EEG or fNIRS filter is not finite.
    """
    if seed is not None:
        np.random.seed(seed)
    model = build_model(name, params, n_components, laplacians)
    if name == "mSPoC":
        check_blocks(data.eeg, data.fnirs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if METHODS[name].temporal:
            model.fit(data.eeg, data.fnirs, featureX_name="channel", featureY_name="channel")
        else:
            model.fit(data.eeg, data.fnirs, sample_name="time", featureX_name="channel",
                      featureY_name="channel")

    W_eeg, W_fnirs = spatial_filter(model, "eeg"), spatial_filter(model, "fnirs")
    bad = [int(k) for k in W_eeg.component.values
           if not (np.isfinite(W_eeg.sel(component=k)).all()
                   and np.isfinite(W_fnirs.sel(component=k)).all())]
    if bad:
        raise DegenerateFit(f"{name} {params}: the filters of {len(bad)} of "
                            f"{W_eeg.sizes['component']} pairs are NaN (pairs {bad[0]} to "
                            f"{bad[-1]}); for sparse methods, the L1 threshold removed every "
                            f"weight of a filter.")
    return FittedModel(method=name, params=params, model=model)


def transform(fitted: FittedModel, data: JoinedRuns) -> tuple[xr.DataArray, xr.DataArray]:
    """
    The EEG and fNIRS components of `data` as the model pairs them, with the sign
    the model gives them.

    For temporal models u(t) = sum over tau of w_tau' x(t - tau), the time-embedded
    EEG filtered to match v(t): it is on the fNIRS time, tau_k after the EEG it
    reflects. `shift_to_eeg_time` moves it to the time of that EEG.

    Parameters
    ----------
    fitted : FittedModel
    data : JoinedRuns

    Returns
    -------
    u, v : xr.DataArray
        EEG and fNIRS components, dims (time, component), on the fNIRS time points
        of `data`, every one of them.

    Raises
    ------
    ValueError
        If a temporal model shifts its own EEG components (`shift_source=True`), as
        models saved by an earlier version of the pipeline do.
    """
    model = fitted.model
    if getattr(model, "shift_source", False):
        raise ValueError(f"The {fitted.method} model shifts its own EEG components "
                         f"(shift_source=True); refit it.")
    if fitted.method == "mSPoC":
        check_blocks(data.eeg, data.fnirs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u, v = model.transform(data.eeg, data.fnirs)
    return (_components(u, model.latent_featureX_name),
            _components(v, model.latent_featureY_name))


def shift_to_eeg_time(fitted: FittedModel, u: xr.DataArray) -> xr.DataArray:
    """
    The EEG components from `transform` on the time of the EEG they came from:
    component k shifted back by its optimal shift tau_k, u_eeg(t) = u(t + tau_k),
    with cedalion's own `shift_by_optimal`. The last tau_k of component k has no
    sample to come from; cedalion fills it with zeros, which are set to NaN here, so
    they never count in a correlation. Non-temporal models are returned unchanged.

    Parameters
    ----------
    fitted : FittedModel
    u : xr.DataArray
        EEG components from `transform`, dims (time, component).

    Returns
    -------
    xr.DataArray
        Dims (time, component), on the time points of `u`.
    """
    if not METHODS[fitted.method].temporal:
        return u
    shifted = fitted.model.shift_by_optimal(u)
    time = u.time.values
    n_shift = [shift_samples(time, shift) for shift in np.atleast_1d(fitted.model.optimal_shift)]
    beyond_data = np.stack([np.arange(len(time)) >= len(time) - n for n in n_shift], axis=1)
    return shifted.where(~beyond_data)


def shift_samples(time: np.ndarray, shift: float) -> int:
    """A time shift (s) in samples of `time`, rounded the way cedalion rounds it."""
    return int(np.searchsorted(time, time[0] + shift))


def _components(da: xr.DataArray, latent: str) -> xr.DataArray:
    """Dequantified, dims (time, component), components numbered from 0."""
    if getattr(da.pint, "units", None) is not None:
        da = da.pint.dequantify()
    da = da.transpose("time", latent).rename({latent: "component"})
    da = da.drop_vars([c for c in da.coords if c not in ("time",)])
    return da.assign_coords(component=np.arange(da.sizes["component"]))


# How far (as a fraction of one block) an EEG block may start from its fNIRS sample.
# The recorded EEG rates are not exact (e.g. 250.0032 Hz for 250 Hz), so
# blocks of k samples drift by a few ms over a run; the error this guards against
# was about one whole block.
BLOCK_TOLERANCE = 0.05


def check_blocks(eeg: xr.DataArray, fnirs: xr.DataArray):
    """
    Raise unless the EEG is cut into one block of k samples per fNIRS sample, with
    block i starting at fNIRS sample i (within BLOCK_TOLERANCE of a block),
    k = EEG rate / fNIRS rate.

    mSPoC pairs EEG block i with fNIRS sample i by position only: it computes the
    block length as (EEG samples) // (fNIRS samples) and cuts blocks from the first
    EEG sample on, without looking at either time axis or checking the counts. Any
    other input is fitted without an error, with every block shifted against its
    fNIRS sample. Called on exactly the data handed to mSPoC.

    Parameters
    ----------
    eeg : xr.DataArray
        Band-passed EEG, dims (time, channel).
    fnirs : xr.DataArray
        HbO, dims (time, channel), at the lower rate.

    Raises
    ------
    ValueError
        Naming what is wrong.
    """
    eeg_dt = float(np.median(np.diff(eeg.time.values)))
    fnirs_dt = float(np.median(np.diff(fnirs.time.values)))
    k = round(fnirs_dt / eeg_dt)
    n_eeg, n_fnirs = eeg.sizes["time"], fnirs.sizes["time"]
    if k < 2 or abs(fnirs_dt / eeg_dt - k) > 1e-3 * k:
        raise ValueError(f"The fNIRS rate ({1 / fnirs_dt:.4g} Hz) must divide the EEG rate "
                         f"({1 / eeg_dt:.4g} Hz) into blocks of at least 2 EEG samples.")
    if n_eeg != k * n_fnirs:
        raise ValueError(f"mSPoC needs {k} EEG samples per fNIRS sample, {k * n_fnirs} in "
                         f"all, got {n_eeg}; it would cut blocks of {n_eeg // n_fnirs}.")
    offsets = eeg.time.values[::k] - fnirs.time.values
    worst = int(np.argmax(np.abs(offsets)))
    if abs(offsets[worst]) > BLOCK_TOLERANCE * fnirs_dt:
        raise ValueError(f"EEG block {worst} starts at {eeg.time.values[::k][worst]:.3f} s, "
                         f"but its fNIRS sample is at {fnirs.time.values[worst]:.3f} s.")
