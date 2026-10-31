"""
A fitted model as plain arrays and back, so it can be saved and applied again later
without refitting.

Every attribute of the cedalion model is kept: arrays (filters, normalisation,
shifts) as dataset variables, everything else as JSON. Rebuilding sets the same
attributes on a new object of the same class, so `transform` reproduces the fitted
model exactly, whatever the method.
"""
import json

import numpy as np
import xarray as xr

from src.types import FittedModel
from src.unmixing.models import METHODS

# Arrays on the EEG (X) and fNIRS (Y) side; their "channel" dims are renamed apart,
# since the two modalities have different channels.
_EEG_SIDE = ("Wx", "x_mean", "x_std")
_FNIRS_SIDE = ("Wy", "y_mean", "y_std")


def fitted_to_dataset(fitted: FittedModel) -> xr.Dataset:
    """
    Parameters
    ----------
    fitted : FittedModel

    Returns
    -------
    xr.Dataset
        Variables "model.<attribute>" for the model's arrays. Attributes: method,
        params and the model's other attributes, as JSON.
    """
    variables, scalars, nones = {}, {}, []
    for key, value in vars(fitted.model).items():
        if value is None:
            nones.append(key)
        elif isinstance(value, xr.DataArray):
            variables[f"model.{key}"] = _apart(_plain(value), key)
        elif isinstance(value, np.ndarray):
            variables[f"model.{key}"] = xr.DataArray(
                value, dims=[f"{key}_{i}" for i in range(value.ndim)])
        else:
            scalars[key] = value
    ds = xr.Dataset({k: _netcdf3(v) for k, v in variables.items()})
    ds.attrs = {"method": fitted.method, "params": _to_json(fitted.params),
                "model_attributes": _to_json(scalars), "model_none": json.dumps(nones)}
    return ds


def fitted_from_dataset(ds: xr.Dataset) -> FittedModel:
    """The FittedModel `fitted_to_dataset` was given, with a working model."""
    method = ds.attrs["method"]
    model = METHODS[method].cls.__new__(METHODS[method].cls)
    for key, value in json.loads(ds.attrs["model_attributes"]).items():
        setattr(model, key, value)
    for key in json.loads(ds.attrs["model_none"]):
        setattr(model, key, None)
    for name in ds.data_vars:
        if name.startswith("model."):
            key = name.removeprefix("model.")
            da = _together(ds[name], key)
            if all(dim.startswith(f"{key}_") for dim in da.dims) and not da.coords:
                setattr(model, key, da.values)
            else:
                setattr(model, key, da)
    params = json.loads(ds.attrs["params"])
    if "time_shifts" in params:
        params["time_shifts"] = np.asarray(params["time_shifts"], dtype=float)
    return FittedModel(method=method, params=params, model=model)


def _apart(da: xr.DataArray, key: str) -> xr.DataArray:
    side = "eeg" if key in _EEG_SIDE else "fnirs" if key in _FNIRS_SIDE else None
    return da.rename(channel=f"channel_{side}") if side and "channel" in da.dims else da


def _together(da: xr.DataArray, key: str) -> xr.DataArray:
    for side in ("eeg", "fnirs"):
        if f"channel_{side}" in da.dims:
            da = da.rename({f"channel_{side}": "channel"})
    return da.astype(float) if da.dtype.kind == "i" else da


def _plain(da: xr.DataArray) -> xr.DataArray:
    if getattr(da.pint, "units", None) is not None:
        da = da.pint.dequantify()
    return da


def _netcdf3(da: xr.DataArray) -> xr.DataArray:
    """NetCDF3 (the scipy writer) has no 64-bit integers."""
    da = da.astype(np.float64) if da.dtype.kind in "iub" else da
    for name, coord in da.coords.items():
        if coord.dtype.kind in "iu":
            da = da.assign_coords({name: coord.astype(np.int32)})
    return da


def _to_json(value) -> str:
    return json.dumps(value, default=lambda o: o.tolist() if isinstance(o, np.ndarray)
                      else o.item() if isinstance(o, np.generic) else str(o))
