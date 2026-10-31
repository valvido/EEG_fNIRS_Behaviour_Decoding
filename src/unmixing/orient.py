"""
Spatial filters and Haufe patterns of the fitted models.

A fitted model gives each component as s or -s with the same fit. Unmixing keeps the
sign the model gives; the scores do not depend on it, and only the plots orient the
components (src/visualization/orient.py).
"""
import numpy as np
import xarray as xr


def spatial_filter(model, modality: str) -> xr.DataArray:
    """
    Fitted filter W as (channel, component), divided by the training channel std so it
    acts on the data as given. Temporal models take each component's slice at its
    optimal shift.

    Parameters
    ----------
    model
        A fitted cedalion model.
    modality : str
        "eeg" (X) or "fnirs" (Y).
    """
    W, latent, std = ((model.Wx, model.latent_featureX_name, model.x_std) if modality == "eeg"
                      else (model.Wy, model.latent_featureY_name, model.y_std))
    if getattr(W.pint, "units", None) is not None:
        W = W.pint.dequantify()
    if std is not None:
        W = W / std
    if "time_shift" in W.dims:
        W = xr.concat([W.sel(time_shift=shift).isel({latent: k})
                       for k, shift in enumerate(model.optimal_shift)], dim=latent)
    W = W.transpose("channel", latent).rename({latent: "component"})
    W = W.drop_vars([c for c in W.coords if c != "channel"])
    return W.assign_coords(component=np.arange(W.sizes["component"]))


def haufe_pattern(W: xr.DataArray, data: xr.DataArray) -> xr.DataArray:
    """
    Activation patterns a_k = Sigma_x w_k / (w_k' Sigma_x w_k) (Haufe et al. 2014,
    Eq. 6, with the source covariance taken diagonal), in the data's units.

    Parameters
    ----------
    W : xr.DataArray
        Filters, dims (channel, component), from `spatial_filter`.
    data : xr.DataArray
        The data the filters are applied to, dims (time, channel).

    Returns
    -------
    xr.DataArray
        Dims (channel, component). A column is NaN when its filter has no variance on
        the data.
    """
    X = data.transpose("time", "channel").sel(channel=W.channel).values
    X = X[np.isfinite(X).all(axis=1)]
    # numpy linked to Apple Accelerate raises floating point flags in matmul on
    # finite inputs whose result is correct, so they are ignored here; a filter with
    # no variance still gives a NaN column below.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        cov_w = np.cov(X, rowvar=False) @ W.values
        var = np.einsum("ck,ck->k", W.values, cov_w)
        A = np.where(var > 0, cov_w / var, np.nan)
    return W.copy(data=A)

