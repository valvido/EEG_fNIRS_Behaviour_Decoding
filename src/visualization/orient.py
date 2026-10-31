"""
The sign the plots give a component, so the colours of a pattern can be read.

A fitted model gives each component as s or -s with the same fit, and unmixing keeps
whichever sign the model gave. For display only, each component is flipped so its
pattern is positive at the channel where it is largest in absolute value. The time
course and the pattern are flipped together (the filter would be flipped with them),
so the plotted solution is the model's. Scores and saved results are never flipped.
"""
import numpy as np
import xarray as xr

from src.types import Recovery


def canonical_sign(A: xr.DataArray) -> xr.DataArray:
    """
    +1 or -1 per component, making each pattern positive at its largest-|value|
    channel. An all-NaN pattern keeps +1.

    Parameters
    ----------
    A : xr.DataArray
        Patterns, dims (channel, component).

    Returns
    -------
    xr.DataArray
        Dims (component,).
    """
    values = A.transpose("channel", "component").values
    signs = np.ones(values.shape[1])
    for k in range(values.shape[1]):
        col = values[:, k]
        if np.isfinite(col).any() and col[np.nanargmax(np.abs(col))] < 0:
            signs[k] = -1.0
    return xr.DataArray(signs, dims="component", coords={"component": A.component.values})


def oriented(recovery: Recovery, method: str, modality: str,
             pair: int) -> tuple[xr.DataArray, xr.DataArray]:
    """
    One pair's component and pattern on the held-out runs, flipped together for
    display (`canonical_sign` of the pattern).

    mSPoC's EEG component is the band power of the filtered EEG, which does not
    change sign when the filter is flipped, so it is never flipped; its pattern is.

    Parameters
    ----------
    recovery : Recovery
    method : str
        The method that was fitted.
    modality : str
        "eeg" or "fnirs".
    pair : int
        Component index.

    Returns
    -------
    component : xr.DataArray
        Dims (time,).
    pattern : xr.DataArray
        Dims (channel,).
    """
    comps, patterns = ((recovery.u, recovery.pattern_eeg) if modality == "eeg"
                       else (recovery.v, recovery.pattern_fnirs))
    sign = float(canonical_sign(patterns).sel(component=pair))
    component = comps.sel(component=pair)
    if not (method == "mSPoC" and modality == "eeg"):
        component = component * sign
    return component, patterns.sel(component=pair) * sign
