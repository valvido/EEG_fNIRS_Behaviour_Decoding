"""
Correlation of components with reference time courses, over the time points where
both are finite.
"""
import numpy as np
import xarray as xr

# Fewer finite samples than this in common: no correlation (NaN), and for a labelled
# pair a stop (TooFewSamples).
MIN_SAMPLES = 11


def correlations(comps: xr.DataArray, sources: xr.DataArray) -> xr.DataArray:
    """
    Signed correlation of every component with every source, over the time points
    both have where both are finite. The sign is the model's own orientation and
    carries no meaning; the scores take |r|.

    Parameters
    ----------
    comps : xr.DataArray
        Dims (time, component).
    sources : xr.DataArray
        Dims (time, source).

    Returns
    -------
    xr.DataArray
        Dims (component, source). NaN where fewer than MIN_SAMPLES samples are shared.
    """
    comps, sources = xr.align(comps.transpose("time", "component"),
                              sources.transpose("time", "source"))
    out = np.full((comps.sizes["component"], sources.sizes["source"]), np.nan)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = correlate(comps.values[:, i], sources.values[:, j])
    return xr.DataArray(out, dims=("component", "source"),
                        coords={"component": comps.component.values,
                                "source": sources.source.values})


def correlate(a: np.ndarray, b: np.ndarray, min_samples: int = MIN_SAMPLES) -> float:
    """Correlation of a and b where both are finite; NaN with fewer than `min_samples`."""
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < min_samples:
        return np.nan
    return float(np.corrcoef(a[keep], b[keep])[0, 1])
