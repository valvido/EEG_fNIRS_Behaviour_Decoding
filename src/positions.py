"""
EEG electrode positions as a cedalion labeled point cloud, so they travel with the
data the same way the optode positions (geo3d) do, and back to an MNE montage where
MNE needs one (topomaps); and the optodes each fNIRS channel is made of. Nothing here
chooses a montage; the subject YAML does.
"""
import cedalion.dataclasses as cdc
import mne
import numpy as np
import xarray as xr

# MNE's names of the head landmarks, mapped to the names cedalion gives them.
_LANDMARKS = {"nasion": "Nz", "lpa": "LPA", "rpa": "RPA"}


def electrode_positions(montage: mne.channels.DigMontage, channels) -> xr.DataArray:
    """
    The positions of `channels` and the montage's head landmarks.

    Parameters
    ----------
    montage : mne.channels.DigMontage
        E.g. `raw.get_montage()`, which is in MNE's head frame.
    channels : sequence of str
        The electrodes to keep, in this order.

    Returns
    -------
    xr.DataArray
        Dims (label, pos), in mm, typed ELECTRODE and LANDMARK, in the montage's frame.

    Raises
    ------
    ValueError
        If a channel has no position in the montage.
    """
    positions = montage.get_positions()
    ch_pos = positions["ch_pos"]
    channels = [str(ch) for ch in channels]
    missing = [ch for ch in channels if ch not in ch_pos or not np.isfinite(ch_pos[ch]).all()]
    if missing:
        raise ValueError(f"EEG channels {missing} have no position in the montage.")
    landmarks = {name: positions[key] for key, name in _LANDMARKS.items()
                 if positions[key] is not None}
    labels = [*channels, *landmarks]
    types = [cdc.PointType.ELECTRODE] * len(channels) + [cdc.PointType.LANDMARK] * len(landmarks)
    coords_m = np.array([*(ch_pos[ch] for ch in channels), *landmarks.values()], dtype=float)
    return cdc.build_labeled_points(coords_m * 1000.0, units="mm", labels=labels, types=types)


def electrode_montage(points: xr.DataArray) -> mne.channels.DigMontage:
    """
    The MNE montage of electrode positions made by `electrode_positions`, in MNE's
    head frame.

    Parameters
    ----------
    points : xr.DataArray
        Dims (label, pos), typed ELECTRODE and LANDMARK, with length units.

    Returns
    -------
    mne.channels.DigMontage
    """
    points_m = points.pint.to("m").pint.dequantify()
    types = list(points_m.type.values)
    ch_pos = {str(label): points_m.sel(label=label).values
              for label, kind in zip(points_m.label.values, types)
              if kind == cdc.PointType.ELECTRODE}
    landmarks = {key: points_m.sel(label=name).values for key, name in _LANDMARKS.items()
                 if name in points_m.label.values}
    return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head", **landmarks)


def channel_optodes(fnirs: xr.DataArray) -> dict[str, tuple[str, str]]:
    """
    The source and detector of every channel of an fNIRS array, from its `source`
    and `detector` coordinates on the channel axis.

    Parameters
    ----------
    fnirs : xr.DataArray
        Any fNIRS array with a channel axis carrying source and detector, e.g. the
        amplitude as loaded.

    Returns
    -------
    dict[str, tuple[str, str]]
        {channel: (source, detector)}, in channel order.
    """
    return {str(ch): (str(src), str(det)) for ch, src, det in
            zip(fnirs.channel.values, fnirs.source.values, fnirs.detector.values)}
