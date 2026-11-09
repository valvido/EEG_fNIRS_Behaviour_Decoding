"""
Detect bad EEG channels with pyprep, in two passes separated by an average re-reference.
"""
import contextlib
import io

import mne
from pyprep import NoisyChannels

from src.config import EegPruningConfig


def detect_bad_eeg_channels_by_recording(
        eegs: dict[str, mne.io.Raw], cfg: EegPruningConfig, random_state: int,
) -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, tuple[str, ...]]]]:
    """
    Detect bad EEG channels on every recording on its own, each judged whole.

    Parameters
    ----------
    eegs : dict[str, mne.io.Raw]
        The EEG of each recording, by recording name.
    cfg : EegPruningConfig
    random_state : int
        Seed for pyprep.

    Returns
    -------
    by_recording : dict[str, tuple[str, ...]]
        Bad channels of each recording, sorted by name.
    detail : dict[str, dict[str, tuple[str, ...]]]
        Channels each criterion flagged in any recording: {"pass1": {...}, "pass2": {...}}.
    """
    by_recording, detail = {}, {}
    for name, eeg in eegs.items():
        by_recording[name], recording_detail = detect_bad_eeg_channels(eeg, cfg, random_state)
        for pass_name, criteria in recording_detail.items():
            merged = detail.setdefault(pass_name, {})
            for criterion, channels in criteria.items():
                merged[criterion] = tuple(sorted({*merged.get(criterion, ()), *channels}))
    return by_recording, detail


def detect_bad_eeg_channels(
        eeg: mne.io.Raw, cfg: EegPruningConfig,
        random_state: int) -> tuple[tuple[str, ...], dict[str, dict[str, tuple[str, ...]]]]:
    """
    Detect bad EEG channels in two passes, separated by an average re-reference.

    Pass 1 runs only the criteria that judge a channel on its own: NaN, flat,
    amplitude deviation, high-frequency noise. Those hold whatever the recording
    reference is. Pass 2 runs the criteria that compare channels against each other:
    correlation, dropout, SNR, RANSAC. They assume the shared reference signal has
    been removed, so they run only after re-referencing to the average of the
    channels pass 1 accepted. A channel flagged in either pass is bad.

    Parameters
    ----------
    eeg : mne.io.Raw
        The EEG to judge. Only EEG-type channels are judged.
    cfg : EegPruningConfig
        Line frequencies, whether to run RANSAC, and the pass 1 / pass 2 thresholds.
    random_state : int
        Seed for pyprep.

    Returns
    -------
    bad : tuple[str, ...]
        Bad channels, sorted by name.
    detail : dict[str, dict[str, tuple[str, ...]]]
        Channels flagged by each criterion: {"pass1": {...}, "pass2": {...}}.
    """
    raw = eeg.copy().pick(picks="eeg")
    # The hf_noise criterion measures power above 50 Hz, where the line peaks sit.
    raw.notch_filter(list(cfg.line_freqs), verbose="ERROR")

    # 1. Reference-robust pass. NaN and flat are detected when NoisyChannels is
    #    constructed; the explicit re-call applies the configured flat_threshold.
    with contextlib.redirect_stdout(io.StringIO()):
        pass1 = NoisyChannels(raw, random_state=random_state)
        pass1.find_bad_by_nan_flat(flat_threshold=cfg.pass1.flat_threshold)
        pass1.find_bad_by_deviation(deviation_threshold=cfg.pass1.deviation_threshold)
        pass1.find_bad_by_hfnoise(HF_zscore_threshold=cfg.pass1.hf_noise_threshold)
    bads_pass1 = {
        "bad_by_nan": tuple(pass1.bad_by_nan),
        "bad_by_flat": tuple(pass1.bad_by_flat),
        "bad_by_deviation": tuple(pass1.bad_by_deviation),
        "bad_by_hf_noise": tuple(pass1.bad_by_hf_noise),
    }
    flagged_pass1 = set().union(*bads_pass1.values())

    # 2. Average-reference over the pass-1-good channels only. MNE leaves channels
    #    listed in info["bads"] out of the average it computes, so marking them is
    #    enough; a channel with a broken amplitude would otherwise leak into every
    #    other channel and depress the correlations pass 2 measures.
    referenced = raw.copy()
    referenced.info["bads"] = sorted(flagged_pass1)
    referenced.set_eeg_reference("average", verbose="ERROR")

    # 3. Inter-channel pass on the re-referenced data. The criteria are called
    #    individually rather than through find_all_bads, which would also re-run the
    #    pass-1 criteria on data they were not meant for, and add bad_by_psd.
    #    find_bad_by_SNR needs hf_noise and computes it here as a side effect; that
    #    referenced verdict feeds the SNR intersection only, never the final list.
    referenced.info["bads"] = []
    with contextlib.redirect_stdout(io.StringIO()):
        pass2 = NoisyChannels(referenced, random_state=random_state)
        pass2.find_bad_by_correlation(
            correlation_secs=cfg.pass2.correlation_secs,
            correlation_threshold=cfg.pass2.correlation_threshold,
            frac_bad=cfg.pass2.correlation_frac_bad,
        )
        pass2.find_bad_by_SNR()
        if cfg.ransac:
            pass2.find_bad_by_ransac(
                n_samples=cfg.pass2.ransac_n_samples,
                sample_prop=cfg.pass2.ransac_sample_prop,
                corr_thresh=cfg.pass2.ransac_corr_thresh,
                frac_bad=cfg.pass2.ransac_frac_bad,
                corr_window_secs=cfg.pass2.ransac_window_secs,
            )
    bads_pass2 = {
        "bad_by_correlation": tuple(pass2.bad_by_correlation),
        "bad_by_dropout": tuple(pass2.bad_by_dropout),
        "bad_by_SNR": tuple(pass2.bad_by_SNR),
        # RANSAC returns numpy str_; plain str keeps the lists printable and saveable.
        "bad_by_ransac": tuple(str(ch) for ch in pass2.bad_by_ransac),
    }

    bad = tuple(sorted(flagged_pass1.union(*bads_pass2.values())))
    return bad, {"pass1": bads_pass1, "pass2": bads_pass2}
