"""
Unmixing results as log lines.
"""
from src.types import Selection


def selection_report(selection: Selection) -> str:
    """
    One block per selected model. The first line is what selection saw on the dev
    groups: the metric, its validation mean over the folds, and the settings it chose.
    Below it, the model refit on all dev groups and scored on the test group, one
    line per source, then the ISI.

    Per source: |r| of the pair's EEG and fNIRS component with the source, the
    pair's signed canonical r, and as diagnostics the pattern |r| with the
    ground-truth mixing and the leakage of the source into the other pair. Without
    ground truth (no `best_pairs`), only the pair and its canonical r per hand.
    """
    s = selection.scores
    width = max(len(src) for src in s.labels.pairs)
    lines = [f"{selection.method} by {selection.metric}: validation (dev) "
             f"{selection.inner_score:.3f} with {selection.params}"]
    if not s.best_pairs:
        lines += [f"    test {selection.test_group}  {src:<{width}} pair {pair:>2}  "
                  f"canonical r {s.canonical_r[src]:+.2f}"
                  for src, pair in s.labels.pairs.items()]
        return "\n".join(lines)
    for src, pair in s.labels.pairs.items():
        lines.append(f"    test {selection.test_group}  {src:<{width}} pair {pair:>2}  "
                     f"r eeg {s.r_eeg[src]:.2f} fnirs {s.r_fnirs[src]:.2f}  "
                     f"canonical r {s.canonical_r[src]:+.2f}  "
                     f"pattern r eeg {s.pattern_r_eeg[src]:.2f} fnirs {s.pattern_r_fnirs[src]:.2f}  "
                     f"leak eeg {s.leak_eeg[src]:.2f} fnirs {s.leak_fnirs[src]:.2f}  "
                     f"(best pair {s.best_pairs[src][0]})")
    lines.append(f"    test {selection.test_group}  ISI eeg {s.isi_eeg:.2f} fnirs {s.isi_fnirs:.2f}")
    return "\n".join(lines)
