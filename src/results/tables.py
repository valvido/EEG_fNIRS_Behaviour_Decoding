"""
Scores as flat table rows and back.
"""
import json
import math

import numpy as np

from src.types import ClassificationScore, PairLabels, Scores


def classification_row(evaluation: str, score: ClassificationScore) -> dict:
    """
    One flat row: evaluation, test_group, method, metric, feature_set, accuracy,
    recall_<class>, n_train, n_test, dropped_train, dropped_test, and per pair of
    classes confusion_<true>_as_<predicted>.
    """
    row = {"evaluation": evaluation, "test_group": score.test_group, "method": score.method,
           "metric": score.metric, "feature_set": score.feature_set,
           "accuracy": score.accuracy}
    row.update({f"recall_{c}": r for c, r in score.recall.items()})
    row.update({"n_train": score.n_train, "n_test": score.n_test,
                "dropped_train": score.dropped_train, "dropped_test": score.dropped_test})
    for true, predicted in score.confusion.items():
        row.update({f"confusion_{true}_as_{p}": n for p, n in predicted.items()})
    return row


def classification_from_row(row: dict, classes: tuple[str, ...]) -> ClassificationScore:
    """The ClassificationScore `classification_row` was given."""
    return ClassificationScore(
        method=row["method"], metric=row["metric"], test_group=row["test_group"],
        feature_set=row["feature_set"], classes=classes,
        confusion={t: {p: int(row[f"confusion_{t}_as_{p}"]) for p in classes} for t in classes},
        n_train=int(row["n_train"]), dropped_train=int(row["dropped_train"]),
        dropped_test=int(row["dropped_test"]))


def score_columns(scores: Scores | None) -> dict:
    """
    One flat row: label_score, isi_eeg, isi_fnirs, and per source pair_<s>, r_eeg_<s>,
    r_fnirs_<s>, canonical_r_<s>, pattern_r_eeg_<s>, pattern_r_fnirs_<s>, leak_eeg_<s>,
    leak_fnirs_<s>, best_pair_<s>, best_r_eeg_<s>, best_r_fnirs_<s> (the best_* only with
    ground truth). Empty for None (a failed fit).
    """
    if scores is None:
        return {}
    row = {"label_score": scores.labels.score, "isi_eeg": scores.isi_eeg,
           "isi_fnirs": scores.isi_fnirs}
    for source, pair in scores.labels.pairs.items():
        row[f"pair_{source}"] = pair
        row[f"r_eeg_{source}"] = scores.r_eeg[source]
        row[f"r_fnirs_{source}"] = scores.r_fnirs[source]
        row[f"canonical_r_{source}"] = scores.canonical_r[source]
        row[f"pattern_r_eeg_{source}"] = scores.pattern_r_eeg[source]
        row[f"pattern_r_fnirs_{source}"] = scores.pattern_r_fnirs[source]
        row[f"leak_eeg_{source}"] = scores.leak_eeg[source]
        row[f"leak_fnirs_{source}"] = scores.leak_fnirs[source]
    for source, (pair, r_eeg, r_fnirs) in scores.best_pairs.items():
        row[f"best_pair_{source}"] = pair
        row[f"best_r_eeg_{source}"] = r_eeg
        row[f"best_r_fnirs_{source}"] = r_fnirs
    return row


def scores_from_row(row: dict) -> Scores | None:
    """The Scores `score_columns` was given; None for a row without scores."""
    sources = [key.removeprefix("pair_") for key in row if key.startswith("pair_")]
    if not sources or _missing(row[f"pair_{sources[0]}"]):
        return None
    return Scores(
        labels=PairLabels(pairs={s: int(row[f"pair_{s}"]) for s in sources},
                          score=float(row["label_score"])),
        r_eeg={s: float(row[f"r_eeg_{s}"]) for s in sources},
        r_fnirs={s: float(row[f"r_fnirs_{s}"]) for s in sources},
        isi_eeg=float(row["isi_eeg"]), isi_fnirs=float(row["isi_fnirs"]),
        canonical_r={s: float(row[f"canonical_r_{s}"]) for s in sources},
        best_pairs={s: (int(row[f"best_pair_{s}"]), float(row[f"best_r_eeg_{s}"]),
                        float(row[f"best_r_fnirs_{s}"]))
                    for s in sources if not _missing(row.get(f"best_pair_{s}"))},
        pattern_r_eeg={s: float(row[f"pattern_r_eeg_{s}"]) for s in sources},
        pattern_r_fnirs={s: float(row[f"pattern_r_fnirs_{s}"]) for s in sources},
        leak_eeg={s: float(row[f"leak_eeg_{s}"]) for s in sources},
        leak_fnirs={s: float(row[f"leak_fnirs_{s}"]) for s in sources},
    )


def params_to_json(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=lambda o: o.tolist()
                      if isinstance(o, np.ndarray) else o.item())


def params_from_json(text: str) -> dict:
    params = json.loads(text)
    if "time_shifts" in params:
        params["time_shifts"] = np.asarray(params["time_shifts"], dtype=float)
    return params


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))
