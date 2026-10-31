"""
Classification results as log lines.
"""
from src.types import ClassificationScore


def classification_report(scores: list[ClassificationScore]) -> str:
    """
    One block per selected model: a line with the trial counts, then one line per
    feature set with the test accuracy and the recall of every class.

    Parameters
    ----------
    scores : list[ClassificationScore]
        The feature sets of one selection.
    """
    first = scores[0]
    width = max(len(s.feature_set) for s in scores)
    lines = [f"{first.method} by {first.metric}: classification of test group "
             f"{first.test_group}, {first.n_train} training and {first.n_test} test trials "
             f"(dropped {first.dropped_train} and {first.dropped_test})"]
    for s in scores:
        recall = "  ".join(f"{c} {r:.2f}" for c, r in s.recall.items())
        lines.append(f"    {s.feature_set:<{width}}  accuracy {s.accuracy:.2f}  recall {recall}")
    return "\n".join(lines)
