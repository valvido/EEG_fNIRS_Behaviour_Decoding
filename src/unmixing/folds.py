"""
Split a subject's runs by group (background recording) for evaluation.
"""
from sklearn.model_selection import LeaveOneGroupOut

from src.types import PreprocessedRecording


def groups_of(recordings: list[PreprocessedRecording]) -> list[str]:
    """The groups present, in the order they first appear."""
    return list(dict.fromkeys(rec.group for rec in recordings))


def split(recordings: list[PreprocessedRecording],
          test_group: str) -> tuple[list[PreprocessedRecording], list[PreprocessedRecording]]:
    """
    The runs of `test_group` held out from all other runs.

    Parameters
    ----------
    recordings : list[PreprocessedRecording]
    test_group : str

    Returns
    -------
    dev, test : list[PreprocessedRecording]
        Each in the order given.

    Raises
    ------
    ValueError
        If `test_group` has no runs, or no other group does.
    """
    test = [rec for rec in recordings if rec.group == test_group]
    dev = [rec for rec in recordings if rec.group != test_group]
    if not test:
        raise ValueError(f"No runs of test group {test_group!r}; groups present: "
                         f"{groups_of(recordings)}.")
    if not dev:
        raise ValueError(f"Only test group {test_group!r} has runs; nothing to train on.")
    return dev, test


def leave_one_group_out(recordings: list[PreprocessedRecording]
                        ) -> list[tuple[list[PreprocessedRecording], list[PreprocessedRecording]]]:
    """
    One (train, validation) fold per group, from sklearn's LeaveOneGroupOut: the
    group's runs are the validation set, all other runs the training set. The folds
    come in sorted group order; within a fold the runs keep the order given.

    Raises
    ------
    ValueError
        With fewer than two groups.
    """
    groups = groups_of(recordings)
    if len(groups) < 2:
        raise ValueError(f"Leave-one-group-out needs at least two groups, got {groups}.")
    labels = [rec.group for rec in recordings]
    return [([recordings[i] for i in train], [recordings[i] for i in val])
            for train, val in LeaveOneGroupOut().split(recordings, groups=labels)]
