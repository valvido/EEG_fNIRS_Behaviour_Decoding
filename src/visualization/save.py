"""
Draw every standard figure of a run into one folder.
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from src.types import RunResult
from src.visualization.assignment import plot_pair_assignment
from src.visualization.classification import plot_classification
from src.visualization.inputs import plot_inputs
from src.visualization.params import plot_selection_landscape, selected_params
from src.visualization.scores import (plot_isi, plot_nested_spread, plot_pattern_recovery,
                                      plot_source_recovery)
from src.visualization.spatial import plot_bad_channels, plot_patterns
from src.visualization.timecourses import plot_time_courses


def save_figures(result: RunResult, folder: str | Path,
                 logger: logging.Logger | None = None) -> Path:
    """
    Write the figures of a run, as PNG, to `folder`:

    - bad_channels.png
    - selected_params.csv
    - source_recovery_<metric>.png, pattern_recovery_<metric>.png, isi_<metric>.png
      (with ground truth)
    - landscape_<method>_<metric>.png (for methods with tuned hyperparameters)
    - nested_recovery_<metric>.png, nested_pattern_<metric>.png (with nested evaluation
      and ground truth)
    - classification_<metric>.png (with classification)
    - inputs_test.png, inputs_train.png (with `result.inputs`)
    - <metric>/<method>/test.png, patterns.png, assignment.png (with ground truth), and
      per validation fold fold_<val_group>.png, fold_<val_group>_patterns.png

    With the "motor_channels" assignment, the patterns circle each hand's motor
    channels.

    A figure that fails to draw is skipped with a warning, so one problem never
    loses the others.

    Parameters
    ----------
    result : RunResult
    folder : str | Path
        Created if missing.
    logger : logging.Logger | None

    Returns
    -------
    Path
        `folder`.
    """
    logger = logger or logging.getLogger(__name__)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    unmixing = result.unmixing
    metrics = list(result.cfg.unmixing.metrics)
    test_group = unmixing.test_group
    assignment = result.cfg.unmixing.assignment
    with_truth = all(s.recovery.s_eeg is not None for s in unmixing.selections)
    motor_channels = assignment.motor_channels if assignment.method == "motor_channels" else None

    def save(name: str, draw):
        try:
            fig = draw()
        except Exception as e:
            logger.warning(f"Figure {name} not drawn: {type(e).__name__}: {e}")
            return
        if fig is None:
            return
        path = folder / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    save("bad_channels", lambda: plot_bad_channels(result))
    selected_params(result).to_csv(folder / "selected_params.csv")
    for metric in metrics:
        if with_truth:
            save(f"source_recovery_{metric}", lambda: plot_source_recovery(result, metric))
            save(f"pattern_recovery_{metric}", lambda: plot_pattern_recovery(result, metric))
            save(f"isi_{metric}", lambda: plot_isi(result, metric))
        for method in result.cfg.unmixing.methods:
            save(f"landscape_{method}_{metric}",
                 lambda: plot_selection_landscape(result, method, metric))
        if unmixing.nested and with_truth:
            save(f"nested_recovery_{metric}", lambda: plot_nested_spread(result, metric, "recovery"))
            save(f"nested_pattern_{metric}", lambda: plot_nested_spread(result, metric, "pattern"))
        if result.classification is not None:
            save(f"classification_{metric}", lambda: plot_classification(result, metric))
    if result.inputs is not None:
        train = sorted({r.group for r in result.inputs} - {test_group})
        save("inputs_test", lambda: plot_inputs(result.inputs, groups=[test_group]))
        save("inputs_train", lambda: plot_inputs(result.inputs, groups=train))
    for selection in unmixing.selections:
        where = f"{selection.metric}/{selection.method}"
        save(f"{where}/test", lambda: plot_time_courses(selection))
        save(f"{where}/patterns",
             lambda: plot_patterns(selection, result.geometry, motor_channels=motor_channels))
        if with_truth:
            save(f"{where}/assignment", lambda: plot_pair_assignment(selection, assignment))
        for fold in selection.folds:
            save(f"{where}/fold_{fold.val_group}",
                 lambda: plot_time_courses(selection, fold.val_group))
            save(f"{where}/fold_{fold.val_group}_patterns",
                 lambda: plot_patterns(selection, result.geometry, fold.val_group,
                                       motor_channels))
    return folder
