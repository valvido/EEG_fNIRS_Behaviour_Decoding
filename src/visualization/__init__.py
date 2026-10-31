"""
Visualization: figures of a run, drawn only from what the run saved.

Every plot function takes a RunResult (from `load_run`), one of its selections, or
the runs from `load_inputs`, and returns a matplotlib figure; nothing here computes
results. `save_figures` draws all of them into a run's figures/ folder.

Example
-------
    result = load_run(run_dir)
    fig = plot_source_recovery(result, "joint_avg")
    selection = result.unmixing.selections[0]
    fig = plot_time_courses(selection)                              # test group
    fig = plot_time_courses(selection, selection.folds[0].val_group)  # one fold
    fig = plot_inputs(load_inputs(run_dir))
    save_figures(result, run_dir / "figures")
"""
from src.visualization.assignment import plot_pair_assignment
from src.visualization.classification import plot_classification
from src.visualization.inputs import plot_inputs
from src.visualization.params import plot_selection_landscape, selected_params
from src.visualization.save import save_figures
from src.visualization.scores import (plot_isi, plot_nested_spread, plot_pattern_recovery,
                                      plot_source_recovery)
from src.visualization.spatial import plot_bad_channels, plot_patterns
from src.visualization.timecourses import plot_time_courses

__all__ = ["plot_source_recovery", "plot_pattern_recovery", "plot_isi", "plot_nested_spread", "plot_time_courses",
           "plot_patterns", "plot_bad_channels", "plot_pair_assignment", "plot_selection_landscape",
           "selected_params", "plot_inputs", "plot_classification", "save_figures"]
