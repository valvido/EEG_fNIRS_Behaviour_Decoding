"""
Results: a run's output as data, written once and read back for any plot or
analysis, without rerunning the pipeline.

Example
-------
    run_dir = save_run(result)        # results/runs/<experiment>/<dataset>/<started>/
    result = load_run(run_dir)        # the same RunResult, models ready to apply
    inputs = load_inputs(run_dir)     # the preprocessed runs, if the run saved them
"""
from src.results.inputs import PREPROCESSED_DIR
from src.results.load import load_inputs, load_run
from src.results.save import RESULTS_DIR, save_run

__all__ = ["save_run", "load_run", "load_inputs", "RESULTS_DIR", "PREPROCESSED_DIR"]
