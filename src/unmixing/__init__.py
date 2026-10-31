"""
Unmixing: select, fit and score the CCA-family models on the preprocessed runs.
"""
from src.unmixing.assignment import best_assignment, match_weights
from src.unmixing.folds import split
from src.unmixing.metrics import METRICS
from src.unmixing.models import check_config
from src.unmixing.report import selection_report
from src.unmixing.run import find_hands, unmix
from src.unmixing.scores import ground_truth_mixing, unmix_run
from src.unmixing.state import fitted_from_dataset, fitted_to_dataset

__all__ = ["check_config", "selection_report", "find_hands", "unmix", "unmix_run", "split",
           "ground_truth_mixing", "best_assignment", "match_weights", "METRICS",
           "fitted_to_dataset", "fitted_from_dataset"]
