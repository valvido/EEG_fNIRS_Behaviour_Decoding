"""
Entry point: run one experiment on one subject.

Usage
-----
    python main.py --config configs/experiments/default.yaml \
                   --subject configs/subjects/synthetic/sub01/sub01_snr_fnirs-5_eeg5.yaml

    Ex. python main.py --config configs/experiments/tutorial.yaml \
               --subject configs/subjects/synthetic/sub01/sub01_snr_fnirs-5_eeg5.yaml

The experiment config says what to run; the subject config says which data to run it
on. Everything else happens in src/experiment.py, which also writes the result to
results/runs/<experiment>/<dataset>/<started>/: the data to results/ (read back with
load_run) and the figures to figures/.
"""
import argparse
import sys
import warnings

from src.config import load_experiment, load_subject
from src.experiment import run
from src.log import setup_logging

# Accelerate-backed NumPy on macOS raises spurious floating point warnings in matmul.
warnings.filterwarnings("ignore", message=".*encountered in matmul",
                        category=RuntimeWarning)
# loky restarts a worker whose memory grew by 300 MB since its first task; the
# temporal fits swing that much without leaking, and no result is lost.
warnings.filterwarnings("ignore", message="A worker stopped while some jobs",
                        category=UserWarning)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one experiment on one subject.")
    parser.add_argument("--config", required=True,
                        help="Main experiment YAML (configs/experiments/*.yaml)")
    parser.add_argument("--subject", required=True,
                        help="Subject YAML (configs/subjects/<format>/<subject_id>/*.yaml)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = setup_logging()

    cfg = load_experiment(args.config)
    subject = load_subject(args.subject)

    run(cfg, subject, logger)
    return 0


if __name__ == "__main__":
    sys.exit(main())
