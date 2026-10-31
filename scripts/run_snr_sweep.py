"""
Run one experiment on every SNR dataset of the synthetic subjects.

Usage
-----
    python scripts/run_snr_sweep.py --config configs/experiments/default.yaml

    Ex. python scripts/run_snr_sweep.py --config configs/experiments/tutorial.yaml \
                                        --subjects sub01

A subject's SNR datasets are its files configs/subjects/synthetic/<subject_id>/*_snr_*.yaml,
run from the lowest fNIRS SNR to the highest. Each run is one call of main.py, so it
writes to results/runs/<experiment>/<dataset>/<started>/ like any other run. A run that
fails is logged and the sweep goes on with the next; the summary at the end lists the
failed runs, and the exit code is 1 if there are any.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import main  
from src.config import load_subject   
from src.log import setup_logging   

SUBJECTS_DIR = PROJECT_ROOT / "configs" / "subjects" / "synthetic"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one experiment on every SNR dataset of the synthetic subjects.")
    parser.add_argument("--config", required=True,
                        help="Main experiment YAML (configs/experiments/*.yaml)")
    parser.add_argument("--subjects", nargs="+",
                        default=sorted(p.name for p in SUBJECTS_DIR.iterdir() if p.is_dir()),
                        help="Subject IDs (default: every subject in configs/subjects/synthetic)")
    return parser.parse_args(argv)


def snr_datasets(subject_id: str) -> list[Path]:
    """
    The subject files of one subject's SNR datasets, lowest fNIRS SNR first.

    Parameters
    ----------
    subject_id : str
        A folder in configs/subjects/synthetic, e.g. "sub01".

    Returns
    -------
    list[Path]
        The subject YAMLs, ordered by the fNIRS SNR in each dataset's simulation config.
    """
    paths = list((SUBJECTS_DIR / subject_id).glob("*_snr_*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No *_snr_*.yaml subject files in {SUBJECTS_DIR / subject_id}")
    return sorted(paths, key=lambda path: load_subject(path).simulation["snr"]["fnirs_db"])


def sweep(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = setup_logging()

    datasets = [path for subject_id in args.subjects for path in snr_datasets(subject_id)]
    logger.info(f"SNR sweep of '{args.config}' over {len(datasets)} datasets "
                f"of subjects {args.subjects}")

    failed = []
    for i, path in enumerate(datasets, start=1):
        logger.info(f"=== [{i}/{len(datasets)}] {path.stem} ===")
        try:
            main.main(["--config", args.config, "--subject", str(path)])
        except Exception:
            logger.exception(f"{path.stem} failed; going on with the next dataset")
            failed.append(path.stem)

    logger.info(f"SNR sweep done: {len(datasets) - len(failed)}/{len(datasets)} runs succeeded")
    if failed:
        logger.error(f"Failed runs: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(sweep())
