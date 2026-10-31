import logging


class _BelowWarning(logging.Filter):
    """
    Drops INFO and DEBUG records logged directly to the root logger. pysnirf2 writes
    one INFO line per step of reading each .snirf file to the root logger, and sets
    the root level back to INFO every time it opens a file, so a level alone cannot
    silence it; a filter stays in place.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING


class _DropMessage(logging.Filter):
    """Drops records whose message starts with `prefix`."""
    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(self.prefix)


def setup_logging():
    import mne
    mne.set_log_level("WARNING")    # kills "FIR filter parameters", "Filtering raw...", joblib lines

    # silence library loggers that log to root or to named loggers
    for name in ("pyprep", "pyprep.reference", "snirf", "snirf.pysnirf2", "cedalion", "h5py"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)   # root → silences "Loading from file ...", etc.
    root = logging.getLogger()
    if not any(isinstance(f, _BelowWarning) for f in root.filters):
        root.addFilter(_BelowWarning())             # pysnirf2 resets the root level per file
    # The synthetic SNIRF files list landmark names without positions; cedalion warns
    # about it once per file, and drops the names, which nothing here uses.
    cedalion_log = logging.getLogger("cedalion")
    if not any(isinstance(f, _DropMessage) for f in cedalion_log.filters):
        cedalion_log.addFilter(_DropMessage("landmark labels were provided but not their positions"))

    logger = logging.getLogger("eeg_batch")
    logger.setLevel(logging.INFO)
    logger.propagate = False                         # don't double-emit through root
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.handlers.clear()                          # avoid duplicate handlers on re-run
    logger.addHandler(ch)
    return logger
