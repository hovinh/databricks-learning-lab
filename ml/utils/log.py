import logging
import sys
from datetime import datetime
from pathlib import Path

import runtime

ML_ROOT = Path(__file__).resolve().parent.parent
LOG_FORMAT = "[%(levelname)s %(asctime)s %(module)s:%(lineno)d] %(message)s"


def setup_logger(process_name: str = "main", logdir: str = "logs") -> None:
    """Call once, from a pipeline's __main__ block. Elsewhere use
    logging.getLogger(__name__): child loggers propagate to these handlers.

    On Databricks: stdout only (captured in the run output; local files there
    are ephemeral). Locally: stdout + logs/<process_name>/<timestamp>.log."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # The SDK logs its auth flow at INFO on every call: noise, not signal.
    logging.getLogger("databricks.sdk").setLevel(logging.WARNING)
    # Connect warns "Effective usage policy for this session..." on every call.
    logging.getLogger("pyspark.sql.connect.logging").setLevel(logging.ERROR)
    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    if not any(type(h) is logging.StreamHandler for h in root.handlers):
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    if not runtime.is_databricks():
        # Windows pipes default to cp1252, and MLflow prints emoji to stdout
        # ("View run ..."): an unencodable character would crash the pipeline.
        for console in (sys.stdout, sys.stderr):
            if hasattr(console, "reconfigure"):
                console.reconfigure(errors="backslashreplace")
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = ML_ROOT / logdir / process_name / f"{stamp}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.info("Logging initialised for %s", process_name)
