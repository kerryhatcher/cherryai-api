"""File logging for the API: JSONL sink with rotation and compressed retention.

Loguru's default stderr sink is left untouched for console output; this module
only adds the persistent file sink and bridges uvicorn's stdlib logging into
loguru so access logs and internal errors land in the same JSONL file.
"""

import logging
import sys
from pathlib import Path

from loguru import logger

_configured = False


class _UvicornLogBridge(logging.Handler):
    """Forward stdlib log records into loguru so uvicorn output is captured."""

    def emit(self, record: logging.LogRecord) -> None:
        # Skip loguru's own internal records to avoid infinite loops.
        if record.name == "loguru":
            return
        # Map stdlib levels to loguru level names (uppercase).
        level = record.levelname
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def setup_file_logging(log_dir: str = "logs") -> None:
    """Add a JSONL file sink rotating at 1 MB, keeping 7 gzipped rotations.

    Idempotent: called from both the CLI entrypoint and the FastAPI lifespan
    (the latter covers ``uvicorn --reload``, which runs the app in a child
    process), but only ever installs one sink per process.
    """
    global _configured
    if _configured:
        return
    _configured = True
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    logger.add(
        path / "cherryai-api.jsonl",
        serialize=True,
        rotation="1 MB",
        retention=7,
        compression="gz",
        enqueue=True,
    )

    # Bridge uvicorn's stdlib logging into loguru so access logs and
    # internal uvicorn errors appear in the JSONL file alongside app logs.
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = [_UvicornLogBridge()]
    uvicorn_logger.propagate = False

    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.handlers = [_UvicornLogBridge()]
    uvicorn_access.propagate = False

    # Intercept unhandled exceptions that would otherwise only go to stderr.
    def _log_unhandled(exc_type, exc_value, exc_tb):
        logger.opt(exception=(exc_type, exc_value, exc_tb)).error("Unhandled exception")

    sys.excepthook = _log_unhandled
