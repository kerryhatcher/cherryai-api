"""File logging for the API: JSONL sink with rotation and compressed retention.

Loguru's default stderr sink is left untouched for console output; this module
only adds the persistent file sink.
"""

from pathlib import Path

from loguru import logger

_configured = False


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
