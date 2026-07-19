"""Tests for the JSONL file logging sink."""

import gzip
import json

from loguru import logger

from cherryai_api import logging_setup


def _reset_configured(monkeypatch) -> None:
    monkeypatch.setattr(logging_setup, "_configured", False)


def test_setup_creates_dir_and_writes_jsonl(tmp_path, monkeypatch):
    _reset_configured(monkeypatch)
    log_dir = tmp_path / "logs"
    logging_setup.setup_file_logging(str(log_dir))
    try:
        logger.bind(check="jsonl").info("hello from test")
        logger.complete()
        log_file = log_dir / "cherryai-api.jsonl"
        assert log_file.exists()
        lines = [line for line in log_file.read_text().splitlines() if line]
        records = [json.loads(line) for line in lines]
        assert any(r["record"]["message"] == "hello from test" for r in records)
    finally:
        _remove_test_sink(log_dir)


def test_setup_is_idempotent(tmp_path, monkeypatch):
    _reset_configured(monkeypatch)
    log_dir = tmp_path / "logs"
    logging_setup.setup_file_logging(str(log_dir))
    try:
        # Second call must not install a second sink (no duplicate lines).
        logging_setup.setup_file_logging(str(log_dir))
        logger.info("once only")
        logger.complete()
        lines = (log_dir / "cherryai-api.jsonl").read_text().splitlines()
        # One sink -> the record lands exactly once (serialize embeds the
        # message twice per line, so count lines, not substring hits).
        assert sum("once only" in line for line in lines) == 1
    finally:
        _remove_test_sink(log_dir)


def test_rotation_compresses_old_files(tmp_path, monkeypatch):
    _reset_configured(monkeypatch)
    log_dir = tmp_path / "logs"
    logging_setup.setup_file_logging(str(log_dir))
    try:
        # ~2500 records of ~600 serialized bytes comfortably exceeds 1 MB.
        for i in range(2500):
            logger.bind(filler="x" * 400).info(f"rotation filler {i}")
        logger.complete()
        rotated = list(log_dir.glob("*.jsonl.gz"))
        assert rotated, "expected at least one gzipped rotation"
        with gzip.open(rotated[0], "rt") as fh:
            json.loads(fh.readline())
    finally:
        _remove_test_sink(log_dir)


def _remove_test_sink(log_dir) -> None:
    """Drop the sink added for this test so later tests don't write to tmp."""
    target = str(log_dir / "cherryai-api.jsonl")
    for handler_id, handler in list(logger._core.handlers.items()):
        if target in str(getattr(handler, "_name", "")):
            logger.remove(handler_id)
