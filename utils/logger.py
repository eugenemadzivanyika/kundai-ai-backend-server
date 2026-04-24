import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

os.makedirs("logs", exist_ok=True)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "meta"):
            entry.update(record.meta)
        return json.dumps(entry)


def _build_logger() -> logging.Logger:
    log = logging.getLogger("kundai_ai")
    if log.handlers:
        return log

    log.setLevel(logging.DEBUG)
    fmt = _JsonFormatter()

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        "logs/error.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(fmt)

    log.addHandler(console)
    log.addHandler(file_handler)
    return log


_logger = _build_logger()


def log_error(message: str, **meta) -> None:
    _logger.error(message, extra={"meta": meta})


def log_info(message: str, **meta) -> None:
    _logger.info(message, extra={"meta": meta})
