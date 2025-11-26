from __future__ import annotations

import logging
import sys
from typing import IO

from .distributed import get_rank, is_main_process

_FORMAT = "[%(asctime)s][%(name)s][rank %(rank)s][%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _RankFilter(logging.Filter):
    """Inject the current process rank into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = get_rank()  # type: ignore[attr-defined]
        return True


class _MainProcessFilter(logging.Filter):
    """Suppress log records on non-main ranks."""

    def filter(self, record: logging.LogRecord) -> bool:
        return is_main_process()


def get_logger(
    name: str,
    level: int = logging.INFO,
    main_process_only: bool = True,
    stream: IO[str] | None = None,
) -> logging.Logger:
    """Return a logger configured for distributed training.

    Args:
        name: Logger name, typically ``__name__``.
        level: Logging verbosity level.
        main_process_only: When *True* (default), suppress output on all ranks
            except rank 0.
        stream: Output stream; defaults to *sys.stdout*.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(_RankFilter())
    if main_process_only:
        handler.addFilter(_MainProcessFilter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_verbosity(level: int) -> None:
    """Set *level* on every active pydist_train logger."""
    manager = logging.Logger.manager
    for name, obj in manager.loggerDict.items():
        if name.startswith("pydist_train") and isinstance(obj, logging.Logger):
            obj.setLevel(level)
