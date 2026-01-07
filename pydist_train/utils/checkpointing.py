from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import torch

from .distributed import is_main_process
from .logging import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    state: dict[str, Any],
    dirpath: str,
    filename: str,
) -> Path:
    """Serialise *state* to ``dirpath/filename`` on the main process only.

    Creates *dirpath* if it does not exist.  Returns the full path regardless
    of whether this process actually wrote the file.
    """
    path = Path(dirpath)
    full_path = path / filename
    if not is_main_process():
        return full_path
    path.mkdir(parents=True, exist_ok=True)
    torch.save(state, full_path)
    logger.info("Saved checkpoint: %s", full_path)
    return full_path


def load_checkpoint(path: str, map_location: str | None = None) -> dict[str, Any]:
    """Load and return a checkpoint from *path*."""
    state: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    logger.info("Loaded checkpoint: %s", path)
    return state


def find_latest_checkpoint(dirpath: str, prefix: str = "epoch") -> str | None:
    """Return the path to the most recently saved checkpoint or ``None``."""
    matches = sorted(glob.glob(os.path.join(dirpath, f"{prefix}_*.pt")))
    return matches[-1] if matches else None


def rotate_checkpoints(dirpath: str, prefix: str, keep_last_n: int) -> None:
    """Delete old ``prefix_*.pt`` checkpoints, keeping the *keep_last_n* newest.

    Only executes on the main process to avoid concurrent filesystem mutations.
    """
    if not is_main_process():
        return
    checkpoints = sorted(glob.glob(os.path.join(dirpath, f"{prefix}_*.pt")))
    for old in checkpoints[:-keep_last_n]:
        os.remove(old)
        logger.debug("Removed old checkpoint: %s", old)
