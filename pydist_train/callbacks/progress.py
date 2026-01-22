from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..utils.distributed import is_main_process
from ..utils.logging import get_logger
from .base import Callback

if TYPE_CHECKING:
    from ..trainer import Trainer

logger = get_logger(__name__)


class ProgressCallback(Callback):
    """Log training progress to stdout.

    Args:
        log_every_n_steps: Emit a log line every N global optimisation steps.
    """

    def __init__(self, log_every_n_steps: int = 10) -> None:
        self.log_every_n_steps = log_every_n_steps
        self._epoch_t0: float = 0.0

    def on_epoch_start(self, trainer: "Trainer", epoch: int) -> None:
        self._epoch_t0 = time.monotonic()
        if is_main_process():
            logger.info("── Epoch %d / %d ──", epoch, trainer.config.max_epochs)

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: dict[str, Any]
    ) -> None:
        if not is_main_process():
            return
        elapsed = time.monotonic() - self._epoch_t0
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info("Epoch %d done in %.1fs  %s", epoch, elapsed, metric_str)

    def on_step_end(
        self, trainer: "Trainer", step: int, metrics: dict[str, Any]
    ) -> None:
        if not is_main_process() or step % self.log_every_n_steps != 0:
            return
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info("Step %d  %s", step, metric_str)

    def on_validation_end(
        self, trainer: "Trainer", metrics: dict[str, Any]
    ) -> None:
        if not is_main_process():
            return
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info("Validation  %s", metric_str)
