from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.checkpointing import rotate_checkpoints, save_checkpoint
from ..utils.logging import get_logger
from .base import Callback

if TYPE_CHECKING:
    from ..trainer import Trainer

logger = get_logger(__name__)


class ModelCheckpoint(Callback):
    """Save model checkpoints during training.

    The best checkpoint (by *monitor*) is always preserved independently
    of the rotation policy.

    Args:
        dirpath: Directory where checkpoints are written.
        every_n_epochs: Checkpoint every N epochs (default 1).
        every_n_steps: Also checkpoint every N global steps if set.
        keep_last_n: Number of recent epoch checkpoints to keep.
        save_best: Always keep the checkpoint with the best metric.
        monitor: Key in the epoch metrics dict to watch.
        mode: ``"min"`` if a lower value is better, ``"max"`` otherwise.
    """

    def __init__(
        self,
        dirpath: str = "./checkpoints",
        every_n_epochs: int = 1,
        every_n_steps: int | None = None,
        keep_last_n: int = 3,
        save_best: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        self.dirpath = dirpath
        self.every_n_epochs = every_n_epochs
        self.every_n_steps = every_n_steps
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self._best: float = math.inf if mode == "min" else -math.inf
        self._best_path: Path | None = None

    def _is_better(self, value: float) -> bool:
        return value < self._best if self.mode == "min" else value > self._best

    def _build_state(self, trainer: "Trainer") -> dict[str, Any]:
        return {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "model": trainer.strategy.model_state_dict(trainer.model),
            "optimizer": trainer.optimizer.state_dict(),
        }

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: dict[str, Any]
    ) -> None:
        if epoch % self.every_n_epochs != 0:
            return
        state = self._build_state(trainer)
        save_checkpoint(state, self.dirpath, f"epoch_{epoch:04d}.pt")
        rotate_checkpoints(self.dirpath, prefix="epoch", keep_last_n=self.keep_last_n)

        if self.save_best and self.monitor in metrics:
            value = float(metrics[self.monitor])
            if self._is_better(value):
                self._best = value
                self._best_path = save_checkpoint(state, self.dirpath, "best.pt")
                logger.info("New best %s=%.6f → %s", self.monitor, value, self._best_path)

    def on_step_end(
        self, trainer: "Trainer", step: int, metrics: dict[str, Any]
    ) -> None:
        if self.every_n_steps is None or step % self.every_n_steps != 0:
            return
        state = self._build_state(trainer)
        save_checkpoint(state, self.dirpath, f"step_{step:08d}.pt")
        rotate_checkpoints(self.dirpath, prefix="step", keep_last_n=self.keep_last_n)
