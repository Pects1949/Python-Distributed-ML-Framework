from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..trainer import Trainer


class Callback:
    """Base class for training callbacks.

    Subclass and override only the hooks you need.  Every hook receives
    the ``Trainer`` instance so you have full access to training state.

    Hook execution order per epoch::

        on_train_start
          on_epoch_start
            on_step_start
            on_step_end
          on_validation_start
          on_validation_end
          on_epoch_end
        on_train_end
    """

    def on_train_start(self, trainer: "Trainer") -> None:
        """Called once before the training loop begins."""

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called once after the training loop ends (even on exception)."""

    def on_epoch_start(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the start of each training epoch."""

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: dict[str, Any]
    ) -> None:
        """Called at the end of each epoch with aggregated metrics."""

    def on_step_start(self, trainer: "Trainer", step: int) -> None:
        """Called before each optimisation step."""

    def on_step_end(
        self, trainer: "Trainer", step: int, metrics: dict[str, Any]
    ) -> None:
        """Called after each optimisation step."""

    def on_validation_start(self, trainer: "Trainer") -> None:
        """Called before the validation loop."""

    def on_validation_end(
        self, trainer: "Trainer", metrics: dict[str, Any]
    ) -> None:
        """Called after the validation loop with aggregated metrics."""
