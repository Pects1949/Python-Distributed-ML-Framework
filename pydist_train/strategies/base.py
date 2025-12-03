from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator

import torch
import torch.nn as nn
from torch.optim import Optimizer


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *_: object) -> None:
        pass


class BaseStrategy(ABC):
    """Abstract base for distributed training strategies.

    Subclasses handle backend initialisation, model wrapping, gradient
    synchronisation, and checkpoint serialisation.  The ``Trainer`` calls
    these hooks so that training code stays strategy-agnostic.
    """

    def __init__(self, precision: str = "fp32") -> None:
        self.precision = precision
        self._device: torch.device | None = None

    # ------------------------------------------------------------------ #
    # Device                                                               #
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> torch.device:
        if self._device is None:
            raise RuntimeError("Strategy has not been set up — call setup() first.")
        return self._device

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def setup(self, rank: int, world_size: int) -> None:
        """Initialise the distributed backend and configure the device."""

    @abstractmethod
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap *model* for distributed execution and move it to the device."""

    @abstractmethod
    def teardown(self) -> None:
        """Shut down the distributed backend."""

    # ------------------------------------------------------------------ #
    # Training primitives                                                  #
    # ------------------------------------------------------------------ #

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def optimizer_step(self, optimizer: Optimizer) -> None:
        optimizer.step()

    def clip_grad_norm(self, model: nn.Module, max_norm: float) -> float:
        """Clip gradients and return the pre-clip total norm."""
        params = [p for p in model.parameters() if p.grad is not None]
        if not params:
            return 0.0
        return torch.nn.utils.clip_grad_norm_(params, max_norm).item()

    @contextmanager
    def autocast(self) -> Generator[None, None, None]:
        """Autocast context appropriate for the configured precision."""
        if self.precision == "fp16":
            with torch.autocast(self.device.type, dtype=torch.float16):
                yield
        elif self.precision == "bf16":
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                yield
        else:
            yield

    # ------------------------------------------------------------------ #
    # Checkpointing                                                        #
    # ------------------------------------------------------------------ #

    def model_state_dict(self, model: nn.Module) -> dict[str, Any]:
        """Return the unwrapped model state dict."""
        raw = getattr(model, "module", model)
        return raw.state_dict()

    def load_model_state_dict(self, model: nn.Module, state: dict[str, Any]) -> None:
        """Load *state* into the (possibly wrapped) model."""
        raw = getattr(model, "module", model)
        raw.load_state_dict(state)
