from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from typing import Any, Callable

from .callbacks.base import Callback
from .callbacks.checkpoint import ModelCheckpoint
from .callbacks.progress import ProgressCallback
from .config import OptimizerConfig, SchedulerConfig, Strategy, TrainConfig
from .strategies.base import BaseStrategy
from .strategies.ddp import DDPStrategy
from .strategies.fsdp import FSDPStrategy
from .utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def launch(
    train_fn: Callable[[int, int], None],
    world_size: int,
    **kwargs: Any,
) -> None:
    """Spawn *world_size* processes each calling ``train_fn(rank, world_size)``.

    Thin wrapper around :func:`torch.multiprocessing.spawn`.

    Args:
        train_fn: Callable with signature ``(rank: int, world_size: int) -> None``.
        world_size: Number of processes to spawn.
        **kwargs: Forwarded to :func:`torch.multiprocessing.spawn`.
    """
    import torch.multiprocessing as mp

    mp.spawn(train_fn, args=(world_size,), nprocs=world_size, join=True, **kwargs)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Orchestrates the distributed training loop.

    ``Trainer`` handles device placement, strategy setup, optimizer and
    scheduler construction, the train/validation loop, callback dispatch,
    and checkpoint delegation.

    Args:
        model: Unwrapped :class:`torch.nn.Module` to train.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation data.
        config: :class:`~pydist_train.config.TrainConfig`; defaults are used
            when not supplied.
        strategy: Pre-built strategy instance.  When omitted one is derived
            from ``config.strategy``.
        callbacks: Extra callbacks appended after the built-in
            :class:`~pydist_train.callbacks.ProgressCallback` and
            :class:`~pydist_train.callbacks.ModelCheckpoint`.
        loss_fn: ``(model, batch) -> scalar_tensor``.  Defaults to
            cross-entropy on the first two elements of each batch.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: TrainConfig | None = None,
        strategy: BaseStrategy | None = None,
        callbacks: list[Callback] | None = None,
        loss_fn: Callable | None = None,
    ) -> None:
        self.raw_model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainConfig()
        self.loss_fn = loss_fn or _cross_entropy

        self.strategy: BaseStrategy = strategy or _make_strategy(self.config)
        self.callbacks: list[Callback] = [
            ProgressCallback(log_every_n_steps=self.config.log_every_n_steps),
            ModelCheckpoint(
                dirpath=self.config.checkpoint.dirpath,
                every_n_epochs=self.config.checkpoint.every_n_epochs,
                every_n_steps=self.config.checkpoint.every_n_steps,
                keep_last_n=self.config.checkpoint.keep_last_n,
                save_best=self.config.checkpoint.save_best,
                monitor=self.config.checkpoint.monitor,
                mode=self.config.checkpoint.mode,
            ),
        ] + (callbacks or [])

        # Set during fit()
        self.model: nn.Module = model
        self.optimizer: Optimizer | None = None
        self.scheduler: LRScheduler | None = None
        self.current_epoch: int = 0
        self.global_step: int = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(self, rank: int = 0, world_size: int = 1) -> None:
        """Run the full training loop.

        For single-process training call ``trainer.fit()``.  For
        multi-GPU training spawn one process per GPU and call
        ``trainer.fit(rank=rank, world_size=world_size)`` in each.
        """
        torch.manual_seed(self.config.seed + rank)
        self.strategy.setup(rank=rank, world_size=world_size)
        self.model = self.strategy.wrap_model(self.raw_model)

        steps_per_epoch = len(self.train_loader)
        total_steps = (
            self.config.max_steps
            if self.config.max_steps is not None
            else self.config.max_epochs * steps_per_epoch
        )

        self.optimizer = _build_optimizer(self.config.optimizer, self.model)
        self.scheduler = _build_scheduler(
            self.config.scheduler, self.optimizer, total_steps
        )

        self._call("on_train_start")
        try:
            for epoch in range(1, self.config.max_epochs + 1):
                self.current_epoch = epoch
                _set_epoch(self.train_loader, epoch)

                self._call("on_epoch_start", epoch)
                train_metrics = self._train_epoch()
                epoch_metrics: dict[str, Any] = {**train_metrics}

                if (
                    self.val_loader is not None
                    and epoch % self.config.val_every_n_epochs == 0
                ):
                    self._call("on_validation_start")
                    val_metrics = self._val_epoch()
                    self._call("on_validation_end", val_metrics)
                    epoch_metrics.update(val_metrics)

                self._call("on_epoch_end", epoch, epoch_metrics)

                if (
                    self.config.max_steps is not None
                    and self.global_step >= self.config.max_steps
                ):
                    logger.info("Reached max_steps=%d — stopping.", self.config.max_steps)
                    break
        finally:
            self._call("on_train_end")
            self.strategy.teardown()

    # ------------------------------------------------------------------ #
    # Training / validation loops                                          #
    # ------------------------------------------------------------------ #

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.strategy.device)
        n_steps = 0
        accum = self.config.gradient_accumulation_steps
        n_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()  # zeroed per-batch (bug: clears accumulated grads)
            batch = _to_device(batch, self.strategy.device)
            self._call("on_step_start", self.global_step)

            with self.strategy.autocast():
                loss = self.loss_fn(self.model, batch)
                loss = loss / accum

            self.strategy.backward(loss)
            total_loss += loss.detach() * accum
            n_steps += 1

            is_accum_boundary = (batch_idx + 1) % accum == 0
            is_last_batch = (batch_idx + 1) == n_batches

            if is_accum_boundary or is_last_batch:
                if self.config.gradient_clip_val is not None:
                    self.strategy.clip_grad_norm(self.model, self.config.gradient_clip_val)
                self.strategy.optimizer_step(self.optimizer)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.global_step += 1
                step_loss = (loss.detach() * accum).item()
                self._call("on_step_end", self.global_step, {"train_loss": step_loss})

                if (
                    self.config.max_steps is not None
                    and self.global_step >= self.config.max_steps
                ):
                    break

        avg = (total_loss / max(n_steps, 1)).item()
        return {"train_loss": avg}

    @torch.no_grad()
    def _val_epoch(self) -> dict[str, float]:
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.strategy.device)
        n_batches = 0
        for batch in self.val_loader:
            batch = _to_device(batch, self.strategy.device)
            with self.strategy.autocast():
                loss = self.loss_fn(self.model, batch)
            total_loss += loss.detach()
            n_batches += 1
        avg = (total_loss / max(n_batches, 1)).item()
        return {"val_loss": avg}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _call(self, hook: str, *args: Any) -> None:
        for cb in self.callbacks:
            getattr(cb, hook)(self, *args)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _cross_entropy(model: nn.Module, batch: Any) -> torch.Tensor:
    inputs, targets = batch[0], batch[1]
    return nn.functional.cross_entropy(model(inputs), targets)


def _to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        converted = [_to_device(b, device) for b in batch]
        return type(batch)(converted)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch


def _set_epoch(loader: DataLoader, epoch: int) -> None:
    if isinstance(getattr(loader, "sampler", None), DistributedSampler):
        loader.sampler.set_epoch(epoch)


def _make_strategy(config: TrainConfig) -> BaseStrategy:
    if config.strategy == Strategy.FSDP:
        return FSDPStrategy(precision=config.precision.value)
    return DDPStrategy(precision=config.precision.value)


def _build_optimizer(config: OptimizerConfig, model: nn.Module) -> Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    name = config.name.lower()
    if name == "adam":
        return torch.optim.Adam(
            params, lr=config.lr, weight_decay=config.weight_decay, **config.kwargs
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params, lr=config.lr, weight_decay=config.weight_decay, **config.kwargs
        )
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=config.lr, weight_decay=config.weight_decay, **config.kwargs
        )
    raise ValueError(f"Unknown optimizer: {config.name!r}.  Choose adam, adamw, or sgd.")


def _build_scheduler(
    config: SchedulerConfig | None,
    optimizer: Optimizer,
    num_training_steps: int,
) -> LRScheduler | None:
    if config is None:
        return None
    name = config.name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, num_training_steps), **config.kwargs
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **config.kwargs)
    if name == "linear":
        return torch.optim.lr_scheduler.LinearLR(optimizer, **config.kwargs)
    raise ValueError(f"Unknown scheduler: {config.name!r}.  Choose cosine, step, or linear.")
