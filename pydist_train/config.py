from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Strategy(str, Enum):
    """Distributed training strategy."""

    DDP = "ddp"
    FSDP = "fsdp"


class Precision(str, Enum):
    """Floating-point precision for training."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""

    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-2
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    """Configuration for the learning-rate scheduler."""

    name: str = "cosine"
    warmup_steps: int = 0
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointConfig:
    """Configuration for automatic checkpointing."""

    dirpath: str = "./checkpoints"
    every_n_epochs: int = 1
    every_n_steps: int | None = None
    keep_last_n: int = 3
    save_best: bool = True
    monitor: str = "val_loss"
    mode: str = "min"  # "min" or "max"

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")


@dataclass
class TrainConfig:
    """Top-level training configuration.

    Example::

        config = TrainConfig(
            max_epochs=10,
            strategy=Strategy.DDP,
            precision=Precision.BF16,
            optimizer=OptimizerConfig(name="adamw", lr=3e-4),
            scheduler=SchedulerConfig(name="cosine", warmup_steps=500),
        )
    """

    # ---- Duration ----
    max_epochs: int = 10
    max_steps: int | None = None

    # ---- Optimisation ----
    gradient_clip_val: float | None = 1.0
    gradient_accumulation_steps: int = 1

    # ---- Logging ----
    log_every_n_steps: int = 10

    # ---- Validation ----
    val_every_n_epochs: int = 1

    # ---- Distribution ----
    strategy: Strategy = Strategy.DDP
    precision: Precision = Precision.FP32
    num_nodes: int = 1
    devices: int = 1

    # ---- Reproducibility ----
    seed: int = 42

    # ---- Sub-configs ----
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig | None = None
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self) -> None:
        if isinstance(self.strategy, str):
            self.strategy = Strategy(self.strategy)
        if isinstance(self.precision, str):
            self.precision = Precision(self.precision)
        if isinstance(self.optimizer, dict):
            self.optimizer = OptimizerConfig(**self.optimizer)
        if isinstance(self.scheduler, dict):
            self.scheduler = SchedulerConfig(**self.scheduler)
        if isinstance(self.checkpoint, dict):
            self.checkpoint = CheckpointConfig(**self.checkpoint)
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.devices < 1:
            raise ValueError("devices must be >= 1")
        if self.num_nodes < 1:
            raise ValueError("num_nodes must be >= 1")
