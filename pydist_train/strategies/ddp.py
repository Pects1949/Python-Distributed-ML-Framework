from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..utils.distributed import destroy_process_group, init_process_group
from .base import BaseStrategy


class DDPStrategy(BaseStrategy):
    """DistributedDataParallel training strategy.

    Synchronises gradients across all ranks via all-reduce after each
    backward pass.  On CPU-only systems the backend automatically falls
    back to ``gloo`` when ``nccl`` is requested.

    Args:
        precision: ``"fp32"``, ``"fp16"``, or ``"bf16"``.
        find_unused_parameters: Passed to DDP; required when some model
            parameters do not receive gradients on every forward pass.
        gradient_as_bucket_view: Reduces peak memory by reusing gradient
            tensors as bucket views.
        backend: Distributed backend (default ``"nccl"``).
    """

    def __init__(
        self,
        precision: str = "fp32",
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True,
        backend: str = "nccl",
    ) -> None:
        super().__init__(precision)
        self.find_unused_parameters = find_unused_parameters
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.backend = backend

    def setup(self, rank: int, world_size: int) -> None:
        backend = self.backend
        if not torch.cuda.is_available() and backend == "nccl":
            backend = "gloo"
        init_process_group(rank=rank, world_size=world_size, backend=backend)
        if torch.cuda.is_available():
            self._device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
            torch.cuda.set_device(self._device)
        else:
            self._device = torch.device("cpu")

    def wrap_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        kwargs: dict = dict(
            find_unused_parameters=self.find_unused_parameters,
            gradient_as_bucket_view=self.gradient_as_bucket_view,
        )
        if torch.cuda.is_available():
            kwargs["device_ids"] = [self.device.index]
        return DDP(model, **kwargs)

    def teardown(self) -> None:
        destroy_process_group()
