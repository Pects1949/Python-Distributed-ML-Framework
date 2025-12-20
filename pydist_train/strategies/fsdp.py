from __future__ import annotations

import functools
from typing import Callable, Type

import torch
import torch.nn as nn

from ..utils.distributed import destroy_process_group, init_process_group
from ..utils.logging import get_logger
from .base import BaseStrategy

logger = get_logger(__name__)

try:
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )

    _FSDP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FSDP_AVAILABLE = False


_SHARDING_STRATEGIES = {
    "full_shard": "FULL_SHARD",
    "shard_grad_op": "SHARD_GRAD_OP",
    "no_shard": "NO_SHARD",
}


class FSDPStrategy(BaseStrategy):
    """FullyShardedDataParallel training strategy.

    Shards model parameters, gradients, and optimiser states across all
    ranks, enabling training of models that exceed single-GPU memory.

    Args:
        precision: ``"fp32"``, ``"fp16"``, or ``"bf16"``.
        sharding_strategy: One of ``"full_shard"`` (default),
            ``"shard_grad_op"``, or ``"no_shard"``.
        transformer_layer_cls: Target this layer class with the FSDP
            wrapping policy (e.g. ``nn.TransformerEncoderLayer``).
        min_num_params: Minimum parameter count for size-based wrapping
            when *transformer_layer_cls* is not given.
        cpu_offload: Offload parameters and gradients to CPU when idle.
        backend: Distributed backend (default ``"nccl"``).
    """

    def __init__(
        self,
        precision: str = "bf16",
        sharding_strategy: str = "full_shard",
        transformer_layer_cls: Type[nn.Module] | None = None,
        min_num_params: int = 100_000,
        cpu_offload: bool = False,
        backend: str = "nccl",
    ) -> None:
        if not _FSDP_AVAILABLE:
            raise ImportError("FSDPStrategy requires PyTorch >= 2.1")
        if sharding_strategy not in _SHARDING_STRATEGIES:
            raise ValueError(
                f"sharding_strategy must be one of {list(_SHARDING_STRATEGIES)}, "
                f"got {sharding_strategy!r}"
            )
        super().__init__(precision)
        self.sharding_strategy = sharding_strategy
        self.transformer_layer_cls = transformer_layer_cls
        self.min_num_params = min_num_params
        self.cpu_offload = cpu_offload
        self.backend = backend

    def _sharding_strategy(self) -> "ShardingStrategy":
        return ShardingStrategy[_SHARDING_STRATEGIES[self.sharding_strategy]]

    def _mixed_precision(self) -> "MixedPrecision | None":
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
        dtype = dtype_map.get(self.precision)
        if dtype is None:
            return None
        return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    def _auto_wrap_policy(self) -> Callable:
        if self.transformer_layer_cls is not None:
            return functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={self.transformer_layer_cls},
            )
        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=self.min_num_params,
        )

    def setup(self, rank: int, world_size: int) -> None:
        init_process_group(rank=rank, world_size=world_size, backend=self.backend)
        if torch.cuda.is_available():
            self._device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
            torch.cuda.set_device(self._device)
        else:
            self._device = torch.device("cpu")

    def wrap_model(self, model: nn.Module) -> nn.Module:
        cpu_offload_cfg = None
        if self.cpu_offload:
            from torch.distributed.fsdp import CPUOffload

            cpu_offload_cfg = CPUOffload(offload_params=True)

        return FSDP(
            model,
            sharding_strategy=self._sharding_strategy(),
            mixed_precision=self._mixed_precision(),
            auto_wrap_policy=self._auto_wrap_policy(),
            cpu_offload=cpu_offload_cfg,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=self._device,
        )

    def teardown(self) -> None:
        destroy_process_group()

    def model_state_dict(self, model: nn.Module) -> dict:
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()

    def load_model_state_dict(self, model: nn.Module, state: dict) -> None:
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            model.load_state_dict(state)
