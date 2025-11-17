from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator

import torch
import torch.distributed as dist


def init_process_group(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: str = "12355",
) -> None:
    """Initialize the distributed process group.

    Also pins the current process to the correct GPU when CUDA is available.
    """
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())


def destroy_process_group() -> None:
    """Destroy the process group if one is active."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Global rank of the current process (0 when not distributed)."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_local_rank() -> int:
    """Local rank within the current node (read from the environment)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Total number of processes (1 when not distributed)."""
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    """Return True only on the global rank-0 process."""
    return get_rank() == 0


def barrier() -> None:
    """Block until all processes reach this point."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average *tensor* across all ranks in-place and return it."""
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(get_world_size())
    return tensor


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast *tensor* from rank *src* to every other rank."""
    if dist.is_initialized():
        dist.broadcast(tensor, src=src)
    return tensor


def gather_object(obj: Any, dst: int = 0) -> list[Any] | None:
    """Gather an arbitrary Python object from all ranks to *dst*.

    Returns a list of objects on *dst* and ``None`` on all other ranks.
    """
    if not dist.is_initialized():
        return [obj]
    world_size = get_world_size()
    output: list[Any] | None = [None] * world_size if get_rank() == dst else None
    dist.gather_object(obj, output, dst=dst)
    return output


@contextmanager
def distributed_context(
    rank: int,
    world_size: int,
    backend: str = "nccl",
) -> Generator[None, None, None]:
    """Context manager that sets up and tears down the process group."""
    init_process_group(rank=rank, world_size=world_size, backend=backend)
    try:
        yield
    finally:
        destroy_process_group()
