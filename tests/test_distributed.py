"""Tests for pydist_train.utils.distributed (single-process / no-dist path)."""
import pytest
import torch
import torch.distributed as dist

from pydist_train.utils.distributed import (
    all_reduce_mean,
    barrier,
    broadcast,
    get_local_rank,
    get_rank,
    get_world_size,
    is_main_process,
)


def test_rank_without_dist():
    assert not dist.is_initialized()
    assert get_rank() == 0


def test_world_size_without_dist():
    assert get_world_size() == 1


def test_is_main_process_without_dist():
    assert is_main_process() is True


def test_get_local_rank_default(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert get_local_rank() == 0


def test_get_local_rank_from_env(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert get_local_rank() == 3


def test_barrier_is_noop_without_dist():
    barrier()  # must not raise


def test_all_reduce_mean_passthrough():
    t = torch.tensor(7.5)
    result = all_reduce_mean(t)
    assert result.item() == pytest.approx(7.5)


def test_all_reduce_mean_does_not_change_value_without_dist():
    t = torch.tensor([1.0, 2.0, 3.0])
    out = all_reduce_mean(t)
    assert torch.allclose(out, torch.tensor([1.0, 2.0, 3.0]))


def test_broadcast_passthrough():
    t = torch.tensor([10.0, 20.0])
    out = broadcast(t, src=0)
    assert torch.allclose(out, t)


def test_all_reduce_mean_with_dist(tmp_path):
    """Use gloo to verify all_reduce_mean in a real single-process group."""
    import os

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29510")
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        t = torch.tensor(4.0)
        result = all_reduce_mean(t)
        assert result.item() == pytest.approx(4.0)
    finally:
        dist.destroy_process_group()
