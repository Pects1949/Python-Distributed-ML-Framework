"""Tests for BaseStrategy contract and DDPStrategy single-process behaviour."""
import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from pydist_train.strategies.base import BaseStrategy
from pydist_train.strategies.ddp import DDPStrategy


# ---------------------------------------------------------------------------
# Minimal concrete strategy for testing the abstract base
# ---------------------------------------------------------------------------

class _CPUStrategy(BaseStrategy):
    def setup(self, rank: int, world_size: int) -> None:
        self._device = torch.device("cpu")

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model.to(self._device)

    def teardown(self) -> None:
        pass


class TestBaseStrategy:
    def test_device_raises_before_setup(self):
        s = _CPUStrategy()
        with pytest.raises(RuntimeError, match="set up"):
            _ = s.device

    def test_device_after_setup(self):
        s = _CPUStrategy()
        s.setup(0, 1)
        assert s.device == torch.device("cpu")

    def test_wrap_model(self):
        s = _CPUStrategy()
        s.setup(0, 1)
        model = nn.Linear(4, 2)
        wrapped = s.wrap_model(model)
        assert isinstance(wrapped, nn.Linear)
        assert next(wrapped.parameters()).device == torch.device("cpu")

    def test_clip_grad_norm_no_grads(self):
        s = _CPUStrategy()
        s.setup(0, 1)
        model = nn.Linear(4, 2)
        # No backward pass → no gradients
        norm = s.clip_grad_norm(model, max_norm=1.0)
        assert norm == pytest.approx(0.0)

    def test_clip_grad_norm_with_grads(self):
        s = _CPUStrategy()
        s.setup(0, 1)
        model = nn.Linear(4, 2)
        out = model(torch.randn(3, 4))
        out.sum().backward()
        norm = s.clip_grad_norm(model, max_norm=1.0)
        assert norm >= 0.0

    def test_model_state_dict_and_load(self):
        s = _CPUStrategy()
        s.setup(0, 1)
        model = nn.Linear(4, 2)
        state = s.model_state_dict(model)
        assert "weight" in state and "bias" in state
        # Corrupt weights then reload
        with torch.no_grad():
            model.weight.fill_(0.0)
        s.load_model_state_dict(model, state)
        assert not torch.all(model.weight == 0.0)

    def test_autocast_fp32_does_not_raise(self):
        s = _CPUStrategy()
        s.setup(0, 1)
        with s.autocast():
            x = torch.randn(2, 4)
            assert x.dtype == torch.float32


class TestDDPStrategy:
    def test_single_process_gloo(self):
        strategy = DDPStrategy(backend="gloo")
        strategy.setup(rank=0, world_size=1)
        model = nn.Linear(4, 2)
        wrapped = strategy.wrap_model(model)
        assert isinstance(wrapped, DistributedDataParallel)
        out = wrapped(torch.randn(3, 4))
        assert out.shape == (3, 2)
        strategy.teardown()

    def test_model_state_dict_unwraps_ddp(self):
        strategy = DDPStrategy(backend="gloo")
        strategy.setup(rank=0, world_size=1)
        model = nn.Linear(4, 2)
        wrapped = strategy.wrap_model(model)
        state = strategy.model_state_dict(wrapped)
        # DDP wraps under .module; state dict should look like raw Linear
        assert "weight" in state
        strategy.teardown()

    def test_nccl_falls_back_to_gloo_on_cpu(self):
        strategy = DDPStrategy(backend="nccl")  # nccl requested, no GPU
        strategy.setup(rank=0, world_size=1)
        # Should succeed and land on CPU
        assert strategy.device == torch.device("cpu")
        strategy.teardown()
