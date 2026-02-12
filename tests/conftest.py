"""Shared fixtures and test helpers."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Auto-cleanup: destroy any lingering process group after each test so that
# subsequent tests can re-initialise the distributed backend cleanly.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _cleanup_dist():
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Common model / data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_model() -> nn.Module:
    """A small two-layer MLP: input 8 → 4 classes."""
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


@pytest.fixture()
def tensor_dataset() -> TensorDataset:
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    y = torch.randint(0, 4, (64,))
    return TensorDataset(x, y)


@pytest.fixture()
def train_loader(tensor_dataset: TensorDataset) -> DataLoader:
    return DataLoader(tensor_dataset, batch_size=16, shuffle=False)


@pytest.fixture()
def val_loader(tensor_dataset: TensorDataset) -> DataLoader:
    return DataLoader(tensor_dataset, batch_size=16, shuffle=False)
