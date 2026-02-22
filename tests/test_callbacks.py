"""Tests for Callback, ModelCheckpoint, and ProgressCallback."""
import math
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from pydist_train.callbacks.base import Callback
from pydist_train.callbacks.checkpoint import ModelCheckpoint
from pydist_train.callbacks.progress import ProgressCallback


# ---------------------------------------------------------------------------
# Stub trainer
# ---------------------------------------------------------------------------

def _make_trainer(epoch: int = 1, step: int = 10) -> MagicMock:
    trainer = MagicMock()
    trainer.current_epoch = epoch
    trainer.global_step = step
    trainer.config.max_epochs = 5
    trainer.strategy.model_state_dict.return_value = {"weight": torch.zeros(2, 2)}
    trainer.optimizer.state_dict.return_value = {}
    return trainer


# ---------------------------------------------------------------------------
# Base Callback
# ---------------------------------------------------------------------------

class TestCallbackBase:
    def test_all_hooks_are_noop(self):
        cb = Callback()
        t = _make_trainer()
        cb.on_train_start(t)
        cb.on_train_end(t)
        cb.on_epoch_start(t, 1)
        cb.on_epoch_end(t, 1, {})
        cb.on_step_start(t, 0)
        cb.on_step_end(t, 0, {})
        cb.on_validation_start(t)
        cb.on_validation_end(t, {})


# ---------------------------------------------------------------------------
# ModelCheckpoint
# ---------------------------------------------------------------------------

class TestModelCheckpoint:
    def test_saves_on_every_epoch(self, tmp_path):
        ckpt = ModelCheckpoint(dirpath=str(tmp_path), every_n_epochs=1, save_best=False)
        t = _make_trainer()
        ckpt.on_epoch_end(t, epoch=1, metrics={})
        assert len(list(tmp_path.glob("epoch_*.pt"))) == 1

    def test_skips_non_multiple_epoch(self, tmp_path):
        ckpt = ModelCheckpoint(dirpath=str(tmp_path), every_n_epochs=2, save_best=False)
        t = _make_trainer()
        ckpt.on_epoch_end(t, epoch=1, metrics={})
        assert len(list(tmp_path.glob("epoch_*.pt"))) == 0

    def test_saves_on_multiple_epoch(self, tmp_path):
        ckpt = ModelCheckpoint(dirpath=str(tmp_path), every_n_epochs=2, save_best=False)
        t = _make_trainer()
        ckpt.on_epoch_end(t, epoch=2, metrics={})
        assert len(list(tmp_path.glob("epoch_*.pt"))) == 1

    def test_best_tracking_min(self, tmp_path):
        ckpt = ModelCheckpoint(
            dirpath=str(tmp_path), save_best=True, monitor="val_loss", mode="min"
        )
        t = _make_trainer()
        ckpt.on_epoch_end(t, epoch=1, metrics={"val_loss": 0.8})
        assert ckpt._best == pytest.approx(0.8)

        ckpt.on_epoch_end(t, epoch=2, metrics={"val_loss": 0.5})
        assert ckpt._best == pytest.approx(0.5)

        ckpt.on_epoch_end(t, epoch=3, metrics={"val_loss": 0.9})
        assert ckpt._best == pytest.approx(0.5)  # no update

    def test_best_tracking_max(self, tmp_path):
        ckpt = ModelCheckpoint(
            dirpath=str(tmp_path), save_best=True, monitor="val_acc", mode="max"
        )
        t = _make_trainer()
        ckpt.on_epoch_end(t, epoch=1, metrics={"val_acc": 0.7})
        assert ckpt._best == pytest.approx(0.7)

        ckpt.on_epoch_end(t, epoch=2, metrics={"val_acc": 0.9})
        assert ckpt._best == pytest.approx(0.9)

    def test_best_pt_written(self, tmp_path):
        ckpt = ModelCheckpoint(dirpath=str(tmp_path), save_best=True, monitor="val_loss")
        t = _make_trainer()
        ckpt.on_epoch_end(t, epoch=1, metrics={"val_loss": 0.4})
        assert (tmp_path / "best.pt").exists()

    def test_step_checkpoint(self, tmp_path):
        ckpt = ModelCheckpoint(dirpath=str(tmp_path), every_n_steps=5, save_best=False)
        t = _make_trainer()
        ckpt.on_step_end(t, step=5, metrics={})
        assert len(list(tmp_path.glob("step_*.pt"))) == 1

    def test_step_checkpoint_skips_non_multiple(self, tmp_path):
        ckpt = ModelCheckpoint(dirpath=str(tmp_path), every_n_steps=5, save_best=False)
        t = _make_trainer()
        ckpt.on_step_end(t, step=3, metrics={})
        assert len(list(tmp_path.glob("step_*.pt"))) == 0

    def test_rotation_keeps_last_n(self, tmp_path):
        ckpt = ModelCheckpoint(
            dirpath=str(tmp_path), every_n_epochs=1, keep_last_n=2, save_best=False
        )
        t = _make_trainer()
        for epoch in range(1, 6):
            ckpt.on_epoch_end(t, epoch=epoch, metrics={})
        remaining = sorted(tmp_path.glob("epoch_*.pt"))
        assert len(remaining) == 2


# ---------------------------------------------------------------------------
# ProgressCallback
# ---------------------------------------------------------------------------

class TestProgressCallback:
    def test_does_not_raise(self):
        cb = ProgressCallback(log_every_n_steps=1)
        t = _make_trainer()
        cb.on_epoch_start(t, 1)
        cb.on_step_end(t, 1, {"train_loss": 0.5})
        cb.on_epoch_end(t, 1, {"train_loss": 0.5, "val_loss": 0.4})
        cb.on_validation_end(t, {"val_loss": 0.4})

    def test_step_skipped_if_not_multiple(self, capfd):
        cb = ProgressCallback(log_every_n_steps=10)
        t = _make_trainer()
        cb.on_step_end(t, step=3, metrics={"train_loss": 0.1})
        # No output expected for step 3 with log_every=10
        # (logger writes to stdout; just verify no exception raised)
