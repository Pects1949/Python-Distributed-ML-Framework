"""Integration tests for the Trainer class."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pydist_train.callbacks.base import Callback
from pydist_train.config import CheckpointConfig, OptimizerConfig, SchedulerConfig, TrainConfig
from pydist_train.strategies.ddp import DDPStrategy
from pydist_train.trainer import Trainer, _build_optimizer, _build_scheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> TrainConfig:
    defaults = dict(
        max_epochs=2,
        log_every_n_steps=1,
        checkpoint=CheckpointConfig(dirpath="/tmp/pydist_test_ckpts", save_best=False),
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _gloo_strategy(**kwargs) -> DDPStrategy:
    return DDPStrategy(backend="gloo", **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainerSingleProcess:
    def test_basic_run(self, tiny_model, train_loader, tmp_path):
        config = _make_config(checkpoint=CheckpointConfig(dirpath=str(tmp_path), save_best=False))
        trainer = Trainer(
            model=tiny_model,
            train_loader=train_loader,
            config=config,
            strategy=_gloo_strategy(),
        )
        trainer.fit(rank=0, world_size=1)
        assert trainer.current_epoch == 2
        assert trainer.global_step > 0

    def test_with_validation(self, tiny_model, train_loader, val_loader, tmp_path):
        config = _make_config(
            max_epochs=2,
            val_every_n_epochs=1,
            checkpoint=CheckpointConfig(dirpath=str(tmp_path), save_best=True, monitor="val_loss"),
        )
        trainer = Trainer(
            model=tiny_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            strategy=_gloo_strategy(),
        )
        trainer.fit(rank=0, world_size=1)
        assert (tmp_path / "best.pt").exists()

    def test_max_steps_stops_early(self, tiny_model, train_loader, tmp_path):
        config = _make_config(
            max_epochs=100,
            max_steps=3,
            checkpoint=CheckpointConfig(dirpath=str(tmp_path), save_best=False),
        )
        trainer = Trainer(
            model=tiny_model,
            train_loader=train_loader,
            config=config,
            strategy=_gloo_strategy(),
        )
        trainer.fit(rank=0, world_size=1)
        assert trainer.global_step == 3

    def test_gradient_accumulation(self, tiny_model, train_loader, tmp_path):
        # 64 samples / batch_size=16 = 4 batches → with accum=2 → 2 optimizer steps
        config = _make_config(
            max_epochs=1,
            gradient_accumulation_steps=2,
            checkpoint=CheckpointConfig(dirpath=str(tmp_path), save_best=False),
        )
        trainer = Trainer(
            model=tiny_model,
            train_loader=train_loader,
            config=config,
            strategy=_gloo_strategy(),
        )
        trainer.fit(rank=0, world_size=1)
        assert trainer.global_step == 2

    def test_custom_loss_fn(self, tiny_model, train_loader, tmp_path):
        def mse_loss(model, batch):
            x, _ = batch
            return torch.nn.functional.mse_loss(model(x), torch.zeros(x.size(0), 4))

        config = _make_config(
            max_epochs=1,
            checkpoint=CheckpointConfig(dirpath=str(tmp_path), save_best=False),
        )
        trainer = Trainer(
            model=tiny_model,
            train_loader=train_loader,
            config=config,
            strategy=_gloo_strategy(),
            loss_fn=mse_loss,
        )
        trainer.fit(rank=0, world_size=1)
        assert trainer.global_step > 0

    def test_callbacks_are_called(self, tiny_model, train_loader, tmp_path):
        events: list[str] = []

        class _Recorder(Callback):
            def on_train_start(self, trainer):
                events.append("train_start")

            def on_epoch_start(self, trainer, epoch):
                events.append(f"epoch_start:{epoch}")

            def on_epoch_end(self, trainer, epoch, metrics):
                events.append(f"epoch_end:{epoch}")

            def on_train_end(self, trainer):
                events.append("train_end")

        config = _make_config(
            max_epochs=2,
            checkpoint=CheckpointConfig(dirpath=str(tmp_path), save_best=False),
        )
        trainer = Trainer(
            model=tiny_model,
            train_loader=train_loader,
            config=config,
            strategy=_gloo_strategy(),
            callbacks=[_Recorder()],
        )
        trainer.fit(rank=0, world_size=1)
        assert events[0] == "train_start"
        assert "epoch_start:1" in events
        assert "epoch_start:2" in events
        assert "epoch_end:1" in events
        assert "epoch_end:2" in events
        assert events[-1] == "train_end"

    def test_sgd_optimizer(self, tiny_model, train_loader, tmp_path):
        config = _make_config(
            max_epochs=1,
            optimizer=OptimizerConfig(name="sgd", lr=0.01, kwargs={"momentum": 0.9}),
            checkpoint=CheckpointConfig(dirpath=str(tmp_path), save_best=False),
        )
        trainer = Trainer(
            model=tiny_model,
            train_loader=train_loader,
            config=config,
            strategy=_gloo_strategy(),
        )
        trainer.fit(rank=0, world_size=1)
        assert isinstance(trainer.optimizer, torch.optim.SGD)

    def test_cosine_scheduler(self, tiny_model, train_loader, tmp_path):
        config = _make_config(
            max_epochs=2,
            scheduler=SchedulerConfig(name="cosine"),
            checkpoint=CheckpointConfig(dirpath=str(tmp_path), save_best=False),
        )
        trainer = Trainer(
            model=tiny_model,
            train_loader=train_loader,
            config=config,
            strategy=_gloo_strategy(),
        )
        trainer.fit(rank=0, world_size=1)
        assert trainer.scheduler is not None


# ---------------------------------------------------------------------------
# Optimizer / scheduler factory unit tests
# ---------------------------------------------------------------------------

class TestOptimizerFactory:
    def _model(self):
        return nn.Linear(4, 2)

    def test_adamw(self):
        opt = _build_optimizer(OptimizerConfig(name="adamw", lr=1e-3), self._model())
        assert isinstance(opt, torch.optim.AdamW)

    def test_adam(self):
        opt = _build_optimizer(OptimizerConfig(name="adam"), self._model())
        assert isinstance(opt, torch.optim.Adam)

    def test_sgd(self):
        opt = _build_optimizer(OptimizerConfig(name="sgd"), self._model())
        assert isinstance(opt, torch.optim.SGD)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            _build_optimizer(OptimizerConfig(name="rmsprop"), self._model())


class TestSchedulerFactory:
    def _optimizer(self):
        return torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)

    def test_none_returns_none(self):
        assert _build_scheduler(None, self._optimizer(), 100) is None

    def test_cosine(self):
        sched = _build_scheduler(SchedulerConfig(name="cosine"), self._optimizer(), 100)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_step(self):
        sched = _build_scheduler(
            SchedulerConfig(name="step", kwargs={"step_size": 5}), self._optimizer(), 100
        )
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scheduler"):
            _build_scheduler(SchedulerConfig(name="polynomial"), self._optimizer(), 100)
