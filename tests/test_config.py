"""Tests for pydist_train.config."""
import pytest
from pydist_train.config import (
    CheckpointConfig,
    OptimizerConfig,
    Precision,
    SchedulerConfig,
    Strategy,
    TrainConfig,
)


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.max_epochs == 10
        assert cfg.strategy == Strategy.DDP
        assert cfg.precision == Precision.FP32
        assert cfg.gradient_accumulation_steps == 1
        assert cfg.seed == 42
        assert cfg.devices == 1
        assert cfg.num_nodes == 1

    def test_strategy_coercion_from_string(self):
        assert TrainConfig(strategy="ddp").strategy == Strategy.DDP
        assert TrainConfig(strategy="fsdp").strategy == Strategy.FSDP

    def test_precision_coercion_from_string(self):
        assert TrainConfig(precision="bf16").precision == Precision.BF16
        assert TrainConfig(precision="fp16").precision == Precision.FP16

    def test_optimizer_coercion_from_dict(self):
        cfg = TrainConfig(optimizer={"name": "sgd", "lr": 0.01})
        assert isinstance(cfg.optimizer, OptimizerConfig)
        assert cfg.optimizer.name == "sgd"

    def test_scheduler_coercion_from_dict(self):
        cfg = TrainConfig(scheduler={"name": "step", "warmup_steps": 50})
        assert isinstance(cfg.scheduler, SchedulerConfig)
        assert cfg.scheduler.warmup_steps == 50

    def test_checkpoint_coercion_from_dict(self):
        cfg = TrainConfig(checkpoint={"dirpath": "/tmp/ckpts"})
        assert isinstance(cfg.checkpoint, CheckpointConfig)
        assert cfg.checkpoint.dirpath == "/tmp/ckpts"

    def test_invalid_gradient_accumulation(self):
        with pytest.raises(ValueError, match="gradient_accumulation_steps"):
            TrainConfig(gradient_accumulation_steps=0)

    def test_invalid_devices(self):
        with pytest.raises(ValueError, match="devices"):
            TrainConfig(devices=0)

    def test_invalid_num_nodes(self):
        with pytest.raises(ValueError, match="num_nodes"):
            TrainConfig(num_nodes=0)


class TestOptimizerConfig:
    def test_defaults(self):
        opt = OptimizerConfig()
        assert opt.name == "adamw"
        assert opt.lr == pytest.approx(1e-3)
        assert opt.weight_decay == pytest.approx(1e-2)

    def test_custom_values(self):
        opt = OptimizerConfig(name="sgd", lr=0.1, weight_decay=5e-4)
        assert opt.name == "sgd"
        assert opt.lr == pytest.approx(0.1)


class TestCheckpointConfig:
    def test_defaults(self):
        cfg = CheckpointConfig()
        assert cfg.dirpath == "./checkpoints"
        assert cfg.keep_last_n == 3
        assert cfg.mode == "min"

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            CheckpointConfig(mode="avg")


class TestSchedulerConfig:
    def test_defaults(self):
        cfg = SchedulerConfig()
        assert cfg.name == "cosine"
        assert cfg.warmup_steps == 0
