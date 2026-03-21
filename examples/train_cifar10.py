"""ResNet-18 CIFAR-10 training example with DDP and mixed precision.

Usage (single GPU):
    python examples/train_cifar10.py

Usage (4 GPUs via mp.spawn):
    python examples/train_cifar10.py --gpus 4

Usage (4 GPUs via torchrun):
    torchrun --nproc_per_node=4 examples/train_cifar10.py
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, models, transforms
except ImportError:
    sys.exit("torchvision is required: pip install 'pydist-train[vision]'")

from pydist_train import Trainer, TrainConfig, launch
from pydist_train.config import CheckpointConfig, OptimizerConfig, SchedulerConfig
from pydist_train.strategies.ddp import DDPStrategy


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2023, 0.1994, 0.2010)


def get_loaders(data_dir: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    val_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=val_tf)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_fn(rank: int, world_size: int, args: argparse.Namespace) -> None:
    train_dl, val_dl = get_loaders(args.data_dir, args.batch_size)

    use_gpu = torch.cuda.is_available()
    config = TrainConfig(
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        precision="fp16" if use_gpu else "fp32",
        optimizer=OptimizerConfig(
            name="sgd",
            lr=0.1,
            weight_decay=5e-4,
            kwargs={"momentum": 0.9, "nesterov": True},
        ),
        scheduler=SchedulerConfig(name="cosine", warmup_steps=500),
        checkpoint=CheckpointConfig(
            dirpath=args.checkpoint_dir,
            save_best=True,
            monitor="val_loss",
            keep_last_n=5,
        ),
    )
    strategy = DDPStrategy(
        precision="fp16" if use_gpu else "fp32",
        backend="nccl" if use_gpu else "gloo",
    )
    trainer = Trainer(
        model=build_model(),
        train_loader=train_dl,
        val_loader=val_dl,
        config=config,
        strategy=strategy,
    )
    trainer.fit(rank=rank, world_size=world_size)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10 distributed training")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/cifar10")
    args = parser.parse_args()

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        train_fn(rank, world_size, args)
    elif args.gpus > 1:
        launch(lambda rank, ws: train_fn(rank, ws, args), world_size=args.gpus)
    else:
        train_fn(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
