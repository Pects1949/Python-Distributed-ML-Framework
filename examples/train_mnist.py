"""Single- and multi-GPU MNIST training example.

Usage (single GPU / CPU):
    python examples/train_mnist.py

Usage (multi-GPU):
    python examples/train_mnist.py --gpus 4
    # or via torchrun:
    torchrun --nproc_per_node=4 examples/train_mnist.py
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms
except ImportError:
    sys.exit("torchvision is required: pip install 'pydist-train[vision]'")

from pydist_train import Trainer, TrainConfig, launch
from pydist_train.config import CheckpointConfig, OptimizerConfig, SchedulerConfig
from pydist_train.strategies.ddp import DDPStrategy


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MnistNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_loaders(data_dir: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    val_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True)
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# Training function (called once per process)
# ---------------------------------------------------------------------------

def train_fn(rank: int, world_size: int, args: argparse.Namespace) -> None:
    train_dl, val_dl = get_loaders(args.data_dir, args.batch_size)

    config = TrainConfig(
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        optimizer=OptimizerConfig(name="adamw", lr=1e-3),
        scheduler=SchedulerConfig(name="cosine", warmup_steps=200),
        checkpoint=CheckpointConfig(
            dirpath=args.checkpoint_dir,
            save_best=True,
            monitor="val_loss",
        ),
    )
    strategy = DDPStrategy(
        backend="nccl" if torch.cuda.is_available() else "gloo",
    )
    trainer = Trainer(
        model=MnistNet(),
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
    parser = argparse.ArgumentParser(description="MNIST distributed training example")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/mnist")
    args = parser.parse_args()

    # torchrun sets RANK/WORLD_SIZE; fall back to mp.spawn for manual launch
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
