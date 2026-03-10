# pydist-train

A simple, batteries-included distributed training toolkit for PyTorch.

---

## Features

- **DDP and FSDP** — out-of-the-box support for `DistributedDataParallel` and
  `FullyShardedDataParallel`
- **Dataclass-based configuration** — no YAML magic; plain Python
- **Callback system** — hook into any training lifecycle event
- **Automatic checkpointing** — save the last *N* checkpoints and always keep
  the best
- **Mixed precision** — fp16 and bf16 with a single config flag
- **Gradient accumulation** — train with effective batch sizes larger than a
  single GPU can hold
- **Scheduler support** — cosine, step, and linear LR schedules with optional
  warmup
- **Single-file launch** — works with `torch.multiprocessing.spawn` or
  `torchrun`

---

## Installation

```bash
pip install pydist-train
```

To also pull in torchvision for the bundled examples:

```bash
pip install "pydist-train[vision]"
```

Development install with linting and test dependencies:

```bash
git clone https://github.com/hesam-mohseni/pydist-train
cd pydist-train
pip install -e ".[dev,vision]"
```

---

## Quick start

### Single GPU (or CPU)

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pydist_train import Trainer, TrainConfig

model = nn.Linear(16, 4)
ds = TensorDataset(torch.randn(256, 16), torch.randint(0, 4, (256,)))
loader = DataLoader(ds, batch_size=32, shuffle=True)

trainer = Trainer(model=model, train_loader=loader, config=TrainConfig(max_epochs=5))
trainer.fit()
```

### Multi-GPU with DDP

```python
from pydist_train import Trainer, TrainConfig, launch
from pydist_train.strategies.ddp import DDPStrategy

def train_fn(rank: int, world_size: int) -> None:
    trainer = Trainer(
        model=build_model(),
        train_loader=build_loader(),
        config=TrainConfig(max_epochs=10),
        strategy=DDPStrategy(),
    )
    trainer.fit(rank=rank, world_size=world_size)

launch(train_fn, world_size=4)
```

### Multi-GPU with torchrun

```bash
torchrun --nproc_per_node=4 train.py
```

```python
# train.py
import os
from pydist_train import Trainer, TrainConfig
from pydist_train.strategies.ddp import DDPStrategy

rank       = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

trainer = Trainer(
    model=build_model(),
    train_loader=build_loader(),
    config=TrainConfig(max_epochs=10),
    strategy=DDPStrategy(),
)
trainer.fit(rank=rank, world_size=world_size)
```

### Large models with FSDP

```python
from pydist_train import Trainer, TrainConfig, launch
from pydist_train.strategies.fsdp import FSDPStrategy

def train_fn(rank: int, world_size: int) -> None:
    trainer = Trainer(
        model=build_large_model(),
        train_loader=build_loader(),
        config=TrainConfig(max_epochs=5, precision="bf16"),
        strategy=FSDPStrategy(
            precision="bf16",
            sharding_strategy="full_shard",
            transformer_layer_cls=MyTransformerLayer,
        ),
    )
    trainer.fit(rank=rank, world_size=world_size)

launch(train_fn, world_size=8)
```

---

## Configuration reference

```python
from pydist_train import TrainConfig
from pydist_train.config import (
    CheckpointConfig,
    OptimizerConfig,
    SchedulerConfig,
    Strategy,
    Precision,
)

config = TrainConfig(
    # ---- Duration ----
    max_epochs=10,
    max_steps=None,            # stops early if set

    # ---- Optimisation ----
    gradient_clip_val=1.0,
    gradient_accumulation_steps=1,

    # ---- Logging ----
    log_every_n_steps=10,

    # ---- Validation ----
    val_every_n_epochs=1,

    # ---- Distribution ----
    strategy=Strategy.DDP,     # or "ddp" / "fsdp"
    precision=Precision.FP32,  # or "fp16" / "bf16"
    devices=1,
    num_nodes=1,

    # ---- Reproducibility ----
    seed=42,

    # ---- Sub-configs ----
    optimizer=OptimizerConfig(
        name="adamw",          # adam | adamw | sgd
        lr=1e-3,
        weight_decay=1e-2,
    ),
    scheduler=SchedulerConfig(
        name="cosine",         # cosine | step | linear
        warmup_steps=500,
    ),
    checkpoint=CheckpointConfig(
        dirpath="./checkpoints",
        every_n_epochs=1,
        every_n_steps=None,
        keep_last_n=3,
        save_best=True,
        monitor="val_loss",
        mode="min",            # min | max
    ),
)
```

All string-valued fields accept their enum equivalents or plain strings, and
dict-valued nested configs are auto-coerced:

```python
config = TrainConfig(
    strategy="fsdp",
    precision="bf16",
    optimizer={"name": "sgd", "lr": 0.1, "kwargs": {"momentum": 0.9}},
)
```

---

## Strategies

| Strategy | Class | Best for |
|----------|-------|----------|
| DDP | `DDPStrategy` | models that fit on a single GPU |
| FSDP | `FSDPStrategy` | models too large for a single GPU |

### DDPStrategy options

```python
from pydist_train.strategies.ddp import DDPStrategy

DDPStrategy(
    precision="fp32",              # fp32 | fp16 | bf16
    find_unused_parameters=False,
    gradient_as_bucket_view=True,
    backend="nccl",                # nccl | gloo
)
```

### FSDPStrategy options

```python
from pydist_train.strategies.fsdp import FSDPStrategy

FSDPStrategy(
    precision="bf16",
    sharding_strategy="full_shard",     # full_shard | shard_grad_op | no_shard
    transformer_layer_cls=MyLayer,      # enables transformer-aware wrapping
    min_num_params=100_000,             # size-based wrapping threshold
    cpu_offload=False,
    backend="nccl",
)
```

---

## Callbacks

```python
from pydist_train.callbacks import Callback

class MyCallback(Callback):
    def on_train_start(self, trainer):
        print("Training starts!")

    def on_epoch_end(self, trainer, epoch, metrics):
        print(f"Epoch {epoch}: {metrics}")

    def on_validation_end(self, trainer, metrics):
        if metrics.get("val_loss", 1.0) < 0.1:
            trainer.config.max_epochs = trainer.current_epoch  # early-stop trick

trainer = Trainer(..., callbacks=[MyCallback()])
```

Built-in callbacks registered automatically:

| Callback | Purpose |
|----------|---------|
| `ProgressCallback` | Logs timing and metrics to stdout |
| `ModelCheckpoint` | Saves epoch/step checkpoints and keeps the best |

---

## Custom loss function

The default loss is cross-entropy applied to the first two elements of each
batch.  Override with any callable of the form
`(model, batch) -> scalar_tensor`:

```python
def contrastive_loss(model, batch):
    anchors, positives, negatives = batch
    fa, fp, fn = model(anchors), model(positives), model(negatives)
    return torch.nn.functional.triplet_margin_loss(fa, fp, fn)

trainer = Trainer(..., loss_fn=contrastive_loss)
```

---

## Resuming from a checkpoint

```python
from pydist_train.utils import load_checkpoint

state = load_checkpoint("checkpoints/best.pt")
model.load_state_dict(state["model"])
optimizer.load_state_dict(state["optimizer"])
start_epoch = state["epoch"] + 1
```

---

## Running the test suite

```bash
pip install -e ".[dev]"
pytest
```

---

## License

MIT
