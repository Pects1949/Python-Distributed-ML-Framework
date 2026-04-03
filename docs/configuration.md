# Configuration Reference

All training behaviour is controlled through a small set of dataclasses in
`pydist_train.config`.  Every field has a sensible default so you only need to
set what differs from the baseline.

---

## TrainConfig

```python
from pydist_train import TrainConfig

config = TrainConfig(
    max_epochs=10,
    max_steps=None,
    gradient_clip_val=1.0,
    gradient_accumulation_steps=1,
    log_every_n_steps=10,
    val_every_n_epochs=1,
    strategy="ddp",        # "ddp" | "fsdp"  (or Strategy enum)
    precision="fp32",      # "fp32" | "fp16" | "bf16"
    devices=1,
    num_nodes=1,
    seed=42,
    optimizer=...,         # OptimizerConfig or dict
    scheduler=None,        # SchedulerConfig, dict, or None
    checkpoint=...,        # CheckpointConfig or dict
)
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `max_epochs` | `int` | `10` | Total epochs to run |
| `max_steps` | `int \| None` | `None` | Stop after this many optimiser steps |
| `gradient_clip_val` | `float \| None` | `1.0` | Max gradient norm; `None` disables clipping |
| `gradient_accumulation_steps` | `int` | `1` | Accumulate gradients across N mini-batches |
| `log_every_n_steps` | `int` | `10` | ProgressCallback log frequency |
| `val_every_n_epochs` | `int` | `1` | Run validation every N epochs |
| `strategy` | `Strategy` | `Strategy.DDP` | Distributed strategy |
| `precision` | `Precision` | `Precision.FP32` | Training precision |
| `devices` | `int` | `1` | GPUs per node (informational) |
| `num_nodes` | `int` | `1` | Node count (informational) |
| `seed` | `int` | `42` | Base random seed (offset by rank) |

### String and dict coercion

`strategy`, `precision`, `optimizer`, `scheduler`, and `checkpoint` all accept
plain strings or dicts in addition to their proper types:

```python
config = TrainConfig(
    strategy="fsdp",
    precision="bf16",
    optimizer={"name": "sgd", "lr": 0.1},
    scheduler={"name": "cosine", "warmup_steps": 500},
    checkpoint={"dirpath": "/mnt/runs/exp1", "keep_last_n": 5},
)
```

---

## OptimizerConfig

```python
from pydist_train.config import OptimizerConfig

OptimizerConfig(
    name="adamw",      # "adam" | "adamw" | "sgd"
    lr=1e-3,
    weight_decay=1e-2,
    kwargs={},         # passed directly to the torch optimizer constructor
)
```

Pass extra keyword arguments via `kwargs`:

```python
OptimizerConfig(name="sgd", lr=0.1, kwargs={"momentum": 0.9, "nesterov": True})
```

---

## SchedulerConfig

```python
from pydist_train.config import SchedulerConfig

SchedulerConfig(
    name="cosine",     # "cosine" | "step" | "linear"
    warmup_steps=0,    # linear warmup before the main schedule
    kwargs={},         # extra args for the torch scheduler
)
```

With warmup the scheduler is a `SequentialLR` of a `LinearLR` warmup phase
followed by the selected schedule.

```python
# Step LR: multiply LR by 0.1 every 30 epochs
SchedulerConfig(name="step", kwargs={"step_size": 30, "gamma": 0.1})
```

---

## CheckpointConfig

```python
from pydist_train.config import CheckpointConfig

CheckpointConfig(
    dirpath="./checkpoints",
    every_n_epochs=1,
    every_n_steps=None,
    keep_last_n=3,
    save_best=True,
    monitor="val_loss",
    mode="min",         # "min" | "max"
)
```

| Field | Notes |
|-------|-------|
| `dirpath` | Created automatically if it does not exist |
| `every_n_epochs` | Save a `epoch_NNNN.pt` file every N epochs |
| `every_n_steps` | Save a `step_NNNNNNNN.pt` file every N global steps |
| `keep_last_n` | Older epoch/step checkpoints beyond this count are deleted |
| `save_best` | `best.pt` is always preserved regardless of `keep_last_n` |
| `monitor` | Must be a key in the epoch metrics dict (e.g. `"val_loss"`) |
| `mode` | `"min"`: lower is better; `"max"`: higher is better |

Checkpoint files are dictionaries with keys `epoch`, `global_step`, `model`,
and `optimizer`.  Load them with `pydist_train.utils.load_checkpoint`.
