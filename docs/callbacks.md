# Callbacks

Callbacks let you inject arbitrary logic at every point in the training
lifecycle without modifying the `Trainer`.

---

## Lifecycle hooks

```
fit() called
│
├─ on_train_start(trainer)
│
├─ for each epoch:
│   ├─ on_epoch_start(trainer, epoch)
│   ├─ for each optimisation step:
│   │   ├─ on_step_start(trainer, step)
│   │   └─ on_step_end(trainer, step, metrics)
│   ├─ on_validation_start(trainer)        ← only when val_loader is set
│   ├─ on_validation_end(trainer, metrics) ← only when val_loader is set
│   └─ on_epoch_end(trainer, epoch, metrics)
│
└─ on_train_end(trainer)                   ← runs even if an exception occurs
```

The `metrics` dict passed to `on_epoch_end` merges training and validation
metrics, e.g. `{"train_loss": 0.45, "val_loss": 0.38}`.

---

## Writing a callback

Subclass `Callback` and override only the hooks you need:

```python
from pydist_train.callbacks import Callback

class EarlyStoppingCallback(Callback):
    def __init__(self, monitor: str = "val_loss", patience: int = 5) -> None:
        self.monitor = monitor
        self.patience = patience
        self._best = float("inf")
        self._wait = 0

    def on_epoch_end(self, trainer, epoch, metrics):
        value = metrics.get(self.monitor, float("inf"))
        if value < self._best:
            self._best = value
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                # Force the loop to stop after this epoch
                trainer.config.max_epochs = epoch
```

Register it when constructing `Trainer`:

```python
trainer = Trainer(..., callbacks=[EarlyStoppingCallback(patience=10)])
```

---

## Trainer state available in hooks

| Attribute | Type | Description |
|-----------|------|-------------|
| `trainer.model` | `nn.Module` | The wrapped (DDP/FSDP) model |
| `trainer.optimizer` | `Optimizer` | Active optimizer |
| `trainer.scheduler` | `LRScheduler \| None` | Active scheduler if any |
| `trainer.current_epoch` | `int` | Current epoch number (1-indexed) |
| `trainer.global_step` | `int` | Total optimiser steps taken |
| `trainer.config` | `TrainConfig` | Full training configuration |
| `trainer.strategy` | `BaseStrategy` | Active strategy |

---

## Built-in callbacks

### ProgressCallback

Registered automatically.  Logs epoch timing and metrics to stdout.

```python
from pydist_train.callbacks import ProgressCallback

ProgressCallback(log_every_n_steps=10)
```

Output is suppressed on non-rank-0 processes.

### ModelCheckpoint

Registered automatically.  Saves epoch and step checkpoints, rotates old
files, and maintains a separate `best.pt`.

```python
from pydist_train.callbacks import ModelCheckpoint

ModelCheckpoint(
    dirpath="./checkpoints",
    every_n_epochs=1,
    every_n_steps=None,
    keep_last_n=3,
    save_best=True,
    monitor="val_loss",
    mode="min",
)
```

The default instance is configured from `TrainConfig.checkpoint`.  To
customise it, pass your own `ModelCheckpoint` in the `callbacks` list — the
built-in one will still be present, so set `save_best=False` on one of them
to avoid double-saving.

---

## Execution order

Multiple callbacks are executed in list order.  Built-in callbacks
(`ProgressCallback`, `ModelCheckpoint`) run first, followed by any callbacks
passed to `Trainer(callbacks=[...])`.

---

## Distributed considerations

Each process runs all callbacks independently.  Inside a callback,
`is_main_process()` returns `True` only on rank 0:

```python
from pydist_train.utils import is_main_process

class WandbCallback(Callback):
    def on_epoch_end(self, trainer, epoch, metrics):
        if is_main_process():
            wandb.log(metrics, step=trainer.global_step)
```

`save_checkpoint()` already guards itself to rank 0, so checkpoint callbacks
do not need an extra `is_main_process()` check.
