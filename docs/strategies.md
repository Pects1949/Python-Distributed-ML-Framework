# Training Strategies

A strategy encapsulates everything that differs between DDP and FSDP:
backend initialisation, model wrapping, gradient synchronisation, precision
management, and checkpoint serialisation.

---

## Choosing a strategy

| | DDP | FSDP |
|-|-----|------|
| Model fits on one GPU | Yes | Yes |
| Model exceeds single-GPU memory | No | **Yes** |
| Overhead | Low | Moderate |
| Complexity | Simple | Higher |
| State dict for checkpointing | Straightforward | Gathered to rank 0 |

Start with DDP.  Switch to FSDP only when the model does not fit.

---

## DDPStrategy

```python
from pydist_train.strategies.ddp import DDPStrategy

strategy = DDPStrategy(
    precision="fp32",              # "fp32" | "fp16" | "bf16"
    find_unused_parameters=False,  # set True for models with optional branches
    gradient_as_bucket_view=True,  # reduces peak memory
    backend="nccl",                # "nccl" (GPU) | "gloo" (CPU / testing)
)
```

**How it works**

After each `backward()` call DDP performs an all-reduce across all ranks so
every replica has identical gradients before the optimiser step.  The model
is replicated in full on every GPU.

**Backend fallback**

When `backend="nccl"` and no GPU is available the strategy automatically
falls back to `gloo` so the same code runs on CPU-only CI machines.

---

## FSDPStrategy

```python
from pydist_train.strategies.fsdp import FSDPStrategy

strategy = FSDPStrategy(
    precision="bf16",
    sharding_strategy="full_shard",     # see below
    transformer_layer_cls=None,         # or e.g. nn.TransformerEncoderLayer
    min_num_params=100_000,             # threshold for size-based wrapping
    cpu_offload=False,                  # offload idle params/grads to CPU
    backend="nccl",
)
```

**Sharding strategies**

| Value | Shards | Memory savings |
|-------|--------|----------------|
| `"full_shard"` | params + grads + optimizer state | Maximum |
| `"shard_grad_op"` | grads + optimizer state | Moderate |
| `"no_shard"` | nothing (behaves like DDP) | None |

**Wrapping policies**

FSDP wraps sub-modules independently.  Two policies are supported:

- **Transformer-aware** — set `transformer_layer_cls` to your repeating block
  class (e.g. `GPTBlock`).  Each block becomes an FSDP unit.
- **Size-based** — any sub-module with more than `min_num_params` parameters
  becomes its own FSDP unit.  Used when `transformer_layer_cls` is `None`.

**Checkpointing**

`FSDPStrategy.model_state_dict()` calls `FSDP.state_dict_type(FULL_STATE_DICT)`
which gathers the complete state dict onto rank 0 and offloads it to CPU.
This means checkpoint files are identical to DDP checkpoints and are portable
between strategies.

---

## Writing a custom strategy

Subclass `BaseStrategy` and implement three abstract methods:

```python
from pydist_train.strategies.base import BaseStrategy
import torch.nn as nn

class MyStrategy(BaseStrategy):
    def setup(self, rank: int, world_size: int) -> None:
        # initialise backend, set self._device
        ...

    def wrap_model(self, model: nn.Module) -> nn.Module:
        # move to device, wrap however needed
        return model.to(self._device)

    def teardown(self) -> None:
        # clean up backend
        ...
```

The optional hooks `backward()`, `optimizer_step()`, `clip_grad_norm()`,
`autocast()`, `model_state_dict()`, and `load_model_state_dict()` all have
sensible defaults in `BaseStrategy` that you can override selectively.

---

## Multi-node training

For multi-node runs set `MASTER_ADDR` and `MASTER_PORT` before launching:

```bash
# Node 0
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
  torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 train.py

# Node 1
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
  torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 train.py
```

`init_process_group()` reads these variables via `os.environ.setdefault` so
they can also be set in Python before calling `trainer.fit()`.
