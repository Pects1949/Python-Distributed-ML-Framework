# Python Distributed ML Training Framework

A Python framework for distributed machine learning training, leveraging PyTorch Distributed and other libraries.

## Overview

This framework provides a flexible and efficient way to train large-scale machine learning models across multiple GPUs and machines using Python. It builds upon the robust capabilities of PyTorch Distributed, offering abstractions and utilities to simplify the development of distributed training pipelines.

## Features

*   **PyTorch Distributed Integration:** Seamless integration with `torch.distributed` for data parallelism and model parallelism.
*   **Distributed Data Loading:** Efficient data loading strategies for distributed environments.
*   **Gradient Accumulation:** Support for training with larger effective batch sizes.
*   **Mixed Precision Training:** Leverage `torch.cuda.amp` for faster training and reduced memory usage.
*   **Configuration Management:** Easy-to-use configuration system for managing distributed training parameters.

## Installation

```bash
git clone https://github.com/Pects1949/Python-Distributed-ML-Framework.git
cd Python-Distributed-ML-Framework
pip install -r requirements.txt
```

## Usage

### Example: Distributed Training with PyTorch DDP

```python
# train_distributed.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for _ in range(10):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10).to(rank))
        labels = torch.randn(20, 5).to(rank)
        loss_fn(outputs, labels).backward()
        optimizer.step()

    cleanup()

if __name__ == "__main__":
    world_size = 2 # Example: run with 2 GPUs
    # This part would typically be launched using torch.distributed.launch or torchrun
    # For demonstration, we simulate it with multiprocessing
    import torch.multiprocessing as mp
    mp.spawn(demo_basic, args=(world_size,), nprocs=world_size, join=True)

    print("Distributed training simulation complete!")
```

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for more details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
