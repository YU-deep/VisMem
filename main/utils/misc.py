from __future__ import annotations
import os
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_torch_dtype(name: str):
    name = str(name).lower()
    if name in ("fp16","float16","half"):
        return torch.float16
    if name in ("bf16","bfloat16"):
        return torch.bfloat16
    if name in ("fp32","float32","full"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path
