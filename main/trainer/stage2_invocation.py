from __future__ import annotations
from typing import List, Optional, Dict, Any, Callable
import torch
import torch.nn as nn

from main.trainer.rewards import exact_match_reward

def compute_penalties(reward_main: float, reward_rev: Optional[float], reward_mean: float) -> Dict[str, float]:
    ptype = 0.0
    if reward_rev is not None:
        ptype = max(0.0, reward_rev - reward_main)
    pneg = max(0.0, reward_mean - reward_main)
    return {"ptype": ptype, "pneg": pneg}

def reinforce_loss(logp: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    return -(advantage.detach() * logp).mean()
