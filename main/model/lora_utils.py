from __future__ import annotations
from typing import Any, Dict, Optional, List
import torch
import torch.nn as nn

def is_peft_available() -> bool:
    try:
        import peft  # noqa
        return True
    except Exception:
        return False

def make_lora_adapters(base_model, adapter_name: str, r: int, alpha: int, dropout: float,
                      target_modules: List[str]):
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    peft_model = get_peft_model(base_model, lora_cfg, adapter_name=adapter_name)
    return peft_model

def set_active_adapter(model, name: str):
    # Compatible with PeftModel and some multi-adapter wrappers
    if hasattr(model, "set_adapter"):
        model.set_adapter(name)
    elif hasattr(model, "active_adapter"):
        model.active_adapter = name
    else:
        raise AttributeError("Model doesn't support adapter switching.")
