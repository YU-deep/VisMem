from __future__ import annotations
import argparse
import yaml
from dataclasses import dataclass
from typing import Any, Dict

from main.model.configuration_vismem import VisMemConfig, QueryBuilderConfig, LoRAConfig

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_vismem_config(cfg_dict: Dict[str, Any]) -> VisMemConfig:
    v = cfg_dict.get("main", {})
    qb = v.get("query_builder", {})
    lora = v.get("lora", {})
    cfg = VisMemConfig(
        short_invoke_token=v.get("short_invoke_token","<ms_I>"),
        short_end_token=v.get("short_end_token","<ms_E>"),
        long_invoke_token=v.get("long_invoke_token","<ml_I>"),
        long_end_token=v.get("long_end_token","<ml_E>"),
        query_len=int(v.get("query_len",8)),
        short_mem_len=int(v.get("short_mem_len",8)),
        long_mem_len=int(v.get("long_mem_len",16)),
        former_backend=str(v.get("former_backend","lora_llm")),
        max_prompt_hidden=int(v.get("max_prompt_hidden",1024)),
        query_builder=QueryBuilderConfig(
            num_layers=int(qb.get("num_layers",2)),
            num_heads=int(qb.get("num_heads",8)),
            dropout=float(qb.get("dropout",0.0)),
            ff_mult=int(qb.get("ff_mult",4)),
        ),
        lora=LoRAConfig(
            r=int(lora.get("r",16)),
            alpha=int(lora.get("alpha",32)),
            dropout=float(lora.get("dropout",0.05)),
            target_modules = list(lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
            short_target_modules = lora.get("short_target_modules", None),
            long_target_modules = lora.get("long_target_modules", None),
        ),
    )
    return cfg
