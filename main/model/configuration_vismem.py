from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class QueryBuilderConfig:
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.0
    ff_mult: int = 4

@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj","k_proj","v_proj","o_proj"])
    short_target_modules: List[str] | None = None
    long_target_modules: List[str] | None = None

@dataclass
class VisMemConfig:
    short_invoke_token: str = "<ms_I>"
    short_end_token: str = "<ms_E>"
    long_invoke_token: str = "<ml_I>"
    long_end_token: str = "<ml_E>"

    query_len: int = 8
    short_mem_len: int = 8
    long_mem_len: int = 16

    query_builder: QueryBuilderConfig = field(default_factory=QueryBuilderConfig)

    former_backend: str = "lora_llm"   # lora_llm | tiny_transformer
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    max_prompt_hidden: int = 1024   # cap to avoid huge query inputs
