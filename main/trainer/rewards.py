from __future__ import annotations
from typing import List, Optional
import re

def exact_match_reward(preds: List[str], refs: List[Optional[str]]) -> List[float]:
    out = []
    for p, r in zip(preds, refs):
        if r is None:
            out.append(0.0)
            continue
        p_norm = re.sub(r"\s+", " ", p.strip().lower())
        r_norm = re.sub(r"\s+", " ", r.strip().lower())
        out.append(1.0 if p_norm == r_norm else 0.0)
    return out

def substring_reward(preds: List[str], refs: List[Optional[str]]) -> List[float]:
    out = []
    for p, r in zip(preds, refs):
        if r is None:
            out.append(0.0)
            continue
        p_norm = p.lower()
        r_norm = r.lower()
        out.append(1.0 if r_norm in p_norm else 0.0)
    return out
