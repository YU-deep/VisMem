from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from torch.utils.data import Dataset

@dataclass
class Sample:
    id: str
    image: Optional[str]
    prompt: str
    answer: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class JsonlVLDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items: List[Sample] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.items.append(
                    Sample(
                        id=str(obj.get("id", len(self.items))),
                        image=obj.get("image", None),
                        prompt=obj["prompt"],
                        answer=obj.get("answer", None),
                        meta={k:v for k,v in obj.items() if k not in ("id","image","prompt","answer")}
                    )
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Sample:
        return self.items[idx]
