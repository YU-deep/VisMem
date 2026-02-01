from __future__ import annotations
from typing import List, Dict, Any, Optional
from PIL import Image
import torch

from .jsonl_dataset import Sample

def load_image(path: Optional[str]) -> Optional[Image.Image]:
    if path is None:
        return None
    return Image.open(path).convert("RGB")

def collate_samples(samples: List[Sample]) -> Dict[str, Any]:
    ids = [s.id for s in samples]
    images = [load_image(s.image) for s in samples]
    prompts = [s.prompt for s in samples]
    answers = [s.answer for s in samples]
    return {"ids": ids, "images": images, "prompts": prompts, "answers": answers}
