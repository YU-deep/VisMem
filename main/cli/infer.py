from __future__ import annotations
import argparse
from PIL import Image
import torch

from main.utils.logging import get_logger
from main.utils.misc import to_torch_dtype
from main.utils.qwen_vl import load_qwen25vl
from main.model.model import VisMemModel
from main.cli.common import load_yaml, build_vismem_config

logger = get_logger("main.infer")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vismem_qwen25vl7b.yaml")
    ap.add_argument("--model_name_or_path", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--enable_vismem", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    args = ap.parse_args()

    cfg_dict = load_yaml(args.config)
    if args.model_name_or_path is not None:
        cfg_dict["model"]["model_name_or_path"] = args.model_name_or_path

    viscfg = build_vismem_config(cfg_dict)
    model_name = cfg_dict["model"]["model_name_or_path"]
    dtype = to_torch_dtype(cfg_dict["model"].get("torch_dtype","bfloat16"))
    device_map = cfg_dict["model"].get("device_map","auto")
    trust = bool(cfg_dict["model"].get("trust_remote_code", True))

    base_model, tokenizer, processor = load_qwen25vl(model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=trust)
    vismem = VisMemModel(base_model, tokenizer, processor, viscfg)
    vismem.eval()

    img = Image.open(args.image).convert("RGB") if args.image else None
    out = vismem.generate(images=[img] if img else [None], prompts=[args.prompt], max_new_tokens=args.max_new_tokens,
                          temperature=args.temperature, top_p=args.top_p, enable_vismem=args.enable_vismem)
    print(out[0])

if __name__ == "__main__":
    main()
