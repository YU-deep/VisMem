from __future__ import annotations
import argparse
from tqdm import tqdm
from PIL import Image

from main.utils.logging import get_logger
from main.utils.misc import to_torch_dtype
from main.utils.qwen_vl import load_qwen25vl
from main.model.model import VisMemModel
from main.data.jsonl_dataset import JsonlVLDataset
from main.data.collate import collate_samples
from main.trainer.rewards import exact_match_reward, substring_reward
from main.cli.common import load_yaml, build_vismem_config

logger = get_logger("main.eval")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vismem_qwen25vl7b.yaml")
    ap.add_argument("--model_name_or_path", default=None)
    ap.add_argument("--ckpt", default=None, help="folder with main.pt")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--enable_vismem", action="store_true")
    ap.add_argument("--metric", choices=["exact","substr"], default="substr")
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
    if args.ckpt is not None:
        import os, torch
        state = torch.load(os.path.join(args.ckpt, "main.pt"), map_location="cpu")
        vismem.load_state_dict(state["vismem_state"], strict=False)
    vismem.eval()

    ds = JsonlVLDataset(args.jsonl)
    preds, refs = [], []
    for i in tqdm(range(len(ds))):
        batch = collate_samples([ds[i]])
        img = batch["images"][0]
        prompt = batch["prompts"][0]
        answer = batch["answers"][0]
        if answer is None:
            continue
        pred = vismem.generate(images=[img], prompts=[prompt], max_new_tokens=args.max_new_tokens, enable_vismem=args.enable_vismem)[0]
        preds.append(pred)
        refs.append(answer)

    if args.metric == "exact":
        rewards = exact_match_reward(preds, refs)
    else:
        rewards = substring_reward(preds, refs)
    score = sum(rewards) / max(1, len(rewards))
    print(f"{args.metric} score: {score:.4f} ({len(rewards)} examples)")

if __name__ == "__main__":
    main()
