from __future__ import annotations
import argparse
import os
from tqdm import tqdm
import torch
import torch.optim as optim

from main.utils.logging import get_logger
from main.utils.misc import set_seed, to_torch_dtype, ensure_dir
from main.utils.qwen_vl import load_qwen25vl
from main.model.model import VisMemModel
from main.data.jsonl_dataset import JsonlVLDataset
from main.data.collate import collate_samples
from main.trainer.stage1_memory_formation import stage1_loss
from main.cli.common import load_yaml, build_vismem_config

logger = get_logger("main.train_stage1")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vismem_qwen25vl7b.yaml")
    ap.add_argument("--model_name_or_path", default=None)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()

    cfg_dict = load_yaml(args.config)
    if args.model_name_or_path is not None:
        cfg_dict["model"]["model_name_or_path"] = args.model_name_or_path
    viscfg = build_vismem_config(cfg_dict)

    set_seed(int(cfg_dict.get("training", {}).get("seed", 42)))

    model_name = cfg_dict["model"]["model_name_or_path"]
    dtype = to_torch_dtype(cfg_dict["model"].get("torch_dtype","bfloat16"))
    device_map = cfg_dict["model"].get("device_map","auto")
    trust = bool(cfg_dict["model"].get("trust_remote_code", True))

    base_model, tokenizer, processor = load_qwen25vl(model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=trust)
    vismem = VisMemModel(base_model, tokenizer, processor, viscfg)

    # Freeze base model
    for p in vismem.base_model.parameters():
        p.requires_grad = False

    # Trainable params
    trainable = [p for p in vismem.parameters() if p.requires_grad]
    lr = args.lr if args.lr is not None else float(cfg_dict.get("training", {}).get("lr", 2e-4))
    opt = optim.AdamW(trainable, lr=lr)

    ds = JsonlVLDataset(args.train_jsonl)
    ensure_dir(args.output_dir)

    vismem.train()
    for epoch in range(args.epochs):
        pbar = tqdm(range(len(ds)), desc=f"Stage1 epoch {epoch}")
        for i in pbar:
            batch = collate_samples([ds[i]])
            img = batch["images"][0]
            prompt = batch["prompts"][0]
            answer = batch["answers"][0]
            if answer is None:
                continue

            inputs = processor(text=[prompt], images=[img], return_tensors="pt", padding=True)
            inputs = {k:v.to(vismem.device) if hasattr(v, "to") else v for k,v in inputs.items()}

            loss_mem, loss_base = stage1_loss(vismem.base_model, vismem, inputs, answer)
            loss = loss_mem - loss_base.detach()
            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix({"loss_mem": float(loss_mem.detach().cpu()), "loss_base": float(loss_base.detach().cpu())})

        # save checkpoint
        ckpt = os.path.join(args.output_dir, f"epoch{epoch}")
        ensure_dir(ckpt)
        torch.save({"vismem_state": vismem.state_dict(), "config": cfg_dict}, os.path.join(ckpt, "main.pt"))
        tokenizer.save_pretrained(ckpt)

    logger.info("Stage1 done.")

if __name__ == "__main__":
    main()
