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
from main.trainer.rewards import exact_match_reward
from main.trainer.grpo import SimpleGRPOTrainer
from main.cli.common import load_yaml, build_vismem_config
from main.trainer.stage2_invocation import compute_penalties

logger = get_logger("main.train_stage2")

reward_ema = 0.0
ema_alpha = 0.05
ptype_w = 1.0
pneg_w = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vismem_qwen25vl7b.yaml")
    ap.add_argument("--model_name_or_path", default=None)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--init_from", default=None, help="Stage1 checkpoint folder containing main.pt")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--kl_beta", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    cfg_dict = load_yaml(args.config)
    if args.model_name_or_path is not None:
        cfg_dict["model"]["model_name_or_path"] = args.model_name_or_path
    viscfg = build_vismem_config(cfg_dict)

    set_seed(int(cfg_dict.get("training", {}).get("seed", 42)))

    model_name = cfg_dict["model"]["model_name_or_path"]
    dtype = to_torch_dtype(cfg_dict["model"].get("torch_dtype", "bfloat16"))
    device_map = cfg_dict["model"].get("device_map", "auto")
    trust = bool(cfg_dict["model"].get("trust_remote_code", True))

    base_model, tokenizer, processor = load_qwen25vl(model_name, torch_dtype=dtype, device_map=device_map,
                                                     trust_remote_code=trust)
    vismem = VisMemModel(base_model, tokenizer, processor, viscfg)

    # load stage1
    if args.init_from is not None:
        state = torch.load(os.path.join(args.init_from, "main.pt"), map_location="cpu")
        vismem.load_state_dict(state["vismem_state"], strict=False)

    # Freeze memory formation; train only a small subset to learn invocation patterns
    for p in vismem.parameters():
        p.requires_grad = False

    # Unfreeze token embeddings
    emb = vismem.base_model.get_input_embeddings()
    emb.weight.requires_grad = True

    special_ids = torch.tensor(
        [vismem.short_invoke_id, vismem.short_end_id, vismem.long_invoke_id, vismem.long_end_id],
        device=vismem.device,
        dtype=torch.long,
    )
    end_ids = torch.tensor([vismem.short_end_id, vismem.long_end_id], device=vismem.device, dtype=torch.long)
    end_lr_mult = 0.1

    def grad_mask_hook(grad):
        g = torch.zeros_like(grad)
        g[special_ids] = grad[special_ids]
        g[end_ids] *= end_lr_mult
        return g

    emb.weight.register_hook(grad_mask_hook)

    opt = optim.AdamW([emb.weight], lr=args.lr)


    ref_model = None
    try:
        import copy
        ref_model = copy.deepcopy(vismem.base_model).eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    except Exception:
        ref_model = None

    trainer = SimpleGRPOTrainer(vismem, ref_model=ref_model, kl_beta=args.kl_beta)

    ds = JsonlVLDataset(args.train_jsonl)
    ensure_dir(args.output_dir)

    vismem.train()
    for epoch in range(args.epochs):
        pbar = tqdm(range(len(ds)), desc=f"Stage2 epoch {epoch}")
        for i in pbar:
            batch = collate_samples([ds[i]])
            img = batch["images"][0]
            prompt = batch["prompts"][0]
            answer = batch["answers"][0]
            if answer is None:
                continue

            # Prepare prompt inputs
            inputs = processor(text=[prompt], images=[img], return_tensors="pt", padding=True)
            inputs = {k: v.to(vismem.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            prompt_ids = inputs["input_ids"]

            pred_list, gen_ids = vismem.generate(
                images=[img],
                prompts=[prompt],
                max_new_tokens=int(cfg_dict.get("training", {}).get("max_new_tokens", 128)),
                enable_vismem=True,
                return_token_ids=True,
                skip_special_tokens=True,
            )
            pred = pred_list[0]
            r = exact_match_reward([pred], [answer])[0]

            # tau_rev: reverse memory type (ptype)
            pred_rev_list = vismem.generate(
                images=[img],
                prompts=[prompt],
                max_new_tokens=int(cfg_dict.get("training", {}).get("max_new_tokens", 128)),
                enable_vismem=True,
                return_token_ids=False,
                skip_special_tokens=True,
                reverse_mem_type=True,
            )
            pred_rev = pred_rev_list[0]
            r_rev = exact_match_reward([pred_rev], [answer])[0]

            reward_ema = (1 - ema_alpha) * reward_ema + ema_alpha * float(r)
            pen = compute_penalties(float(r), float(r_rev), reward_ema)
            r_eff = float(r) - ptype_w * pen["ptype"] - pneg_w * pen["pneg"]

        sampled_ids = tokenizer(pred, return_tensors="pt").input_ids.to(vismem.device)
        rewards = torch.tensor([r_eff], device=vismem.device, dtype=torch.float32)

        loss = trainer.loss_from_samples({"input_ids": prompt_ids}, sampled_ids, rewards)

        opt.zero_grad()
        loss.backward()
        opt.step()

        pbar.set_postfix({"reward": float(r), "loss": float(loss.detach().cpu())})

    ckpt = os.path.join(args.output_dir, f"epoch{epoch}")
    ensure_dir(ckpt)
    torch.save({"vismem_state": vismem.state_dict(), "config": cfg_dict}, os.path.join(ckpt, "main.pt"))
    tokenizer.save_pretrained(ckpt)


logger.info("Stage2 done.")

if __name__ == "__main__":
    main()
