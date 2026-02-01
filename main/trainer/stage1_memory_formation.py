from __future__ import annotations
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F

def stage1_loss(base_model, vismem_model, inputs: Dict[str, Any], target_text: str):

    tokenizer = vismem_model.tokenizer
    device = vismem_model.device

    # Encode target
    tgt_ids = tokenizer(target_text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        full_ids = torch.cat([inputs["input_ids"], tgt_ids], dim=1)
        attn = torch.ones_like(full_ids, dtype=torch.long)
        labels = full_ids.clone()
        labels[:, :inputs["input_ids"].size(1)] = -100
        out = base_model(input_ids=full_ids, attention_mask=attn)
        logits = out.logits[:, :-1, :]
        loss_base = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)

    base_out = base_model(**inputs, output_hidden_states=True)
    hidden = base_out.hidden_states[-1]  # (B,T,D)
    # Build H
    H = hidden
    M = vismem_model.form_memory(H, mem_type="short")  # or "long" / both, configurable

    # Feed
    emb = base_model.get_input_embeddings()(inputs["input_ids"])
    inp_embeds = torch.cat([emb, M, base_model.get_input_embeddings()(tgt_ids)], dim=1)
    attn2 = torch.ones(inp_embeds.size()[:-1], device=device, dtype=torch.long)

    labels2 = torch.cat([inputs["input_ids"], torch.full((inputs["input_ids"].size(0), M.size(1)), -100, device=device, dtype=torch.long), tgt_ids], dim=1)
    labels2[:, :inputs["input_ids"].size(1) + M.size(1)] = -100

    out2 = base_model(inputs_embeds=inp_embeds, attention_mask=attn2)
    logits2 = out2.logits[:, :-1, :]
    loss_mem = F.cross_entropy(logits2.reshape(-1, logits2.size(-1)), labels2[:, 1:].reshape(-1), ignore_index=-100)

    # Optimize
    return loss_mem, loss_base
