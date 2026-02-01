from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GRPOBatch:
    input_ids: torch.LongTensor          # (B, T)
    attention_mask: torch.LongTensor     # (B, T)
    labels: torch.LongTensor             # (B, T) with -100 for prompt positions

def sequence_logprobs(logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    # gather label logp
    mask = labels != -100
    safe_labels = labels.clone()
    safe_labels[~mask] = 0
    gathered = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * mask
    return gathered.sum(dim=-1)

def kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:

    p = F.log_softmax(logits_p, dim=-1)
    q = F.log_softmax(logits_q, dim=-1)
    p_prob = p.exp()
    kl = (p_prob * (p - q)).sum(dim=-1)  # (B,T)
    return kl.mean(dim=-1)

class SimpleGRPOTrainer:
    def __init__(self, model, ref_model=None, kl_beta: float = 0.02):
        self.model = model
        self.ref_model = ref_model
        self.kl_beta = kl_beta

    def loss_from_samples(self, prompts_inputs: Dict[str, Any], sampled_ids: torch.LongTensor, rewards: torch.Tensor) -> torch.Tensor:
        # Build full input: prompt + sampled
        input_ids = torch.cat([prompts_inputs["input_ids"], sampled_ids], dim=1)
        attn = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        labels = input_ids.clone()
        labels[:, :prompts_inputs["input_ids"].size(1)] = -100  # ignore prompt

        out = self.model.base_model(input_ids=input_ids, attention_mask=attn, output_hidden_states=False)
        logits = out.logits
        logp = sequence_logprobs(logits[:, :-1, :], labels[:, 1:])

        # Normalize rewards -> advantages
        adv = rewards - rewards.mean()
        pg_loss = -(adv.detach() * logp).mean()

        if self.ref_model is None or self.kl_beta <= 0:
            return pg_loss

        with torch.no_grad():
            ref_out = self.ref_model(input_ids=input_ids, attention_mask=attn)
        kl = kl_divergence(logits[:, :-1, :], ref_out.logits[:, :-1, :])
        return pg_loss + self.kl_beta * kl.mean()
