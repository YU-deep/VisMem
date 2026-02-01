from __future__ import annotations
import math
from dataclasses import asdict
from typing import Any, Dict, Optional, List, Tuple, Union

import torch
import torch.nn as nn

from main.model.configuration_vismem import VisMemConfig
from main.model.query_builder import QueryBuilder
from main.model.memory_former import TinyMemoryFormer
from main.model.lora_utils import is_peft_available, make_lora_adapters, set_active_adapter

class VisMemModel(nn.Module):


    def __init__(self, base_model, tokenizer, processor, config: VisMemConfig):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.cfg = config

        # Identify hidden size
        hidden_size = getattr(base_model.config, "hidden_size", None)
        if hidden_size is None and hasattr(base_model.config, "text_config"):
            hidden_size = getattr(base_model.config.text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from base_model.config.")

        self.hidden_size = hidden_size

        # Query builder
        qb = config.query_builder
        self.query_builder = QueryBuilder(
            hidden_size=hidden_size,
            query_len=config.query_len,
            num_layers=qb.num_layers,
            num_heads=qb.num_heads,
            dropout=qb.dropout,
            ff_mult=qb.ff_mult,
        )

        # Memory formers
        self.former_backend = config.former_backend
        self.short_former = None
        self.long_former = None

        if self.former_backend == "tiny_transformer" or not is_peft_available():
            self.short_former = TinyMemoryFormer(hidden_size, config.short_mem_len, num_layers=2, num_heads=8)
            self.long_former  = TinyMemoryFormer(hidden_size, config.long_mem_len,  num_layers=2, num_heads=8)
            self.peft_model = None
        elif self.former_backend == "lora_llm":
            lora = config.lora
            short_targets = lora.short_target_modules or lora.target_modules
            long_targets = lora.long_target_modules or lora.target_modules
            #
            self.peft_model = make_lora_adapters(base_model, "short_former", lora.r, lora.alpha, lora.dropout, short_targets)
            from peft import LoraConfig
            self.peft_model.add_adapter(
                "long_former",
                LoraConfig(
                    r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
                    bias="none", task_type="CAUSAL_LM", target_modules=long_targets
                )
            )
            self.m_init_short = nn.Parameter(torch.randn(1, config.short_mem_len, hidden_size) * 0.02)
            self.m_init_long = nn.Parameter(torch.randn(1, config.long_mem_len, hidden_size) * 0.02)
        else:
            raise ValueError(f"Unknown former_backend: {self.former_backend}")

        # Token ids
        self.short_invoke_id = tokenizer.convert_tokens_to_ids(config.short_invoke_token)
        self.short_end_id    = tokenizer.convert_tokens_to_ids(config.short_end_token)
        self.long_invoke_id  = tokenizer.convert_tokens_to_ids(config.long_invoke_token)
        self.long_end_id     = tokenizer.convert_tokens_to_ids(config.long_end_token)

        if any(x is None or x == tokenizer.unk_token_id for x in [self.short_invoke_id, self.short_end_id, self.long_invoke_id, self.long_end_id]):
            raise ValueError("Special tokens not found in tokenizer. Make sure to call add_vismem_tokens().")

    @property
    def device(self):
        return next(self.parameters()).device

    def _select_visual_positions(self, input_ids: torch.LongTensor) -> torch.BoolTensor:
        # Try to locate special ids if present
        vs_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        ve_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        if vs_id is None or ve_id is None or vs_id == self.tokenizer.unk_token_id or ve_id == self.tokenizer.unk_token_id:
            # Fallback: no visual positions
            return torch.zeros_like(input_ids, dtype=torch.bool)

        B, T = input_ids.shape
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for b in range(B):
            ids = input_ids[b].tolist()
            try:
                s = ids.index(vs_id)
                e = ids.index(ve_id)
                if e > s:
                    mask[b, s:e+1] = True
            except ValueError:
                pass
        return mask

    def _build_H(self, visual_states: torch.Tensor, text_states: torch.Tensor) -> torch.Tensor:
        # Cap length to reduce compute
        if text_states.size(1) > self.cfg.max_prompt_hidden:
            text_states = text_states[:, -self.cfg.max_prompt_hidden:, :]
        return torch.cat([visual_states, text_states], dim=1)


    def _maybe_project_short_memory(self, M: torch.Tensor) -> torch.Tensor:
        proj = (
                getattr(self.base_model, "visual_projector", None)
                or getattr(self.base_model, "vision_projector", None)
                or getattr(self.base_model, "multi_modal_projector", None)
        )
        if proj is None:
            return M
        try:
            return proj(M)
        except Exception:
            return M


    def _former_forward_lora(self, X: torch.Tensor, Q: torch.Tensor, mem_len: int, adapter_name: str) -> torch.Tensor:
        # Use the underlying LLM forward on embeddings; assumes base_model supports inputs_embeds.
        peft_model = self.peft_model
        set_active_adapter(peft_model, adapter_name)
        B = X.size(0)

        if adapter_name == "short_former":
            m_init = self.m_init_short.expand(B, -1, -1).to(dtype=X.dtype, device=X.device)
        else:
            m_init = self.m_init_long.expand(B, -1, -1).to(dtype=X.dtype, device=X.device)

        inp = torch.cat([X, Q, m_init], dim=1)
        attn = torch.ones(B, inp.size(1), device=X.device, dtype=torch.long)
        out = peft_model(inputs_embeds=inp, attention_mask=attn, use_cache=False, output_hidden_states=True)
        hs = out.hidden_states[-1]
        M = hs[:, -mem_len:, :]
        return M

    def form_memory(self, H: torch.Tensor, mem_type: str) -> torch.Tensor:
        Q = self.query_builder(H)
        if self.peft_model is None:
            if mem_type == "short":
                return self.short_former(H, Q)
            else:
                return self.long_former(H, Q)
        else:
            if mem_type == "short":
                return self._former_forward_lora(H, Q, self.cfg.short_mem_len, "short_former")
            else:
                return self._former_forward_lora(H, Q, self.cfg.long_mem_len, "long_former")

    def _gather_padded(self, states: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        # states: (B,T,D), mask: (B,T)
        B, T, D = states.shape
        lens = mask.sum(dim=1)
        max_len = int(lens.max().item()) if lens.numel() else 0
        out = states.new_zeros((B, max_len, D))
        for b in range(B):
            idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                out[b, : idx.numel()] = states[b, idx]
        return out


    @torch.no_grad()
    def generate(
        self,
        images: Optional[List[Any]],
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        enable_vismem: bool = True,
        return_token_ids: bool = False,
        skip_special_tokens: bool = True,
        reverse_mem_type: bool = False,
        ):
        # batch size 1 recommended; we keep batch support
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k:v.to(self.device) if hasattr(v, "to") else v for k,v in inputs.items()}

        # Initial forward
        out = self.base_model(**inputs, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        hidden_last = out.hidden_states[-1]  # (B, T, D)

        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            raise ValueError("Processor did not return input_ids; check your Qwen2.5-VL processor.")
        B, T = input_ids.shape

        visual_mask = self._select_visual_positions(input_ids)

        if visual_mask.any():
            visual_states = self._gather_padded(hidden_last, visual_mask)
        else:
            visual_states = torch.zeros(B, 0, self.hidden_size, device=self.device, dtype=hidden_last.dtype)

        seg_hiddens: List[torch.Tensor] = []  # each (B,1,D)

        generated = []

        def sample_next(logits_):
            if temperature <= 0:
                return torch.argmax(logits_, dim=-1)
            probs = torch.softmax(logits_ / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum > top_p
                # keep at least 1
                mask[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_idx = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
                return sorted_idx.gather(-1, next_idx.unsqueeze(-1)).squeeze(-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

        # decoding loop
        cur_logits = logits
        for step in range(max_new_tokens):
            next_id = sample_next(cur_logits)
            generated.append(next_id)
            seg_hiddens.append(hidden_last[:, -1:, :])

            # Check invocation
            if enable_vismem and (((next_id == self.short_invoke_id).any()) or ((next_id == self.long_invoke_id).any())):
                # Feed invocation token
                out = self.base_model(
                    input_ids=next_id.unsqueeze(-1),
                    use_cache=True,
                    past_key_values=past,
                    output_hidden_states=True,
                    attention_mask=None,
                )
                past = out.past_key_values
                hidden_last = out.hidden_states[-1]  # (B,1,D)
                seg_hiddens.append(hidden_last)

                token_type = "short" if (next_id == self.short_invoke_id).any() else "long"
                mem_type = (
                    "long" if token_type == "short" else "short") if reverse_mem_type else token_type

                end_id = self.short_end_id if mem_type == "short" else self.long_end_id

                # Build H
                text_states = torch.cat(seg_hiddens, dim=1)  # (B, z, D)
                H = self._build_H(visual_states, text_states)

                M = self.form_memory(H, mem_type)  # (B, N, D)
                if mem_type == "short":
                    M = self._maybe_project_short_memory(M)
                # Insert memory tokens
                out = self.base_model(
                    inputs_embeds=M,
                    use_cache=True,
                    past_key_values=past,
                    output_hidden_states=True,
                    attention_mask=None,
                )
                past = out.past_key_values
                hidden_last = out.hidden_states[-1]  # (B,N,D)

                # end
                end_tensor = torch.full((B,), end_id, device=self.device, dtype=torch.long)
                generated.append(end_tensor)
                out = self.base_model(
                    input_ids=end_tensor.unsqueeze(-1),
                    use_cache=True,
                    past_key_values=past,
                    output_hidden_states=True,
                    attention_mask=None,
                )
                past = out.past_key_values
                hidden_last = out.hidden_states[-1]
                seg_hiddens = []  # reset

                cur_logits = out.logits[:, -1, :]
                continue

            # Normal step: feed token to model
            out = self.base_model(
                input_ids=next_id.unsqueeze(-1),
                use_cache=True,
                past_key_values=past,
                output_hidden_states=True,
                attention_mask=None,
            )
            past = out.past_key_values
            hidden_last = out.hidden_states[-1]  # (B,1,D)
            cur_logits = out.logits[:, -1, :]

            # stop token
            if (next_id == self.tokenizer.eos_token_id).all():
                break

        gen_ids = torch.stack(generated, dim=1)  # (B, Lg)
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=skip_special_tokens)
        if return_token_ids:
            return texts, gen_ids
        return texts
