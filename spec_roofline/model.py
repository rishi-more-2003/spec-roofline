"""spec_roofline/model.py — load target + draft, the batched verify forward.

Thin integration seam (mirrors kv-roofline's model.py). Loads the two pinned
Qwen models so they co-reside on 8 GB, and exposes the *single batched target
forward* that verifies K drafted tokens at once — the whole point of speculative
decoding (one expensive step amortised over many drafted tokens).

The verify forward uses ``cache_position`` so a DynamicCache can be advanced and
later cropped (KV rollback on rejection, engine.py). Caches always hold the
committed prefix; the last committed token is re-fed as the first verify input so
the target produces a distribution for the *first* drafted slot.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from .config import ModelConfig


@dataclass
class TwoModels:
    target: torch.nn.Module
    draft: torch.nn.Module
    tokenizer: object
    device: str
    vocab_size: int

    def new_cache(self) -> DynamicCache:
        return DynamicCache()


def load_models(mcfg: ModelConfig | None = None) -> TwoModels:
    mcfg = mcfg or ModelConfig()
    dtype = getattr(torch, mcfg.dtype)
    target = AutoModelForCausalLM.from_pretrained(
        mcfg.target_id, torch_dtype=dtype, attn_implementation=mcfg.attn_impl)
    draft = AutoModelForCausalLM.from_pretrained(
        mcfg.draft_id, torch_dtype=dtype, attn_implementation=mcfg.attn_impl)
    target.to(mcfg.device).eval()
    draft.to(mcfg.device).eval()
    tok = AutoTokenizer.from_pretrained(mcfg.target_id)
    # verify requires identical vocab over both models.
    assert target.config.vocab_size == draft.config.vocab_size, (
        f"vocab mismatch: target {target.config.vocab_size} != draft "
        f"{draft.config.vocab_size}; verify step is undefined")
    return TwoModels(target, draft, tok, mcfg.device,
                     vocab_size=target.config.vocab_size)


@torch.no_grad()
def forward_logits(model: torch.nn.Module, input_ids: torch.Tensor,
                   cache: DynamicCache, past_len: int) -> torch.Tensor:
    """One forward over ``input_ids`` (shape [1, L]) continuing ``cache``.

    Returns logits [L, vocab] (float32). ``past_len`` is the number of tokens
    already in the cache; ``cache_position`` is set to arange(past_len, past_len+L)
    so the cache appends correctly and stays croppable.
    """
    device = input_ids.device
    L = input_ids.shape[1]
    cache_position = torch.arange(past_len, past_len + L, device=device)
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True,
                cache_position=cache_position)
    return out.logits[0].float()


def sync_cache(model: torch.nn.Module, cache: DynamicCache, seen: int,
               target_ids: torch.Tensor) -> int:
    """Make ``cache`` hold exactly ``target_ids`` (shape [T]); crop or extend.

    The KV-rollback primitive (taskspec §5 "the subtle systems part"). After a
    speculation round the caches are over-extended with rejected drafts; this
    crops them back, and feeds any missing tokens when a full-accept left a cache
    one short. Returns the new ``seen``.
    """
    T = int(target_ids.shape[0])
    if seen > T:
        cache.crop(T)
        seen = T
    if seen < T:
        inp = target_ids[seen:T].unsqueeze(0)
        forward_logits(model, inp, cache, seen)
        seen = T
    return seen
