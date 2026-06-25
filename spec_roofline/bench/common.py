"""spec_roofline/bench/common.py — shared timing + long-context prompt builders."""

from __future__ import annotations

import time
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_call(fn):
    """Wall time (s) of fn() with CUDA sync on both sides; returns (result, secs)."""
    cuda_sync()
    t0 = time.perf_counter()
    out = fn()
    cuda_sync()
    return out, time.perf_counter() - t0


def build_context(tokenizer, n_tokens: int, device: str) -> torch.Tensor:
    """A coherent prompt of ~n_tokens, ending at an assistant generation point.

    Repetitive natural-language filler (also exercises prompt-lookup) wrapped in
    the chat template, truncated to n_tokens. Batch-1 long-context decode regime.
    """
    filler = ("In a quiet town surrounded by rolling green hills, the seasons "
              "turned slowly and the people kept careful records of the weather, "
              "the harvest, and the small daily events that made up their lives. ")
    reps = max(1, n_tokens // 32)
    text = filler * reps
    body = tokenizer(text, return_tensors="pt").input_ids[0, :max(8, n_tokens - 24)]
    body_text = tokenizer.decode(body, skip_special_tokens=True)
    msgs = [{"role": "user",
             "content": body_text + "\n\nContinue the story in the same style."}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                        return_tensors="pt")
    return ids.to(device)


def vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0
