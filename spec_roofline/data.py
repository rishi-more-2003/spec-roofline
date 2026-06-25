"""spec_roofline/data.py — small synthetic task suite + a wikitext slice.

Taskspec §2: avoid heavy eval deps; synthesize a small task suite and use a
wikitext slice. These feed the gates (fixed prompt suite), the quality bench
(passkey + ppl + task accuracy), and the conformal calibration (held-out prompts
whose target-greedy continuation is the quality reference).
"""

from __future__ import annotations

import random

import torch


# A fixed, varied instruction suite — short factual / reasoning / generative
# prompts. Used for the §7 gate and as calibration/test prompts.
TASK_PROMPTS = [
    "Explain why the sky appears blue during the day.",
    "List three differences between Python lists and tuples.",
    "Summarize the water cycle in two sentences.",
    "What is the capital of France, and name one landmark there?",
    "Write a short definition of machine learning.",
    "Give two reasons regular exercise is good for health.",
    "Describe how a bill becomes a law in a parliamentary system.",
    "What causes the seasons to change throughout the year?",
    "Explain the difference between weather and climate.",
    "Name three uses of a hash table in computer science.",
    "Describe the plot of a story about a robot learning to paint.",
    "What are the primary colors and how do they combine?",
    "Explain recursion to a beginner with one example.",
    "List the planets of the solar system in order from the sun.",
    "Why do leaves change color in autumn?",
    "Give a one-paragraph overview of how vaccines work.",
]


def chat_ids(tokenizer, prompt: str) -> torch.Tensor:
    """Apply the chat template; return input_ids [1, L]."""
    msgs = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt")


def gate_prompts(tokenizer, n: int = 8):
    """Fixed prompt suite for the §7 gate (deterministic order)."""
    return [chat_ids(tokenizer, p) for p in TASK_PROMPTS[:n]]


def calibration_prompts(tokenizer, n_cal: int, n_test: int, seed: int = 0):
    """Disjoint cal/test prompt splits for RCPS coverage (taskspec §7 lossy gate).

    Built by pairing the task suite with varied lengths/topics so the calibration
    is not a single distribution; deterministic given seed.
    """
    rng = random.Random(seed)
    pool = []
    suffixes = ["", " Be concise.", " Give a detailed answer.",
                " Answer in a numbered list.", " Explain step by step."]
    for p in TASK_PROMPTS:
        for sfx in suffixes:
            pool.append(p + sfx)
    rng.shuffle(pool)
    need = n_cal + n_test
    while len(pool) < need:        # repeat with marker if suite is small
        pool += [p + f" ({len(pool)})" for p in TASK_PROMPTS]
    cal = [chat_ids(tokenizer, p) for p in pool[:n_cal]]
    test = [chat_ids(tokenizer, p) for p in pool[n_cal:n_cal + n_test]]
    return cal, test


def wikitext_slice(tokenizer, n_tokens: int = 4096):
    """A wikitext-2 test slice as a single id tensor (for perplexity)."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :n_tokens]
    return ids


def passkey_prompt(tokenizer, context_len: int, passkey: int, depth: float = 0.5):
    """Build a passkey-retrieval prompt: a passkey buried in filler at `depth`,
    padded to ~context_len tokens, then asked back. Long-context drafting stress."""
    filler = ("The grass is green and the sky is blue. " * 2000)
    needle = f" The secret passkey is {passkey}. Remember it. "
    fid = tokenizer(filler, return_tensors="pt").input_ids[0]
    target = max(1, context_len)
    # place needle at fractional depth.
    body = tokenizer.decode(fid[:target], skip_special_tokens=True)
    cut = int(len(body) * depth)
    text = (body[:cut] + needle + body[cut:]
            + "\n\nWhat is the secret passkey? The passkey is")
    return tokenizer(text, return_tensors="pt").input_ids
