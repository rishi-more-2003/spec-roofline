"""spec_roofline/drafter.py — the Drafter (taskspec §9.1).

Proposes K candidate tokens to be verified in one batched target forward. Two
drafters behind the §3 flag:

  * "model"         — the 0.5B small model, autoregressive. Returns the proposed
    tokens *and* their draft distributions q (needed by the exact sampling
    verify). Keeps its own KV cache; the engine crops it on rollback.
  * "prompt_lookup" — zero draft weights. Finds the most recent earlier
    occurrence of the current suffix n-gram and copies what followed it (Saxena
    2023). Excellent on repetitive / long context, free. Greedy-verify only
    (a copied span has no sampling distribution), so q is None; the engine
    asserts greedy mode for this drafter.

Both can return fewer than K tokens (prompt-lookup may find no match -> 0, and the
engine then takes a single plain target step). The draft *distribution* q is only
materialised in sample mode (it is what the rejection-sampling verify needs).
"""

from __future__ import annotations

import torch

from .config import DraftConfig
from .model import forward_logits


def _propose_model(draft_model, cache, committed, draft_seen, k, mode,
                   temperature, gen):
    """Autoregressive small-model proposal. Re-feeds committed[-1] first so the
    draft cache (holding committed[:-1]) produces a distribution for slot 1."""
    device = committed.device
    tokens, qs = [], []
    cur = committed[-1:].view(1, 1)
    past = draft_seen
    for _ in range(k):
        logits = forward_logits(draft_model, cur, cache, past)   # [1, V]
        past += 1
        logit = logits[-1]
        if mode == "greedy":
            q = torch.softmax(logit, dim=-1)
            tok = int(logit.argmax())
        else:
            probs = torch.softmax(logit / temperature, dim=-1)
            q = probs
            tok = int(torch.multinomial(probs, 1, generator=gen).item())
        tokens.append(tok)
        qs.append(q)
        cur = torch.tensor([[tok]], device=device)
    toks = torch.tensor(tokens, device=device, dtype=torch.long)
    qmat = torch.stack(qs) if mode == "sample" else None
    return toks, qmat, past


def _propose_prompt_lookup(committed, k, ngram):
    """Copy the K tokens that followed the most recent match of the suffix n-gram.
    Greedy-verify only; returns (tokens, None). Empty if no match (engine steps)."""
    ids = committed.tolist()
    n = min(ngram, len(ids) - 1)
    if n <= 0:
        return torch.empty(0, dtype=torch.long, device=committed.device), None
    suffix = ids[-n:]
    # search backwards for the most recent earlier occurrence of `suffix`.
    for start in range(len(ids) - n - 1, -1, -1):
        if ids[start:start + n] == suffix:
            nxt = ids[start + n:start + n + k]
            if nxt:
                return torch.tensor(nxt, device=committed.device, dtype=torch.long), None
            break
    return torch.empty(0, dtype=torch.long, device=committed.device), None


class Drafter:
    """Dispatches to the configured drafter; uniform (tokens, q, draft_seen) out."""

    def __init__(self, cfg: DraftConfig):
        self.cfg = cfg

    def propose(self, draft_model, cache, committed, draft_seen, mode, gen):
        if self.cfg.method == "prompt_lookup":
            if mode != "greedy":
                raise ValueError("prompt_lookup drafter supports greedy verify only "
                                 "(no draft distribution to rejection-sample)")
            toks, q = _propose_prompt_lookup(committed, self.cfg.k, self.cfg.pl_ngram)
            return toks, q, draft_seen          # cache untouched for this drafter
        return _propose_model(draft_model, cache, committed, draft_seen,
                              self.cfg.k, mode, self.cfg.temperature, gen)
