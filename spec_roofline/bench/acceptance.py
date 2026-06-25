"""spec_roofline/bench/acceptance.py — acceptance rate vs K / context (§6).

Acceptance rate = mean accepted drafts per round / K (how often the draft "guesses
right"). Together with K it determines accepted-tokens-per-target-call, the actual
speedup lever. Swept over draft length K and over context length, for the small-
model drafter vs prompt-lookup.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict

import torch

from ..config import Config
from ..engine import SpeculativeDecoder
from .common import build_context, RESULTS


@dataclass
class AcceptRow:
    drafter: str
    context: int
    k: int
    acceptance_rate: float       # accepted drafts / (rounds * K)
    accepted_per_call: float
    n_generated: int


@torch.no_grad()
def run(models, cfg: Config | None = None, ks=(2, 4, 6, 8),
        contexts=(1024, 4096, 16384), n_new: int = 96) -> dict:
    cfg = cfg or Config()
    rows = []
    for drafter in ("model", "prompt_lookup"):
        for ctx in contexts:
            ids = build_context(models.tokenizer, ctx, models.device)
            for k in ks:
                c = cfg.with_method(drafter).with_k(k)
                dec = SpeculativeDecoder(models, c)
                r = dec.generate_spec(ids, n_new, mode="greedy", gamma=0.0)
                total_drafts = max(1, r.n_rounds * k)
                rows.append(AcceptRow(
                    drafter=drafter, context=ctx, k=k,
                    acceptance_rate=round(r.n_accept_total / total_drafts, 3),
                    accepted_per_call=round(r.accepted_per_call, 3),
                    n_generated=r.n_generated))
            del ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    _write(rows)
    return {"rows": rows}


def _write(rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "acceptance.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return path
