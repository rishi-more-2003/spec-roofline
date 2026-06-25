"""spec_roofline/bench/throughput.py — the lead plot (taskspec §6).

Decode tok/s vs context length for the three regimes: no-spec / lossless-spec /
lossy-spec. The through-line of the trilogy made measurable: batch-1 decode is
HBM-bound, so the cheapest token is the one you never run the big model for —
here, the expensive unit is a *target forward*, and speculation amortises it over
several accepted tokens.

Reports tok/s, mean accepted-tokens-per-target-call, and peak VRAM (OOM at long
context is a reported data point, not a crash — taskspec §2). Run foreground.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict

import torch

from ..config import Config
from ..engine import SpeculativeDecoder
from .common import time_call, build_context, vram_mb, cuda_sync, RESULTS


@dataclass
class ThroughputRow:
    context: int
    regime: str          # no_spec | lossless | lossy
    gamma: float
    tok_s: float
    accepted_per_call: float
    target_forwards: int
    n_generated: int
    peak_vram_mb: float
    oom: bool = False


def _measure(dec, ids, n_new, warmup, regime, gamma):
    mode = "greedy"
    def run():
        if regime == "no_spec":
            return dec.generate_baseline(ids, n_new, mode=mode)
        return dec.generate_spec(ids, n_new, mode=mode, gamma=gamma)
    # warmup
    dec.generate_baseline(ids, min(warmup, 8), mode=mode) if regime == "no_spec" else \
        dec.generate_spec(ids, min(warmup, 8), mode=mode, gamma=gamma)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    res, secs = time_call(run)
    return res, secs


def run(models, cfg: Config | None = None, lossy_gamma: float = 0.3) -> dict:
    cfg = cfg or Config()
    dec = SpeculativeDecoder(models, cfg)
    n_new = cfg.bench.decode_steps
    rows = []
    for ctx in cfg.bench.context_lengths:
        try:
            ids = build_context(models.tokenizer, ctx, models.device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append(ThroughputRow(ctx, "no_spec", 0.0, 0, 0, 0, 0, 0, oom=True))
            continue
        for regime, gamma in (("no_spec", 0.0), ("lossless", 0.0), ("lossy", lossy_gamma)):
            try:
                res, secs = _measure(dec, ids, n_new, cfg.bench.warmup_steps, regime, gamma)
                rows.append(ThroughputRow(
                    context=ctx, regime=regime, gamma=gamma,
                    tok_s=round(res.n_generated / secs, 2),
                    accepted_per_call=round(res.accepted_per_call, 3),
                    target_forwards=res.n_target_forwards,
                    n_generated=res.n_generated,
                    peak_vram_mb=round(vram_mb(), 1)))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rows.append(ThroughputRow(ctx, regime, gamma, 0, 0, 0, 0,
                                          round(vram_mb(), 1), oom=True))
        del ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _write(rows)
    return {"rows": rows}


def _write(rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "throughput.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return path
