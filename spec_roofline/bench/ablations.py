"""spec_roofline/bench/ablations.py — K sweep, drafter, leniency, crossover (§6).

Feeds the headline "tok/s vs guaranteed quality loss" curve (plot 2) and the
honest caveat plot (plot 4: where lossy stops helping). Three sweeps:

  * gamma sweep — accepted/call, tok/s, and risk (vs target greedy) vs leniency,
    at fixed context. The speed<->fidelity trade laid bare.
  * K sweep — accepted/call vs draft length.
  * lossy-vs-context — lossless and lossy speedup vs no-spec across contexts; the
    crossover where draft overhead eats the win (short context / low acceptance).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict

import torch

from ..config import Config
from ..engine import SpeculativeDecoder
from .common import build_context, time_call, RESULTS


def _risk(out, ref):
    L = min(len(out), len(ref))
    mism = sum(1 for i in range(L) if out[i] != ref[i]) + abs(len(out) - len(ref))
    return round(mism / max(1, len(out), len(ref)), 3)


@dataclass
class GammaRow:
    gamma: float
    accepted_per_call: float
    tok_s: float
    risk: float


@dataclass
class KRow:
    k: int
    accepted_per_call: float
    acceptance_rate: float


@dataclass
class CrossRow:
    context: int
    nospec_tok_s: float
    lossless_tok_s: float
    lossy_tok_s: float
    lossless_speedup: float
    lossy_speedup: float


@torch.no_grad()
def gamma_sweep(models, cfg, ctx=4096, n_new=64):
    dec = SpeculativeDecoder(models, cfg)
    ids = build_context(models.tokenizer, ctx, models.device)
    ref = dec.generate_baseline(ids, n_new, mode="greedy").tokens
    rows = []
    for g in cfg.lossy.gamma_grid:
        res, secs = time_call(lambda: dec.generate_spec(ids, n_new, mode="greedy", gamma=g))
        rows.append(GammaRow(g, round(res.accepted_per_call, 3),
                             round(res.n_generated / secs, 2), _risk(res.tokens, ref)))
    return rows


@torch.no_grad()
def k_sweep(models, cfg, ctx=4096, ks=(1, 2, 4, 6, 8), n_new=96):
    ids = build_context(models.tokenizer, ctx, models.device)
    rows = []
    for k in ks:
        dec = SpeculativeDecoder(models, cfg.with_k(k))
        r = dec.generate_spec(ids, n_new, mode="greedy", gamma=0.0)
        rows.append(KRow(k, round(r.accepted_per_call, 3),
                         round(r.n_accept_total / max(1, r.n_rounds * k), 3)))
    return rows


@torch.no_grad()
def crossover(models, cfg, contexts=(256, 1024, 4096, 16384), n_new=64, lossy_gamma=0.3):
    dec = SpeculativeDecoder(models, cfg)
    rows = []
    for ctx in contexts:
        ids = build_context(models.tokenizer, ctx, models.device)
        _, t_ns = time_call(lambda: dec.generate_baseline(ids, n_new, mode="greedy"))
        _, t_ll = time_call(lambda: dec.generate_spec(ids, n_new, mode="greedy", gamma=0.0))
        _, t_ly = time_call(lambda: dec.generate_spec(ids, n_new, mode="greedy", gamma=lossy_gamma))
        rows.append(CrossRow(ctx, round(n_new / t_ns, 2), round(n_new / t_ll, 2),
                             round(n_new / t_ly, 2), round(t_ns / t_ll, 2),
                             round(t_ns / t_ly, 2)))
        del ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def run_all(models, cfg: Config | None = None, *, gamma_ctx=4096, gamma_new=64,
            k_ctx=4096, ks=(1, 2, 4, 6, 8), k_new=96,
            crossover_contexts=(256, 1024, 4096, 16384), crossover_new=64) -> dict:
    cfg = cfg or Config()
    out = {"gamma": gamma_sweep(models, cfg, ctx=gamma_ctx, n_new=gamma_new),
           "k": k_sweep(models, cfg, ctx=k_ctx, ks=ks, n_new=k_new),
           "crossover": crossover(models, cfg, contexts=crossover_contexts, n_new=crossover_new)}
    _write("ablation_gamma.csv", out["gamma"])
    _write("ablation_k.csv", out["k"])
    _write("ablation_crossover.csv", out["crossover"])
    return out


def _write(name, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
