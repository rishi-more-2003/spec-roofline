"""spec_roofline/bench/headroom.py — where the lossy knob actually buys speed,
and where it does not (the contribution, with its scope drawn).

A leniency knob can only recover speed where the draft *leaves recoverable
rejections on the table*. Two things have to hold: (1) the draft must get rejected
often enough (low acceptance leaves headroom), and (2) those rejections must be
*near-misses* at low-confidence target positions, so admitting them costs little
quality. For a well-aligned Qwen 1.5B/0.5B pair under greedy decode, exact-verify
rejections are mostly **far-misses** — accepting them drifts the greedy trajectory
hard (a single rank-2 acceptance cascades). So the knob's payoff is small and lives
only at the low-confidence positions the top-mass rule is built to find.

This experiment measures the knob on the **diverse prompt distribution it is
calibrated on** (the same one `conformal.run_coverage` uses) — not the repetitive
context where the target is too confident for the knob to ever bite. For each
gamma we report mean accepted-per-target-call (the speed lever) *and* the risk;
then the RCPS-calibrated gamma per alpha with the acc/call **lift** it buys under
the warranty. The model drafter (high acceptance, small headroom) and prompt-lookup
(near-zero proposals on non-repetitive prompts) bracket the honest scope: leniency
is a lever for *weak-draft / high-rejection* regimes, and a strong draft has little
for it to recover.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict

import torch

from ..config import Config
from ..engine import SpeculativeDecoder
from ..conformal import _eb_ucb, rcps_calibrate
from ..data import calibration_prompts, open_ended_prompts
from .common import RESULTS


def _risk(out, ref):
    L = min(len(out), len(ref))
    mism = sum(1 for i in range(L) if out[i] != ref[i]) + abs(len(out) - len(ref))
    return max(0.0, min(1.0, mism / max(1, len(out), len(ref))))


@dataclass
class HeadroomRow:
    regime: str          # "greedy" or "sample@T"
    drafter: str
    gamma: float
    accepted_per_call: float
    mean_risk: float
    risk_ucb: float


@dataclass
class CalibratedRow:
    regime: str
    drafter: str
    alpha: float
    gamma: float
    accepted_per_call: float     # at the calibrated gamma
    lift_vs_lossless: float      # acc/call(gamma) / acc/call(gamma=0)
    realized_risk: float
    valid: bool


def _matrices_greedy(dec, prompts, gammas, n_new):
    """Greedy regime: risk = token-disagreement vs the gamma=0 (lossless) path."""
    n = len(prompts)
    refs = [dec.generate_spec(p, n_new, mode="greedy", gamma=0.0).tokens for p in prompts]
    risk_M = torch.zeros(n, len(gammas))
    acc_M = torch.zeros(n, len(gammas))
    for i, (p, ref) in enumerate(zip(prompts, refs)):
        for j, g in enumerate(gammas):
            r = dec.generate_spec(p, n_new, mode="greedy", gamma=g)
            risk_M[i, j] = _risk(r.tokens, ref)
            acc_M[i, j] = r.accepted_per_call
        _progress("greedy", i + 1, n)
    return risk_M, acc_M


def _matrices_sample(dec, prompts, gammas, n_new, temperature, device, seed):
    """Sampling regime: risk = mean per-step emission TV(p_gamma||p) (distributional,
    not token-match). A fixed per-prompt seed keeps the trajectories comparable."""
    n = len(prompts)
    risk_M = torch.zeros(n, len(gammas))
    acc_M = torch.zeros(n, len(gammas))
    for i, p in enumerate(prompts):
        for j, g in enumerate(gammas):
            gen = torch.Generator(device=device).manual_seed(seed + 1000 + i)
            r = dec.generate_spec(p, n_new, mode="sample", gamma=g, gen=gen)
            risk_M[i, j] = r.risk_tv
            acc_M[i, j] = r.accepted_per_call
        _progress("sample", i + 1, n)
    return risk_M, acc_M


def _progress(tag, i, n):
    import sys
    print(f"  [headroom/{tag}] {i}/{n} prompts", file=sys.stderr, flush=True)


@torch.no_grad()
def run(models, cfg: Config | None = None, n_prompts: int = 30, n_new: int = 32,
        drafters=("model", "prompt_lookup"), seed: int = 0,
        sampling: bool = False, temperature: float = 1.0) -> dict:
    """Greedy regime (sampling=False) or the sampling regime (sampling=True), where
    the distribution-preserving verify + distribution-free warranty actually bite.
    Sampling uses high-entropy open-ended prompts and the TV risk metric."""
    from dataclasses import replace as _replace
    cfg = cfg or Config()
    gammas = list(cfg.lossy.gamma_grid)
    delta = cfg.lossy.delta
    regime = f"sample@T{temperature}" if sampling else "greedy"

    if sampling:
        prompts = open_ended_prompts(models.tokenizer, n_prompts, seed)
        drafters = ("model",)        # prompt-lookup has no draft distribution to relax
    else:
        prompts, _ = calibration_prompts(models.tokenizer, n_prompts, 0, seed)
    prompts = [p.to(models.device) for p in prompts]

    sweep_rows, calib_rows = [], []
    for drafter in drafters:
        c = cfg.with_method(drafter)
        if sampling:
            c = _replace(c, draft=_replace(c.draft, temperature=temperature))
        dec = SpeculativeDecoder(models, c)
        if sampling:
            risk_M, acc_M = _matrices_sample(dec, prompts, gammas, n_new,
                                             temperature, models.device, seed)
        else:
            risk_M, acc_M = _matrices_greedy(dec, prompts, gammas, n_new)

        accpc = acc_M.mean(0)
        for j, g in enumerate(gammas):
            sweep_rows.append(HeadroomRow(regime, drafter, g, round(float(accpc[j]), 3),
                                          round(float(risk_M[:, j].mean()), 4),
                                          round(_eb_ucb(risk_M[:, j], delta), 4)))
        base_acc = float(accpc[gammas.index(0.0)]) if 0.0 in gammas else float(accpc[0])
        gidx = {g: j for j, g in enumerate(gammas)}
        for alpha in cfg.lossy.alphas:
            g, _ = rcps_calibrate(risk_M, gammas, alpha, delta)
            j = gidx[g]
            calib_rows.append(CalibratedRow(
                regime, drafter, alpha, g, round(float(accpc[j]), 3),
                round(float(accpc[j]) / max(1e-9, base_acc), 3),
                round(float(risk_M[:, j].mean()), 4),
                float(risk_M[:, j].mean()) <= alpha))

    tag = "sampling" if sampling else "greedy"
    _write(f"headroom_{tag}_sweep.csv", sweep_rows)
    _write(f"headroom_{tag}_calibrated.csv", calib_rows)
    return {"sweep": sweep_rows, "calibrated": calib_rows}


def _write(name, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
