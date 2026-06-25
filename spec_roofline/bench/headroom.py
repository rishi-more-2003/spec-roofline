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
from ..data import calibration_prompts
from .common import RESULTS


def _risk(out, ref):
    L = min(len(out), len(ref))
    mism = sum(1 for i in range(L) if out[i] != ref[i]) + abs(len(out) - len(ref))
    return max(0.0, min(1.0, mism / max(1, len(out), len(ref))))


@dataclass
class HeadroomRow:
    drafter: str
    gamma: float
    accepted_per_call: float
    mean_risk: float
    risk_ucb: float


@dataclass
class CalibratedRow:
    drafter: str
    alpha: float
    gamma: float
    accepted_per_call: float     # at the calibrated gamma
    lift_vs_lossless: float      # acc/call(gamma) / acc/call(gamma=0)
    realized_risk: float
    valid: bool


@torch.no_grad()
def run(models, cfg: Config | None = None, n_prompts: int = 30, n_new: int = 32,
        drafters=("model", "prompt_lookup"), seed: int = 0) -> dict:
    cfg = cfg or Config()
    gammas = list(cfg.lossy.gamma_grid)
    delta = cfg.lossy.delta
    prompts, _ = calibration_prompts(models.tokenizer, n_prompts, 0, seed)
    prompts = [p.to(models.device) for p in prompts]

    import sys
    sweep_rows, calib_rows = [], []
    for drafter in drafters:
        dec = SpeculativeDecoder(models, cfg.with_method(drafter))
        # gamma=0 reference (== target greedy, gated) + risk/accept matrices.
        refs = [dec.generate_spec(p, n_new, mode="greedy", gamma=0.0).tokens for p in prompts]
        risk_M = torch.zeros(n_prompts, len(gammas))
        acc_M = torch.zeros(n_prompts, len(gammas))
        for i, (p, ref) in enumerate(zip(prompts, refs)):
            for j, g in enumerate(gammas):
                r = dec.generate_spec(p, n_new, mode="greedy", gamma=g)
                risk_M[i, j] = _risk(r.tokens, ref)
                acc_M[i, j] = r.accepted_per_call
            print(f"  [headroom/{drafter}] {i+1}/{n_prompts} prompts", file=sys.stderr, flush=True)
        accpc = acc_M.mean(0)
        for j, g in enumerate(gammas):
            sweep_rows.append(HeadroomRow(drafter, g, round(float(accpc[j]), 3),
                                          round(float(risk_M[:, j].mean()), 4),
                                          round(_eb_ucb(risk_M[:, j], delta), 4)))
        # RCPS: calibrated gamma per alpha + the acc/call lift it buys.
        base_acc = float(accpc[gammas.index(0.0)]) if 0.0 in gammas else float(accpc[0])
        gidx = {g: j for j, g in enumerate(gammas)}
        for alpha in cfg.lossy.alphas:
            g, _ = rcps_calibrate(risk_M, gammas, alpha, delta)
            j = gidx[g]
            calib_rows.append(CalibratedRow(
                drafter, alpha, g, round(float(accpc[j]), 3),
                round(float(accpc[j]) / max(1e-9, base_acc), 3),
                round(float(risk_M[:, j].mean()), 4),
                float(risk_M[:, j].mean()) <= alpha))

    _write("headroom_sweep.csv", sweep_rows)
    _write("headroom_calibrated.csv", calib_rows)
    return {"sweep": sweep_rows, "calibrated": calib_rows}


def _write(name, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
