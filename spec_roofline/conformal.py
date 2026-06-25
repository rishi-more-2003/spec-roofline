"""spec_roofline/conformal.py — RCPS calibration of the leniency knob (§9.3, §7).

Ports kv-roofline's risk-control discipline from *memory bytes* to *acceptance
leniency*. Same machinery (RCPS / Bates et al. 2021 with an empirical-Bernstein
UCB, reused from kv_roofline.adaptive.conformal and conformal-serve's UCB search):
pick the most aggressive knob whose **expected quality loss is provably <= alpha
with confidence 1 - delta**, distribution-free, from a calibration set; then show
the realised loss on a held-out test set sits <= alpha (the §7 lossy coverage gate).

The knob (lossy.py): gamma, the top-mass acceptance leniency. Quality loss is
*monotone non-decreasing* in gamma (more leniency -> more drift), so scanning
gamma descending returns the most aggressive valid setting.

THE RISK METRIC (ours to define, taskspec §9.3): for a prompt, the reference is
the **lossless-spec** greedy continuation (gamma=0) — which the §7 lossless gate
independently certifies is token-for-token equal to the *target's* greedy output.
The lossy run's risk is the token-disagreement rate against that reference over the
generated window — in [0,1], structurally 0 at gamma=0 (same code path), monotone
in gamma. Referencing the lossless-spec path (not a separate sequential baseline)
keeps gamma=0 at exactly zero risk; a sequential baseline reintroduces a spurious
floor because batched-verify vs one-token-at-a-time forwards can tie-break greedy
argmax differently, and greedy decode cascades a single early difference. The
warranty bounds "how far did relaxing acceptance pull us off the lossless path",
which (by the gate) is the target's path.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)

from .config import Config

# engine/data are imported lazily inside run_coverage so the pure calibrator math
# (_eb_ucb, rcps_calibrate) imports with torch only — keeps CI GPU-/transformers-free.

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _eb_ucb(risks: torch.Tensor, delta: float) -> float:
    """Empirical-Bernstein UCB on E[risk] (Maurer & Pontil 2009), as in
    kv-roofline. Tighter than Hoeffding when risk variance is low — which it is
    here (small gamma rarely disagrees), letting RCPS certify aggressive leniency
    without a huge calibration set. Risk in [0,1]."""
    n = risks.numel()
    mean = float(risks.mean())
    var = float(risks.var(unbiased=True)) if n > 1 else 0.0
    t = math.log(2.0 / delta)
    return float(mean + math.sqrt(2.0 * var * t / n) + 7.0 * t / (3.0 * (n - 1)))


@torch.no_grad()
def _reference(dec: SpeculativeDecoder, ids: torch.Tensor, n_new: int) -> list:
    """Lossless-spec greedy continuation — the per-prompt quality reference.

    Uses the gamma=0 spec path (== target greedy, per the §7 lossless gate) so the
    risk is structurally 0 at gamma=0 and free of batched-vs-sequential argmax
    tie-break noise."""
    return dec.generate_spec(ids, n_new, mode="greedy", gamma=0.0).tokens


@torch.no_grad()
def risk(dec: SpeculativeDecoder, ids: torch.Tensor, ref: list, gamma: float,
         n_new: int) -> float:
    """Token-disagreement rate of lossy-spec(gamma) vs the target greedy ref."""
    out = dec.generate_spec(ids, n_new, mode="greedy", gamma=gamma).tokens
    L = min(len(out), len(ref))
    if L == 0:
        return 0.0
    mism = sum(1 for i in range(L) if out[i] != ref[i])
    # length mismatch (early eos divergence) counts against the shorter tail.
    mism += abs(len(out) - len(ref))
    return max(0.0, min(1.0, mism / max(len(out), len(ref))))


def _risk_matrix(dec, prompts, refs, gammas, n_new, label=""):
    """[n_prompts, n_gammas] risk tensor, with live progress."""
    n = len(prompts)
    M = torch.zeros(n, len(gammas))
    t0 = time.perf_counter()
    for i, (ids, ref) in enumerate(zip(prompts, refs)):
        for j, g in enumerate(gammas):
            M[i, j] = risk(dec, ids, ref, g, n_new)
        done = i + 1
        rate = (time.perf_counter() - t0) / done
        eta = rate * (n - done)
        bar = "#" * int(20 * done / n) + "-" * (20 - int(20 * done / n))
        _log(f"  [{label}] [{bar}] {done}/{n} prompts  "
             f"({rate:.1f}s/prompt, ETA {eta:4.0f}s)")
    return M


def rcps_calibrate(cal_risks: torch.Tensor, gammas, alpha: float, delta: float):
    """Largest gamma whose empirical-Bernstein UCB on E[risk] <= alpha.

    Risk increases with gamma, so scan descending and return the first (largest)
    gamma that certifies. gamma=0 (exact) always certifies (risk==0)."""
    order = sorted(range(len(gammas)), key=lambda j: gammas[j], reverse=True)
    for j in order:
        ucb = _eb_ucb(cal_risks[:, j], delta)
        if ucb <= alpha:
            return gammas[j], ucb
    return 0.0, 0.0


@dataclass
class CoverageRow:
    target_alpha: float
    gamma: float
    cal_ucb: float
    realized_test_risk: float
    valid: bool


@torch.no_grad()
def run_coverage(models, cfg: Config | None = None, n_cal: int = 20, n_test: int = 20,
                 n_new: int = 32, seed: int = 0) -> list:
    """RCPS coverage experiment (the §7 lossy gate, reported as a curve).

    For each target alpha: calibrate gamma on the cal split, then report the
    realised mean risk on the held-out test split. valid <=> realised <= alpha.
    """
    from .engine import SpeculativeDecoder
    from .data import calibration_prompts
    cfg = cfg or Config()
    dec = SpeculativeDecoder(models, cfg)
    gammas = list(cfg.lossy.gamma_grid)
    cal, test = calibration_prompts(models.tokenizer, n_cal, n_test, seed)
    cal = [x.to(models.device) for x in cal]
    test = [x.to(models.device) for x in test]

    _log(f"[coverage] computing references ({len(cal)} cal + {len(test)} test)...")
    cal_refs = [_reference(dec, x, n_new) for x in cal]
    test_refs = [_reference(dec, x, n_new) for x in test]
    _log(f"[coverage] cal risk matrix ({len(cal)}x{len(gammas)})...")
    cal_M = _risk_matrix(dec, cal, cal_refs, gammas, n_new, label="cal")
    _log(f"[coverage] test risk matrix ({len(test)}x{len(gammas)})...")
    test_M = _risk_matrix(dec, test, test_refs, gammas, n_new, label="test")
    gidx = {g: j for j, g in enumerate(gammas)}

    rows = []
    for alpha in cfg.lossy.alphas:
        g, ucb = rcps_calibrate(cal_M, gammas, alpha, cfg.lossy.delta)
        realized = float(test_M[:, gidx[g]].mean())
        rows.append(CoverageRow(alpha, g, round(ucb, 4), round(realized, 4),
                                realized <= alpha))
    _write(rows)
    return rows


@dataclass
class LossyGateResult:
    passed: bool
    rows: list

    def __str__(self):
        s = "PASS" if self.passed else "FAIL"
        n_ok = sum(r.valid for r in self.rows)
        return (f"[lossy gate {s}] {n_ok}/{len(self.rows)} alphas covered "
                f"(realized risk <= alpha on held-out test)")


def run_lossy_gate(models, cfg: Config | None = None, **kw) -> LossyGateResult:
    rows = run_coverage(models, cfg, **kw)
    return LossyGateResult(all(r.valid for r in rows), rows)


def _write(rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "lossy_coverage.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target_alpha", "gamma", "cal_ucb", "realized_test_risk", "valid"])
        for r in rows:
            w.writerow([r.target_alpha, r.gamma, r.cal_ucb, r.realized_test_risk, r.valid])
    return path
