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

THE RISK METRIC (ours to define, taskspec §9.3) — two regimes:

  * sampling (the knob's home): risk = the per-step **emission TV(p_gamma || p)**
    (lossy.emission_tv), how far relaxed acceptance moves the realised next-token
    distribution from the target's true one. A *distributional* quantity — 0 at
    gamma=0 by the speculative-sampling identity, low-variance, in [0,1]. This is
    what the distribution-free warranty is *for*; under sampling the knob is active
    and RCPS certifies meaningfully higher gamma at bounded alpha.

  * greedy: risk = token-disagreement vs the **lossless-spec** greedy continuation
    (gamma=0) — which the §7 lossless gate certifies equals the target's greedy
    output. Referencing the lossless-spec path (not a sequential baseline) keeps
    gamma=0 at exactly zero risk (no batched-vs-sequential argmax tie-break floor).
    Under greedy the target is a point mass, so the knob barely bites — a useful
    contrast, not the headline.
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


def _risk_matrix_sample(dec, prompts, gammas, n_new, temperature, device, seed, label=""):
    """Sampling-regime risk matrix: risk = mean per-step emission TV(p_gamma||p), the
    *distributional* loss (conformal's natural home — there is a distribution to
    preserve). gamma=0 -> TV 0 by the speculative-sampling identity."""
    n = len(prompts)
    M = torch.zeros(n, len(gammas))
    t0 = time.perf_counter()
    for i, ids in enumerate(prompts):
        for j, g in enumerate(gammas):
            gen = torch.Generator(device=device).manual_seed(seed + 1000 + i)
            M[i, j] = dec.generate_spec(ids, n_new, mode="sample", gamma=g, gen=gen).risk_tv
        done = i + 1
        rate = (time.perf_counter() - t0) / done
        bar = "#" * int(20 * done / n) + "-" * (20 - int(20 * done / n))
        _log(f"  [{label}] [{bar}] {done}/{n}  ({rate:.1f}s/prompt, ETA {rate*(n-done):4.0f}s)")
    return M


@torch.no_grad()
def run_coverage(models, cfg: Config | None = None, n_cal: int = 20, n_test: int = 20,
                 n_new: int = 32, seed: int = 0, sampling: bool = False,
                 temperature: float = 1.0) -> list:
    """RCPS coverage experiment (the §7 lossy gate, reported as a curve).

    For each target alpha: calibrate gamma on the cal split, then report the
    realised mean risk on the held-out test split. valid <=> realised <= alpha.

    Two regimes: greedy (risk = token-disagreement vs the lossless path) and
    *sampling* (risk = distributional emission TV vs the target) — the latter is
    where the distribution-preserving verify + distribution-free warranty bite.
    """
    from .engine import SpeculativeDecoder
    from .data import calibration_prompts, open_ended_prompts
    from dataclasses import replace as _replace
    cfg = cfg or Config()
    gammas = list(cfg.lossy.gamma_grid)
    dev = models.device

    if sampling:
        cfg = _replace(cfg, draft=_replace(cfg.draft, temperature=temperature))
        dec = SpeculativeDecoder(models, cfg)
        cal = [x.to(dev) for x in open_ended_prompts(models.tokenizer, n_cal, seed)]
        test = [x.to(dev) for x in open_ended_prompts(models.tokenizer, n_test, seed + 7)]
        _log(f"[coverage/sample T={temperature}] cal matrix ({n_cal}x{len(gammas)})...")
        cal_M = _risk_matrix_sample(dec, cal, gammas, n_new, temperature, dev, seed, "cal")
        _log(f"[coverage/sample] test matrix ({n_test}x{len(gammas)})...")
        test_M = _risk_matrix_sample(dec, test, gammas, n_new, temperature, dev, seed + 7, "test")
    else:
        dec = SpeculativeDecoder(models, cfg)
        cal, test = calibration_prompts(models.tokenizer, n_cal, n_test, seed)
        cal = [x.to(dev) for x in cal]
        test = [x.to(dev) for x in test]
        _log(f"[coverage] references ({len(cal)} cal + {len(test)} test)...")
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
    _write(rows, "lossy_coverage_sampling.csv" if sampling else "lossy_coverage.csv")
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


def _write(rows, name="lossy_coverage.csv"):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target_alpha", "gamma", "cal_ucb", "realized_test_risk", "valid"])
        for r in rows:
            w.writerow([r.target_alpha, r.gamma, r.cal_ucb, r.realized_test_risk, r.valid])
    return path
