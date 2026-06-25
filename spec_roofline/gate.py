"""spec_roofline/gate.py — §7 correctness gating (non-negotiable).

Mirrors kv-roofline's gate discipline: no throughput number ships unless the
producing path proves it preserves what it claims to.

  * Lossless gate (the oracle check). Two parts on a fixed prompt suite, fixed
    seed:
      - greedy:  lossless-spec greedy output == target greedy output, token-for-
        token. Reported as exact-match rate over the suite (must be 1.0).
      - sample:  token-frequency KL(target || lossless-spec) must sit at the
        finite-sample noise floor — i.e. within tol of KL(target || target')
        between two independent target samples of the same size. The rejection-
        sampling verify is distribution-preserving, so the only gap is sampling
        noise.
    Fails => lossless speedups are not reported.

  * Lossy gate (RCPS coverage). For each target alpha, the realised quality loss
    on a held-out split must be <= alpha (conformal.py). Reported as a curve.
    Fails => that alpha's lossy speedup is not reported.

CI ties the perf bench to both: gate first, measure second.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import torch

from .config import Config
from .engine import SpeculativeDecoder
from .data import gate_prompts


GREEDY_MUST_MATCH = 1.0      # exact-match rate required
KL_TOL = 0.05                # token-freq KL tolerance (sample path)


@dataclass
class LosslessGateResult:
    passed: bool
    greedy_match_rate: float
    sample_kl: float
    noise_floor_kl: float
    n_prompts: int
    cases: list = field(default_factory=list)

    def __str__(self):
        s = "PASS" if self.passed else "FAIL"
        return (f"[lossless gate {s}] greedy_match={self.greedy_match_rate:.3f} "
                f"(== {GREEDY_MUST_MATCH}) sample_KL={self.sample_kl:.5f} vs "
                f"noise_floor={self.noise_floor_kl:.5f} (gap < {KL_TOL}) "
                f"over {self.n_prompts} prompts")


def _kl(p_counts: Counter, q_counts: Counter) -> float:
    """Symmetric-support smoothed KL(p || q) over token frequencies."""
    keys = set(p_counts) | set(q_counts)
    tp, tq, n = sum(p_counts.values()), sum(q_counts.values()), len(keys)
    kl = 0.0
    for k in keys:
        p = (p_counts[k] + 1) / (tp + n)
        q = (q_counts[k] + 1) / (tq + n)
        kl += p * math.log(p / q)
    return kl


@torch.no_grad()
def run_lossless_gate(models, cfg: Config | None = None, n_new: int = 48,
                      n_sample_runs: int = 40, seed: int = 0) -> LosslessGateResult:
    cfg = cfg or Config()
    dec = SpeculativeDecoder(models, cfg)
    prompts = gate_prompts(models.tokenizer)
    matches, cases = 0, []
    # --- greedy: token-for-token equality vs target greedy ---
    for i, ids in enumerate(prompts):
        ids = ids.to(models.device)
        base = dec.generate_baseline(ids, n_new, mode="greedy")
        spec = dec.generate_spec(ids, n_new, mode="greedy", gamma=0.0)
        ok = base.tokens == spec.tokens
        matches += int(ok)
        cases.append({"prompt": i, "greedy_match": ok,
                      "accepted_per_call": round(spec.accepted_per_call, 3)})
    greedy_rate = matches / len(prompts)

    # --- sample: token-freq KL(target || spec) vs the noise floor KL(target || target') ---
    # Two independent target sample sets (A, B) give the finite-sample noise floor;
    # the spec set (S) is distribution-preserving iff KL(A||S) ~ KL(A||B).
    baseA, baseB, spec_c = Counter(), Counter(), Counter()
    ids0 = prompts[0].to(models.device)
    for s in range(n_sample_runs):
        gA = torch.Generator(device=models.device).manual_seed(seed + s)
        gB = torch.Generator(device=models.device).manual_seed(seed + 10_000 + s)
        gS = torch.Generator(device=models.device).manual_seed(seed + 20_000 + s)
        baseA.update(dec.generate_baseline(ids0, 16, mode="sample", gen=gA).tokens)
        baseB.update(dec.generate_baseline(ids0, 16, mode="sample", gen=gB).tokens)
        spec_c.update(dec.generate_spec(ids0, 16, mode="sample", gamma=0.0, gen=gS).tokens)
    floor = _kl(baseA, baseB)
    kl = _kl(baseA, spec_c)

    passed = (greedy_rate >= GREEDY_MUST_MATCH) and (kl <= floor + KL_TOL)
    return LosslessGateResult(passed, greedy_rate, kl, floor, len(prompts), cases)
