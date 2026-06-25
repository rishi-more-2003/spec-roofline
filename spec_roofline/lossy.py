"""spec_roofline/lossy.py — relaxed (lossy) acceptance (taskspec §9.3, §4).

The contribution. One monotone leniency knob ``gamma`` loosens the exact verify
so the target accepts more drafts -> fewer expensive target calls -> more speed,
at a bounded cost in fidelity. ``gamma = 0`` reduces *exactly* to verify.py's
oracle (asserted in tests); larger gamma accepts more (monotone), which is what
lets RCPS (conformal.py) calibrate gamma against a quality risk.

  * greedy (the calibrated path): a top-mass / nucleus-gated accept. Let
    ``m(d) = sum of target prob mass strictly above p_target(d)`` — i.e. where the
    drafted token sits in the target's own ranking. Accept d_i iff ``m(d) <= gamma``.
      - gamma = 0 -> only the argmax has zero mass above it, so this is exact
        greedy verify (lossless).
      - gamma -> 1 -> accept anything the draft proposes.
    Monotone and *smoothly biting* (confident targets still have low-mass runner-
    ups), unlike a raw prob-ratio test. The corrected token on a reject stays the
    target argmax, so the only quality drift is the accepted near-top drafts — a
    clean, bounded, monotone risk for the calibrator (taskspec §4 lists "top-p /
    entropy-gated accept" as an allowed relaxation).

  * sample: accept prob min(1, p(d) / ((1 - gamma) * q(d))); residual resample
    from norm((p - (1 - gamma) q)_+). gamma=0 -> exact Leviathan.

The risk the calibrator bounds is defined in conformal.py (token disagreement vs
the target's own greedy continuation), not here — lossy.py only owns the rule.
"""

from __future__ import annotations

import torch

from .verify import VerifyResult, verify_greedy, verify_sample, _residual_sample


@torch.no_grad()
def verify_lossy_greedy(draft_tokens: torch.Tensor, target_logits: torch.Tensor,
                        gamma: float) -> VerifyResult:
    """Top-mass / nucleus-gated greedy leniency. gamma=0 -> exact greedy verify.

    Accept d_i iff the target prob mass strictly above p_target(d_i) is <= gamma
    (the drafted token sits within the target's top-gamma nucleus)."""
    if gamma <= 0.0:
        return verify_greedy(draft_tokens, target_logits)
    K = int(draft_tokens.shape[0])
    p_all = torch.softmax(target_logits, dim=-1)        # [K+1, V]
    argmax = target_logits.argmax(dim=-1)
    flags = []
    for i in range(K):
        d = int(draft_tokens[i])
        p = p_all[i]
        pd = p[d]
        mass_above = float((p * (p > pd)).sum())        # nucleus position of d
        if mass_above <= gamma:
            flags.append(True)
        else:
            flags.append(False)
            return VerifyResult(i, int(argmax[i]), flags + [False] * (K - 1 - i), False)
    return VerifyResult(K, int(argmax[K]), flags, True)


@torch.no_grad()
def verify_lossy_sample(draft_tokens: torch.Tensor, draft_q: torch.Tensor,
                        target_logits: torch.Tensor, gamma: float,
                        gen: torch.Generator, temperature: float = 1.0) -> VerifyResult:
    """Relaxed rejection sampling. gamma=0 -> exact verify_sample."""
    if gamma <= 0.0:
        return verify_sample(draft_tokens, draft_q, target_logits, gen, temperature)
    K = int(draft_tokens.shape[0])
    scale = 1.0 - gamma
    p_all = torch.softmax(target_logits / temperature, dim=-1)
    flags = []
    for i in range(K):
        d = int(draft_tokens[i])
        p, q = p_all[i], draft_q[i]
        qd = float(q[d]) * scale
        ratio = 1.0 if qd <= 0 else min(1.0, float(p[d]) / qd)
        u = float(torch.rand(1, generator=gen, device=p.device).item())
        if u <= ratio:
            flags.append(True)
        else:
            flags.append(False)
            corr = _residual_sample(p, scale * q, gen)
            return VerifyResult(i, corr, flags + [False] * (K - 1 - i), False)
    bonus = int(torch.multinomial(p_all[K], 1, generator=gen).item())
    return VerifyResult(K, bonus, flags, True)
