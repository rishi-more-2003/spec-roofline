"""spec_roofline/verify.py — the exact rejection-sampling verify (taskspec §9.2).

THE ORACLE. Given the target's distributions over the K drafted positions and the
draft's own distributions, accept the longest valid prefix and correct the first
rejected token so the realised output distribution **exactly equals the target's**
(Leviathan et al. 2023 / Chen et al. 2023). This is provably distribution-
preserving — the speedup floor and the correctness reference for everything else.

Two regimes, both exact:

  * greedy  — accept d_i iff it is the target's argmax at slot i; the first
    mismatch is replaced by the target argmax. With a fixed seed this reproduces
    the target's greedy decode token-for-token (the §7 lossless gate).
  * sample  — the rejection-sampling rule: accept d_i with probability
    min(1, p_i(d_i)/q_i(d_i)); on the first rejection, resample the corrected
    token from the normalised residual (p_i - q_i)_+. The accepted+corrected
    stream is distributed exactly as the target (the §7 KL gate, ~0).

Layout of ``target_logits`` (shape [K+1, V]): the distribution that token d_i
(1-indexed) is tested against is ``softmax(target_logits[i-1])``; row K is the
bonus distribution used when all K drafts are accepted.

lossy.py reuses this contract with a single leniency knob; gamma=0 there calls
straight back into the exact rule here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VerifyResult:
    n_accept: int                 # number of leading draft tokens accepted (0..K)
    correction: int               # the appended token: corrected-reject OR bonus
    accept_flags: list            # per-draft accept booleans (len K), for telemetry
    is_bonus: bool                # True if correction is the all-accepted bonus token


def _residual_sample(p: torch.Tensor, q: torch.Tensor, gen: torch.Generator) -> int:
    """Sample from norm((p - q)_+); the distribution-preserving correction."""
    resid = torch.clamp(p - q, min=0.0)
    s = resid.sum()
    if s <= 0:                    # numerically degenerate -> fall back to p
        resid, s = p, p.sum()
    probs = resid / s
    return int(torch.multinomial(probs, 1, generator=gen).item())


@torch.no_grad()
def verify_greedy(draft_tokens: torch.Tensor, target_logits: torch.Tensor) -> VerifyResult:
    """Exact greedy verify. q (draft dist) is irrelevant — only token identity is.

    Equivalent to target greedy decode: accept the run of drafts that match the
    target argmax; correct the first miss with that argmax; if all match, append
    the bonus argmax. Deterministic — the lossless token-for-token gate.
    """
    K = int(draft_tokens.shape[0])
    tgt_argmax = target_logits.argmax(dim=-1)        # [K+1]
    flags = []
    for i in range(K):
        if int(draft_tokens[i]) == int(tgt_argmax[i]):
            flags.append(True)
        else:
            flags.append(False)
            return VerifyResult(i, int(tgt_argmax[i]), flags + [False] * (K - 1 - i), False)
    return VerifyResult(K, int(tgt_argmax[K]), flags, True)


@torch.no_grad()
def verify_sample(draft_tokens: torch.Tensor, draft_q: torch.Tensor,
                  target_logits: torch.Tensor, gen: torch.Generator,
                  temperature: float = 1.0) -> VerifyResult:
    """Exact rejection-sampling verify (Leviathan/Chen). Distribution-preserving.

    ``draft_q`` is [K, V] (draft probabilities at each slot). Accept d_i with prob
    min(1, p_i(d_i)/q_i(d_i)); first rejection -> resample from (p_i - q_i)_+.
    """
    K = int(draft_tokens.shape[0])
    p_all = torch.softmax(target_logits / temperature, dim=-1)    # [K+1, V]
    flags = []
    for i in range(K):
        d = int(draft_tokens[i])
        p = p_all[i]
        q = draft_q[i]
        pd, qd = float(p[d]), float(q[d])
        ratio = 1.0 if qd <= 0 else min(1.0, pd / qd)
        u = float(torch.rand(1, generator=gen, device=p.device).item())
        if u <= ratio:
            flags.append(True)
        else:
            flags.append(False)
            corr = _residual_sample(p, q, gen)
            return VerifyResult(i, corr, flags + [False] * (K - 1 - i), False)
    # all accepted -> bonus token from the target's next-position distribution.
    bonus = int(torch.multinomial(p_all[K], 1, generator=gen).item())
    return VerifyResult(K, bonus, flags, True)
