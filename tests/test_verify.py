"""Exact-verify invariants (no model needed; pure tensor math)."""

import torch

from spec_roofline.verify import verify_greedy, verify_sample
from spec_roofline.lossy import verify_lossy_greedy, verify_lossy_sample


def _logits(K, V=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(K + 1, V, generator=g)


def test_greedy_accepts_argmax_run():
    tl = _logits(4)
    am = tl.argmax(-1)
    drafts = am[:4].clone()           # all drafts == target argmax
    r = verify_greedy(drafts, tl)
    assert r.n_accept == 4 and r.is_bonus
    assert r.correction == int(am[4])


def test_greedy_corrects_first_mismatch():
    tl = _logits(4, seed=1)
    am = tl.argmax(-1)
    drafts = am[:4].clone()
    drafts[2] = (drafts[2] + 1) % tl.shape[1]   # force a miss at position 2
    r = verify_greedy(drafts, tl)
    assert r.n_accept == 2
    assert r.correction == int(am[2])           # corrected to target argmax


def test_lossy_gamma0_equals_exact_greedy():
    for seed in range(5):
        tl = _logits(4, seed=seed)
        drafts = torch.randint(0, tl.shape[1], (4,), generator=torch.Generator().manual_seed(seed))
        a = verify_greedy(drafts, tl)
        b = verify_lossy_greedy(drafts, tl, gamma=0.0)
        assert (a.n_accept, a.correction) == (b.n_accept, b.correction)


def test_lossy_greedy_monotone_in_gamma():
    # acceptance count is non-decreasing in gamma for a fixed draft.
    tl = _logits(6, seed=3)
    drafts = torch.randint(0, tl.shape[1], (6,), generator=torch.Generator().manual_seed(7))
    last = -1
    for g in (0.0, 0.1, 0.3, 0.6, 0.9, 1.0):
        n = verify_lossy_greedy(drafts, tl, gamma=g).n_accept
        assert n >= last
        last = n


def test_emission_tv_zero_at_gamma0_and_monotone():
    # TV(p_gamma || p): 0 at gamma=0 (speculative-sampling identity), rises with gamma.
    from spec_roofline.lossy import emission_tv
    g = torch.Generator().manual_seed(4)
    p = torch.softmax(torch.randn(64, generator=g), 0)
    q = torch.softmax(torch.randn(64, generator=g), 0)
    assert emission_tv(p, q, 0.0) < 1e-6
    last = -1.0
    for gamma in (0.0, 0.1, 0.3, 0.6, 0.9):
        tv = emission_tv(p, q, gamma)
        assert 0.0 <= tv <= 1.0
        assert tv >= last - 1e-9
        last = tv


def test_lossy_sample_gamma0_matches_exact_distribution():
    # With a shared seed, gamma=0 lossy-sample must reproduce exact verify_sample.
    tl = _logits(4, seed=2)
    q = torch.softmax(_logits(3, seed=9)[:4], -1)   # draft dist over K slots
    drafts = torch.randint(0, tl.shape[1], (4,))
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = verify_sample(drafts, q, tl, g1)
    b = verify_lossy_sample(drafts, q, tl, 0.0, g2)
    assert (a.n_accept, a.correction) == (b.n_accept, b.correction)
