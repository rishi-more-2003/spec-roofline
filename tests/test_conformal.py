"""RCPS calibrator invariants (no model; synthetic risk matrices)."""

import torch

from spec_roofline.conformal import _eb_ucb, rcps_calibrate


def test_eb_ucb_bounds_mean():
    r = torch.full((200,), 0.1)
    ucb = _eb_ucb(r, delta=0.1)
    assert ucb >= 0.1                      # UCB is above the mean
    assert ucb < 0.2                       # but tight for low variance


def test_rcps_picks_largest_valid_gamma():
    gammas = [0.0, 0.1, 0.2, 0.3, 0.5]
    # risk increases with gamma; constant columns + large n so the EB finite-
    # sample term is negligible and the pick is driven by the mean.
    cols = [0.0, 0.02, 0.05, 0.2, 0.4]
    M = torch.stack([torch.full((5000,), c) for c in cols], dim=1)
    g, ucb = rcps_calibrate(M, gammas, alpha=0.1, delta=0.1)
    # at alpha=0.1, gamma=0.2 (risk 0.05) is the largest whose UCB stays <= 0.1.
    assert g == 0.2
    assert ucb <= 0.1


def test_rcps_falls_back_to_zero_when_nothing_valid():
    gammas = [0.0, 0.5]
    M = torch.stack([torch.zeros(50), torch.full((50,), 0.9)], dim=1)
    g, ucb = rcps_calibrate(M, gammas, alpha=0.01, delta=0.1)
    assert g == 0.0                        # only the exact knob certifies
