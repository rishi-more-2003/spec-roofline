"""spec_roofline/bench/plots.py — the four README hero plots (taskspec §0).

Reads results/*.csv (whatever exists) and writes PNGs:
  1. throughput.png        — tok/s vs context: no-spec / lossless / lossy (LEAD).
  2. tok_s_vs_loss.png     — tok/s vs guaranteed max quality loss alpha (the knob).
  3. acceptance.png        — acceptance rate vs K, per drafter.
  4. coverage.png          — realized loss vs target alpha (RCPS guarantee, y<=x).
  5. crossover.png         — the honest caveat: where lossy stops helping.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import RESULTS


def _read(name):
    p = RESULTS / name
    if not p.exists():
        return None
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def plot_throughput():
    rows = _read("throughput.csv")
    if not rows:
        return
    by = defaultdict(lambda: ([], []))
    for r in rows:
        if r.get("oom") == "True" or _f(r["tok_s"]) <= 0:
            continue
        ctx, ts = _f(r["context"]), _f(r["tok_s"])
        by[r["regime"]][0].append(ctx)
        by[r["regime"]][1].append(ts)
    plt.figure(figsize=(7, 4.5))
    labels = {"no_spec": "no-spec (target only)", "lossless": "lossless-spec",
              "lossy": "lossy-spec (γ)"}
    for regime in ("no_spec", "lossless", "lossy"):
        if regime in by:
            xs, ys = by[regime]
            plt.plot(xs, ys, "o-", label=labels[regime])
    plt.xscale("log", base=2)
    plt.xlabel("context length (tokens)")
    plt.ylabel("decode tok/s")
    plt.title("Decode throughput vs context: no-spec / lossless / lossy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "throughput.png", dpi=130)
    plt.close()


def plot_tok_s_vs_loss():
    cov = _read("lossy_coverage.csv")
    gam = _read("ablation_gamma.csv")
    if not cov or not gam:
        return
    g2ts = {round(_f(r["gamma"]), 4): _f(r["tok_s"]) for r in gam}
    xs, ys, gs = [], [], []
    for r in cov:
        a, g = _f(r["target_alpha"]), round(_f(r["gamma"]), 4)
        if g in g2ts:
            xs.append(a)
            ys.append(g2ts[g])
            gs.append(g)
    if not xs:
        return
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, "o-", color="C3")
    for a, y, g in zip(xs, ys, gs):
        plt.annotate(f"γ={g}", (a, y), textcoords="offset points", xytext=(4, 6), fontsize=8)
    plt.xlabel("guaranteed max quality loss α (RCPS, held-out ≤ α)")
    plt.ylabel("decode tok/s at calibrated γ")
    plt.title("Throughput vs guaranteed quality loss (the lossy knob)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "tok_s_vs_loss.png", dpi=130)
    plt.close()


def plot_acceptance():
    rows = _read("acceptance.csv")
    if not rows:
        return
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["drafter"]][_f(r["context"])].append((_f(r["k"]), _f(r["acceptance_rate"])))
    plt.figure(figsize=(7, 4.5))
    for drafter, ctxs in by.items():
        # plot the largest context for clarity
        ctx = max(ctxs)
        pts = sorted(ctxs[ctx])
        plt.plot([k for k, _ in pts], [a for _, a in pts], "o-",
                 label=f"{drafter} (ctx={int(ctx)})")
    plt.xlabel("draft length K")
    plt.ylabel("acceptance rate")
    plt.title("Acceptance rate vs draft size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "acceptance.png", dpi=130)
    plt.close()


def plot_coverage():
    rows = _read("lossy_coverage.csv")
    if not rows:
        return
    xs = [_f(r["target_alpha"]) for r in rows]
    ys = [_f(r["realized_test_risk"]) for r in rows]
    plt.figure(figsize=(6, 5))
    lim = max(xs) * 1.1
    plt.plot([0, lim], [0, lim], "--", color="gray", label="realized = α (boundary)")
    plt.scatter(xs, ys, color="C2", zorder=3, label="realized test risk")
    plt.fill_between([0, lim], [0, 0], [lim, lim], color="C2", alpha=0.06)
    plt.xlabel("target α (guaranteed max loss)")
    plt.ylabel("realized loss on held-out test")
    plt.title("RCPS coverage: realized ≤ α (the warranty holds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "coverage.png", dpi=130)
    plt.close()


def plot_crossover():
    rows = _read("ablation_crossover.csv")
    if not rows:
        return
    xs = [_f(r["context"]) for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, [_f(r["lossless_speedup"]) for r in rows], "o-", label="lossless speedup")
    plt.plot(xs, [_f(r["lossy_speedup"]) for r in rows], "s-", label="lossy speedup")
    plt.axhline(1.0, color="gray", ls="--", label="no-spec (1×)")
    plt.xscale("log", base=2)
    plt.xlabel("context length (tokens)")
    plt.ylabel("speedup vs no-spec")
    plt.title("Where speculation helps (and the honest caveat)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "crossover.png", dpi=130)
    plt.close()


def plot_headroom():
    rows = _read("headroom_sweep.csv")
    if not rows:
        return
    # model drafter frontier: acc/call (left) + risk (right) vs gamma.
    m = [r for r in rows if r["drafter"] == "model"]
    if not m:
        return
    m = sorted(m, key=lambda r: _f(r["gamma"]))
    g = [_f(r["gamma"]) for r in m]
    acc = [_f(r["accepted_per_call"]) for r in m]
    risk = [_f(r["mean_risk"]) for r in m]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(g, acc, "o-", color="C0", label="acc/call (speed)")
    ax1.set_xlabel("leniency γ")
    ax1.set_ylabel("accepted tokens / target call", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.plot(g, risk, "s--", color="C3", label="risk (quality loss)")
    ax2.set_ylabel("token-disagreement risk vs target", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax1.set_title("The lossy frontier: speed rises, but the risk rises faster")
    fig.tight_layout()
    fig.savefig(RESULTS / "headroom.png", dpi=130)
    plt.close(fig)


def plot_all():
    plot_throughput()
    plot_headroom()
    plot_tok_s_vs_loss()
    plot_acceptance()
    plot_coverage()
    plot_crossover()
