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
    # plot both regimes if present; sampling (the warranty's home) leads.
    series = []
    samp = _read("lossy_coverage_sampling.csv")
    greedy = _read("lossy_coverage.csv")
    if samp:
        series.append(("sampling (held-out, emission TV)", samp, "C2", "o"))
    if greedy:
        series.append(("greedy (held-out, token-disagree)", greedy, "C0", "s"))
    if not series:
        return
    allx = [_f(r["target_alpha"]) for _, rows, _, _ in series for r in rows]
    lim = max(allx) * 1.1
    plt.figure(figsize=(6, 5))
    plt.plot([0, lim], [0, lim], "--", color="gray", label="realized = α (boundary)")
    plt.fill_between([0, lim], [0, 0], [lim, lim], color="C2", alpha=0.06)
    for label, rows, c, mk in series:
        xs = [_f(r["target_alpha"]) for r in rows]
        ys = [_f(r["realized_test_risk"]) for r in rows]
        plt.scatter(xs, ys, color=c, marker=mk, zorder=3, label=label)
    plt.xlabel("target α (guaranteed max loss)")
    plt.ylabel("realized loss on held-out test")
    plt.title("RCPS coverage: realized ≤ α (the warranty holds, both regimes)")
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


def _headroom_panel(ax, rows, title, risk_label):
    m = sorted([r for r in rows if r["drafter"] == "model"], key=lambda r: _f(r["gamma"]))
    if not m:
        return None
    g = [_f(r["gamma"]) for r in m]
    acc = [_f(r["accepted_per_call"]) for r in m]
    risk = [_f(r["mean_risk"]) for r in m]
    ax.plot(g, acc, "o-", color="C0", label="acc/call")
    ax.set_xlabel("leniency γ")
    ax.set_ylabel("accepted tokens / target call", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2 = ax.twinx()
    ax2.plot(g, risk, "s--", color="C3", label="risk")
    ax2.set_ylabel(risk_label, color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax.set_title(title)
    return ax2


def plot_headroom():
    greedy = _read("headroom_greedy_sweep.csv")
    samp = _read("headroom_sampling_sweep.csv")
    panels = [(samp, "Sampling (T): the knob's home", "emission TV vs target"),
              (greedy, "Greedy: the knob barely bites", "token-disagreement vs target")]
    panels = [(r, t, rl) for r, t, rl in panels if r]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (rows, title, rl) in zip(axes, panels):
        _headroom_panel(ax, rows, title, rl)
    fig.suptitle("Where the lossy knob buys speed: sampling vs greedy")
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
