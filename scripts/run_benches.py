"""scripts/run_benches.py — one foreground driver for all Phase-3 benches.

Loads the two models once, then runs throughput / acceptance / quality / ablations
and the wikitext ppl, writing every CSV under results/ and a combined summary to
results/SUMMARY.txt. Single GPU run (no repeated model loads). Run foreground.

    python scripts/run_benches.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spec_roofline.model import load_models
from spec_roofline.bench import throughput, acceptance, quality, ablations, plots
from spec_roofline.bench.common import RESULTS

LOSSY_GAMMA = 0.3


def main():
    out = []

    def log(s=""):
        print(s, flush=True)
        out.append(s)

    m = load_models()
    log("# spec-roofline bench summary\n")

    log("== throughput (tok/s vs context: no-spec / lossless / lossy γ=0.3) ==")
    for r in throughput.run(m, lossy_gamma=LOSSY_GAMMA)["rows"]:
        tag = "OOM" if r.oom else (f"{r.tok_s:8.1f} tok/s  acc/call={r.accepted_per_call:.2f}  "
                                   f"fwds={r.target_forwards}  vram={r.peak_vram_mb:.0f}MB")
        log(f"  ctx={r.context:>6} {r.regime:9s} γ={r.gamma}: {tag}")

    log("\n== acceptance (rate vs K / context / drafter) ==")
    for r in acceptance.run(m)["rows"]:
        log(f"  {r.drafter:13s} ctx={r.context:>6} K={r.k}: accept_rate={r.acceptance_rate:.3f}  "
            f"acc/call={r.accepted_per_call:.2f}")

    log("\n== quality (no-spec / lossless / lossy) ==")
    for r in quality.run(m, lossy_gamma=LOSSY_GAMMA)["rows"]:
        log(f"  {r.regime:9s} γ={r.gamma}: passkey={r.passkey_acc:.2f}  task_acc={r.task_acc:.2f}  "
            f"token_agreement_vs_target={r.token_agreement:.3f}")
    ppl = quality.wikitext_ppl(m)
    log(f"  wikitext-2 ppl (target == lossless): {ppl:.3f}")

    log("\n== ablations ==")
    ab = ablations.run_all(m)
    log("  -- γ sweep (speed<->fidelity) --")
    for r in ab["gamma"]:
        log(f"    γ={r.gamma:<5} acc/call={r.accepted_per_call:<6} tok/s={r.tok_s:<8} risk={r.risk}")
    log("  -- K sweep --")
    for r in ab["k"]:
        log(f"    K={r.k} acc/call={r.accepted_per_call} accept_rate={r.acceptance_rate}")
    log("  -- lossy-vs-context crossover (the caveat) --")
    for r in ab["crossover"]:
        log(f"    ctx={r.context:>6}: lossless {r.lossless_speedup}x  lossy {r.lossy_speedup}x")

    plots.plot_all()
    log("\n(plots + CSVs written to results/)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "SUMMARY.txt").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
