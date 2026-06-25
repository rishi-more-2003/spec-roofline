"""scripts/run_benches.py — one foreground driver for all Phase-3 benches.

Loads the two models once, then runs throughput / acceptance / quality / ablations
and the wikitext ppl, writing every CSV under results/ and a combined summary to
results/SUMMARY.txt. Single GPU run (no repeated model loads). Run foreground.

    python scripts/run_benches.py
"""

from __future__ import annotations

import sys
import time
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

    def stage(name):
        print(f">>> [{time.strftime('%H:%M:%S')}] starting: {name}", file=sys.stderr, flush=True)

    from dataclasses import replace
    from spec_roofline.config import Config, BenchConfig

    m = load_models()
    log("# spec-roofline bench summary\n")

    # Trimmed for a fast foreground run (the launch-bound Python decode loop makes
    # long contexts slow). Context range kept for the lead plot; 32k dropped.
    tcfg = replace(Config(), bench=BenchConfig(
        context_lengths=(1024, 4096, 16384), decode_steps=32, warmup_steps=4))

    stage("throughput (3 contexts x 3 regimes, 32 tok)")
    log("== throughput (tok/s vs context: no-spec / lossless / lossy γ=0.3) ==")
    for r in throughput.run(m, tcfg, lossy_gamma=LOSSY_GAMMA)["rows"]:
        tag = "OOM" if r.oom else (f"{r.tok_s:8.1f} tok/s  acc/call={r.accepted_per_call:.2f}  "
                                   f"fwds={r.target_forwards}  vram={r.peak_vram_mb:.0f}MB")
        log(f"  ctx={r.context:>6} {r.regime:9s} γ={r.gamma}: {tag}")

    stage("acceptance (2 drafters x 2 contexts x 3 K)")
    log("\n== acceptance (rate vs K / context / drafter) ==")
    for r in acceptance.run(m, ks=(2, 4, 6), contexts=(1024, 4096), n_new=48)["rows"]:
        log(f"  {r.drafter:13s} ctx={r.context:>6} K={r.k}: accept_rate={r.acceptance_rate:.3f}  "
            f"acc/call={r.accepted_per_call:.2f}")

    stage("quality (passkey + task + ppl)")
    log("\n== quality (no-spec / lossless / lossy) ==")
    for r in quality.run(m, lossy_gamma=LOSSY_GAMMA)["rows"]:
        log(f"  {r.regime:9s} γ={r.gamma}: passkey={r.passkey_acc:.2f}  task_acc={r.task_acc:.2f}  "
            f"token_agreement_vs_target={r.token_agreement:.3f}")
    ppl = quality.wikitext_ppl(m)
    log(f"  wikitext-2 ppl (target == lossless): {ppl:.3f}")

    stage("ablations (gamma + K + crossover)")
    log("\n== ablations ==")
    ab = ablations.run_all(m, gamma_ctx=2048, gamma_new=48, k_ctx=2048,
                           ks=(1, 2, 4, 6), k_new=48,
                           crossover_contexts=(256, 1024, 4096), crossover_new=48)
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
