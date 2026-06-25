"""spec_roofline/cli.py — generate / bench / eval / gate / calibrate (§5).

    python -m spec_roofline.cli gate                 # §7 lossless + lossy gates
    python -m spec_roofline.cli bench --throughput    # lead plot data
    python -m spec_roofline.cli bench --acceptance
    python -m spec_roofline.cli eval                  # passkey + task + ppl
    python -m spec_roofline.cli calibrate             # RCPS coverage curve
    python -m spec_roofline.cli ablate                # K / gamma / crossover + plots
    python -m spec_roofline.cli generate --prompt "..." [--gamma 0.3] [--mode greedy]
    python -m spec_roofline.cli all                   # gates -> benches -> plots
"""

from __future__ import annotations

import argparse

import torch

from .config import Config
from .model import load_models


def _models():
    return load_models(Config().model)


def _cmd_gate(args):
    from .gate import run_lossless_gate
    from .conformal import run_lossy_gate
    m = _models()
    print("== §7 lossless gate (distribution-preserving) ==")
    ll = run_lossless_gate(m)
    print(" ", ll)
    for c in ll.cases:
        print(f"    prompt {c['prompt']}: greedy_match={c['greedy_match']} "
              f"acc/call={c['accepted_per_call']}")
    print("== §7 lossy gate — sampling T=1.0 (the warranty's home; emission-TV risk) ==")
    lys = run_lossy_gate(m, n_cal=args.n_cal, n_test=args.n_test, n_new=args.n_new,
                         sampling=True, temperature=1.0)
    print(" ", lys)
    for r in lys.rows:
        print(f"    α={r.target_alpha:<5} γ={r.gamma:<5} realized_TV={r.realized_test_risk:<7} valid={r.valid}")
    print("== §7 lossy gate — greedy (token-disagreement risk; knob near-inert) ==")
    ly = run_lossy_gate(m, n_cal=args.n_cal, n_test=args.n_test, n_new=args.n_new)
    print(" ", ly)
    for r in ly.rows:
        print(f"    α={r.target_alpha:<5} γ={r.gamma:<5} realized={r.realized_test_risk:<7} valid={r.valid}")


def _cmd_bench(args):
    m = _models()
    do_all = not (args.throughput or args.acceptance)
    if args.throughput or do_all:
        from .bench import throughput
        print("== throughput (tok/s vs context) ==")
        for r in throughput.run(m)["rows"]:
            tag = "OOM" if r.oom else f"{r.tok_s:8.1f} tok/s  acc/call={r.accepted_per_call:.2f}"
            print(f"  ctx={r.context:>6} {r.regime:9s} γ={r.gamma}: {tag} "
                  f"(fwds={r.target_forwards}, vram={r.peak_vram_mb:.0f}MB)")
    if args.acceptance or do_all:
        from .bench import acceptance
        print("== acceptance (rate vs K / context / drafter) ==")
        for r in acceptance.run(m)["rows"]:
            print(f"  {r.drafter:13s} ctx={r.context:>6} K={r.k}: "
                  f"accept_rate={r.acceptance_rate:.3f} acc/call={r.accepted_per_call:.2f}")


def _cmd_eval(args):
    from .bench import quality
    m = _models()
    print("== quality (no-spec / lossless / lossy) ==")
    for r in quality.run(m)["rows"]:
        print(f"  {r.regime:9s} γ={r.gamma}: passkey={r.passkey_acc:.2f} "
              f"task_acc={r.task_acc:.2f} token_agreement_vs_target={r.token_agreement:.3f}")
    ppl = quality.wikitext_ppl(m)
    print(f"  wikitext-2 ppl (target == lossless): {ppl:.3f}")


def _cmd_calibrate(args):
    from .conformal import run_coverage
    from .bench import plots
    m = _models()
    print("== RCPS calibration of γ (coverage curve) ==")
    for r in run_coverage(m, n_cal=args.n_cal, n_test=args.n_test, n_new=args.n_new):
        print(f"  α={r.target_alpha:<5} -> γ={r.gamma:<5} cal_ucb={r.cal_ucb:<7} "
              f"realized_test_risk={r.realized_test_risk:<7} valid={r.valid}")
    plots.plot_all()
    print("(plots in results/)")


def _cmd_headroom(args):
    from .bench import headroom, plots
    m = _models()

    def _show(out, label):
        print(f"== headroom [{label}]: γ frontier ==")
        for r in out["sweep"]:
            print(f"  {r.regime:11s} {r.drafter:12s} γ={r.gamma:<4} "
                  f"acc/call={r.accepted_per_call:<6} risk={r.mean_risk:<7} ucb={r.risk_ucb}")
        print(f"-- RCPS-deployed γ + acc/call lift at bounded loss [{label}] --")
        for r in out["calibrated"]:
            print(f"  α={r.alpha:<5} γ={r.gamma:<4} acc/call={r.accepted_per_call:<6} "
                  f"lift={r.lift_vs_lossless}× realized={r.realized_risk} valid={r.valid}")

    # sampling is the knob's home (distribution to preserve); greedy is the contrast.
    _show(headroom.run(m, n_prompts=args.n_prompts, n_new=args.n_new,
                       sampling=True, temperature=args.temperature), f"sampling T={args.temperature}")
    if args.greedy:
        _show(headroom.run(m, n_prompts=args.n_prompts, n_new=args.n_new,
                           drafters=("model",)), "greedy")
    plots.plot_all()
    print("(plots in results/)")


def _cmd_ablate(args):
    from .bench import ablations, plots
    m = _models()
    out = ablations.run_all(m)
    print("== γ sweep (speed<->fidelity) ==")
    for r in out["gamma"]:
        print(f"  γ={r.gamma:<5} acc/call={r.accepted_per_call:<6} "
              f"tok/s={r.tok_s:<8} risk={r.risk}")
    print("== K sweep ==")
    for r in out["k"]:
        print(f"  K={r.k} acc/call={r.accepted_per_call} accept_rate={r.acceptance_rate}")
    print("== lossy-vs-context crossover (the caveat) ==")
    for r in out["crossover"]:
        print(f"  ctx={r.context:>6}: lossless {r.lossless_speedup}× lossy {r.lossy_speedup}×")
    plots.plot_all()
    print("(plots in results/)")


def _cmd_generate(args):
    m = _models()
    from .engine import SpeculativeDecoder
    dec = SpeculativeDecoder(m, Config().with_gamma(args.gamma))
    ids = m.tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True,
        return_tensors="pt").to(m.device)
    gen = torch.Generator(device=m.device).manual_seed(0) if args.mode == "sample" else None
    if args.regime == "no_spec":
        r = dec.generate_baseline(ids, args.max_new_tokens, mode=args.mode, gen=gen)
    else:
        r = dec.generate_spec(ids, args.max_new_tokens, mode=args.mode, gamma=args.gamma, gen=gen)
    print(m.tokenizer.decode(r.tokens, skip_special_tokens=True))
    print(f"\n[{args.regime} γ={args.gamma}: {r.n_generated} tok in "
          f"{r.n_target_forwards} target forwards, acc/call={r.accepted_per_call:.2f}]")


def _cmd_all(args):
    _cmd_gate(args)
    _cmd_bench(argparse.Namespace(throughput=True, acceptance=True))
    _cmd_eval(args)
    _cmd_ablate(args)


def main(argv=None):
    p = argparse.ArgumentParser(prog="spec_roofline")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate")
    g.add_argument("--n-cal", type=int, default=20, dest="n_cal")
    g.add_argument("--n-test", type=int, default=20, dest="n_test")
    g.add_argument("--n-new", type=int, default=32, dest="n_new")
    g.set_defaults(fn=_cmd_gate)

    b = sub.add_parser("bench")
    b.add_argument("--throughput", action="store_true")
    b.add_argument("--acceptance", action="store_true")
    b.set_defaults(fn=_cmd_bench)

    e = sub.add_parser("eval")
    e.set_defaults(fn=_cmd_eval)

    c = sub.add_parser("calibrate")
    c.add_argument("--n-cal", type=int, default=24, dest="n_cal")
    c.add_argument("--n-test", type=int, default=24, dest="n_test")
    c.add_argument("--n-new", type=int, default=32, dest="n_new")
    c.set_defaults(fn=_cmd_calibrate)

    a = sub.add_parser("ablate")
    a.set_defaults(fn=_cmd_ablate)

    hr = sub.add_parser("headroom")
    hr.add_argument("--n-prompts", type=int, default=50, dest="n_prompts")
    hr.add_argument("--n-new", type=int, default=32, dest="n_new")
    hr.add_argument("--temperature", type=float, default=1.0)
    hr.add_argument("--greedy", action="store_true", help="also run the greedy contrast")
    hr.set_defaults(fn=_cmd_headroom)

    gen = sub.add_parser("generate")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--gamma", type=float, default=0.0)
    gen.add_argument("--mode", default="greedy", choices=["greedy", "sample"])
    gen.add_argument("--regime", default="spec", choices=["spec", "no_spec"])
    gen.add_argument("--max-new-tokens", type=int, default=64, dest="max_new_tokens")
    gen.set_defaults(fn=_cmd_generate)

    al = sub.add_parser("all")
    al.add_argument("--n-cal", type=int, default=20, dest="n_cal")
    al.add_argument("--n-test", type=int, default=20, dest="n_test")
    al.add_argument("--n-new", type=int, default=32, dest="n_new")
    al.set_defaults(fn=_cmd_all)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
