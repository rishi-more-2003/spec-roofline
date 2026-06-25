# spec-roofline

Speculative decoding for **batch-1 long-context decode**, with roofline-style
*step-cost* accounting and a **conformally-bounded "lossy" acceptance** knob.

Third in a trilogy — `decode-roofline` (weight bytes) → `kv-roofline` (KV bytes)
→ **`spec-roofline` (number of expensive steps)**. The through-line holds: batch-1
decode is HBM-bound, and the cheapest token is the one you never run the big model
for. Here the expensive unit is a **target forward**; speculation amortises it over
several accepted draft tokens.

## The idea

A small **draft** model (Qwen2.5-0.5B-Instruct) proposes `K` tokens; the **target**
(Qwen2.5-1.5B-Instruct) verifies them in one batched forward. Two regimes:

- **lossless** — the standard rejection-sampling verify (Leviathan/Chen) is
  *provably distribution-preserving*: same output distribution as the target, just
  faster. The speedup floor and the correctness oracle.
- **lossy (the contribution)** — relax acceptance with a single monotone leniency
  knob `γ` (a top-mass / nucleus-gated accept) so the target accepts more drafts →
  fewer target calls → more speed, and put an **RCPS / conformal bound on the
  resulting quality drop**. A tunable speed↔fidelity knob *with a receipt* — the
  same "compression with a warranty" idea as `kv-roofline`, applied to **time**
  instead of **memory**. `γ = 0` reduces exactly to the lossless rule.

## Correctness gating (non-negotiable, taskspec §7)

No speedup number ships unless its path proves what it claims:

- **Lossless gate** — with a fixed seed, lossless-spec greedy output equals the
  target's greedy output **token-for-token**, and the sampled token-frequency KL
  vs the target sits at the finite-sample noise floor. Tested by both the gate
  and `tests/test_engine.py` (which only passes if KV rollback is exact).
- **Lossy gate** — the realised quality loss on a held-out split is **≤ α** (RCPS
  coverage), reported as a curve.

## Install

```bash
pip install -e .          # torch (CUDA 12.x), transformers, datasets, matplotlib
```

Target + draft co-reside on an 8 GB RTX 4070 Laptop. Run benchmarks **foreground**
(background GPU jobs die on laptop sleep).

## Use

```bash
python -m spec_roofline.cli gate           # §7 lossless + lossy gates
python -m spec_roofline.cli bench          # tok/s vs context + acceptance
python -m spec_roofline.cli eval           # passkey + task accuracy + wikitext ppl
python -m spec_roofline.cli calibrate      # RCPS coverage curve (γ per α)
python -m spec_roofline.cli ablate         # K / γ / crossover sweeps + plots
python -m spec_roofline.cli generate --prompt "Explain why the sky is blue." --gamma 0.3
```

## Architecture

```
spec_roofline/
  config.py        # pinned models, K, the single γ leniency knob, contexts, RCPS α/δ
  model.py         # load target+draft, batched verify forward, sync_cache (KV rollback)
  drafter.py       # Drafter: small-model + prompt-lookup propose()         [§9.1]
  verify.py        # exact rejection-sampling verify — the oracle (greedy+sample) [§9.2]
  lossy.py         # relaxed acceptance: γ top-mass knob; γ=0 ≡ exact        [§9.3]
  conformal.py     # RCPS / empirical-Bernstein calibration of γ to risk ≤ α [§9.3]
  engine.py        # SpeculativeDecoder: draft→verify→accept loop + KV rollback
  gate.py          # §7 lossless gate (token-for-token + KL noise floor)
  data.py          # task suite, gate prompts, passkey, wikitext slice
  bench/           # throughput / acceptance / quality / ablations / plots
  cli.py
tests/             # verify & RCPS invariants (CPU) + engine rollback (GPU)
```

### The three cores (taskspec §9)

1. **Drafter** (`drafter.py`) — the 0.5B small model (autoregressive, returns draft
   distributions for the sampling verify) **and** a zero-weight prompt-lookup /
   n-gram drafter behind a flag (great on repetitive / long context).
2. **Exact verify** (`verify.py`) — the distribution-preserving accept/correct step.
   Greedy: accept the run matching the target argmax, correct the first miss.
   Sample: accept `d` w.p. `min(1, p(d)/q(d))`, resample the reject from
   `norm((p−q)₊)`. This is the oracle.
3. **Lossy + RCPS** (`lossy.py` + `conformal.py`) — accept `d` iff the target mass
   strictly above `p(d)` is `≤ γ` (γ=0 ≡ argmax ≡ lossless). The risk the warranty
   bounds is the token-disagreement rate vs the target's own greedy continuation;
   RCPS picks the largest γ whose empirical-Bernstein UCB on `E[risk] ≤ α`.

### Engine + KV rollback (the subtle systems part)

Each round runs **one** target forward over `[last_committed, d₁…d_K]` (K+1
distributions), accepts the longest valid prefix + one correction/bonus token, and
**crops both KV caches** back to the committed length (`model.sync_cache`),
discarding the rejected drafts' KV. Caches hold `committed[:-1]`; the last committed
token is re-fed so the models produce a distribution for the first drafted slot.
`n_target_forwards == n_rounds`; mean emitted-tokens-per-target-call is the lever.

## Results

See [RESULTS.md](RESULTS.md) for the full tables and `results/` for plots and CSVs.
Hero plots: tok/s vs context (lead), tok/s vs guaranteed quality-loss α, acceptance
vs K, RCPS coverage (realised ≤ α), and the honest caveat (where lossy stops
helping at short context / low acceptance).

## Non-goals

Training a draft model, multi-GPU, batched serving, or beating vLLM's spec-decode
in absolute tok/s. The contribution is the analysis + the bounded-lossy knob. The
serving home for the knob is `conformal-serve` (taskspec §10): one α budget jointly
governing reasoning length + KV precision + speculative acceptance.
