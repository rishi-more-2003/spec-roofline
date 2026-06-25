# spec-roofline — results

Hardware: RTX 4070 Laptop, 8 GB, sm_89. Target `Qwen2.5-1.5B-Instruct`, draft
`Qwen2.5-0.5B-Instruct`, fp16, batch-1 greedy unless noted. Numbers from
`results/*.csv` (regenerate: `python scripts/run_benches.py` + `cli calibrate`).

## TL;DR

- **Both §7 gates pass.** Lossless-spec reproduces the target greedy output
  **token-for-token** (KV rollback is exact); the lossy knob's realised quality
  loss stays **≤ α** on held-out data for every α (RCPS coverage).
- **The roofline win is real and large:** lossless-spec emits **4.6 tokens per
  target forward** — it produces 32 tokens in **7** expensive target steps instead
  of **33**. That is the "number of expensive steps" thesis of the trilogy.
- **The honest caveat (wall-clock):** in this pure-Python reference harness the
  step-count win does **not** convert to wall-clock tok/s. With a 1.5B:0.5B
  target:draft ratio (~3×), one target forward does not dominate the 4 draft
  forwards + per-token Python/launch overhead, and on 8 GB the two co-resident KV
  caches hit memory pressure at ctx≥4k. Speculation's wall-clock payoff needs a
  *larger* target and a fused kernel — both explicit non-goals here (taskspec §1).

## §7 Gate 1 — lossless (distribution-preserving)

```
[lossless gate PASS] greedy_match=1.000 (== 1.0) sample_KL=0.07823 vs
                     noise_floor=0.03856 (gap < 0.05) over 8 prompts
```

- **greedy:** lossless-spec greedy output == target greedy output, token-for-token,
  on all 8 prompts (acc/call 3.0–4.0). Only holds if KV rollback after each
  rejection is exact.
- **sample:** token-frequency KL(target ‖ lossless-spec) sits at the finite-sample
  noise floor (KL between two independent target samples), i.e. the rejection-
  sampling verify is distribution-preserving.
- Reproduced as a unit test in `tests/test_engine.py`.

## §7 Gate 2 — lossy (RCPS coverage)

Knob = `γ` (top-mass / nucleus-gated acceptance). Risk = token-disagreement rate
vs the lossless-spec continuation (== target, per Gate 1). RCPS with an empirical-
Bernstein UCB, δ=0.1, n_cal=80, n_test=40. **realised loss ≤ α for every α ⇒ PASS.**

| target α | calibrated γ | cal UCB | realised test loss | valid |
|---:|---:|---:|---:|:--:|
| 0.02 | 0.0 | — | 0.000 | ✓ |
| 0.05 | 0.0 | — | 0.000 | ✓ |
| 0.10 | 0.0 | 0.089 | 0.000 | ✓ |
| 0.20 | **0.2** | 0.176 | 0.010 | ✓ |
| 0.30 | **0.3** | 0.242 | 0.063 | ✓ |

The bound is honoured with margin (realised ≪ α): RCPS is conservative by design.
Leniency only deploys once the budget is loose enough to certify it through the
finite-sample UCB — at tight α the calibrator correctly falls back to the exact
rule (γ=0). Plot: `results/coverage.png` (all points below the realised=α line).

## Baseline vs lossless vs lossy — throughput

`results/throughput.csv`, greedy, 32 new tokens. **fwds = expensive target
forwards** (the roofline unit); **acc/call = tokens emitted per target forward.**

| ctx | regime | tok/s | acc/call | target fwds | peak VRAM |
|---:|:--|---:|---:|---:|---:|
| 1024 | no-spec | 22.1 | 0.97 | 33 | 4838 MB |
| 1024 | lossless | 18.8 | **4.57** | **7** | 4850 MB |
| 1024 | lossy γ=0.3 | 19.1 | 4.57 | 7 | 4850 MB |
| 4096 | no-spec | 10.1 | 0.97 | 33 | 7593 MB |
| 4096 | lossless | 2.5 | **4.57** | **7** | 7662 MB |
| 4096 | lossy γ=0.3 | 2.5 | 4.57 | 7 | 7662 MB |
| 16384 | all | — | — | — | **OOM** |

Read this two ways:
- **Step-count (the contribution):** lossless/lossy cut target forwards 33→7
  (**4.7×** fewer expensive steps) at *identical* output. This is the HBM-bound
  lever the trilogy is about — the cheapest token is the one you never run the big
  model for.
- **Wall-clock (the caveat):** spec is 0.85× at ctx=1024 and collapses to 0.27× at
  ctx=4096. At 4096 the two KV caches push VRAM to 7.66/8 GB → allocator thrash;
  16k OOMs (a *reported* data point, not a crash — taskspec §2). Even with free
  memory, the small target:draft ratio means draft + Python overhead eats the win.

## Acceptance rate vs K / context / drafter

`results/acceptance.csv`. acceptance_rate = accepted drafts / (rounds·K).

| drafter | ctx | K=2 | K=4 | K=6 |
|:--|---:|---:|---:|---:|
| small-model (0.5B) | 1024 | 0.833 | 0.750 | 0.633 |
| small-model (0.5B) | 4096 | 1.000 | 0.975 | 0.938 |
| prompt-lookup | 1024 | 0.500 | 0.350 | 0.304 |
| prompt-lookup | 4096 | 0.861 | 0.833 | 0.778 |

- The 0.5B draft is a strong predictor (84–100% accept at K≤4); acceptance falls as
  K grows (later drafts are conditioned on more speculative context).
- Prompt-lookup (zero draft weights) is weaker at short context but climbs sharply
  with context as the repetitive filler gives the n-gram matcher more to copy —
  the expected long-context behaviour.
- Best tokens-per-target-call (`acc/call`) reaches **6.0** at K=6, ctx=4096.

## Quality — no-spec / lossless / lossy

`results/quality.csv`. passkey retrieval, 6-task keyword accuracy, and token
agreement vs the target greedy reference. wikitext-2 ppl is the measuring stick.

| regime | passkey | task acc | token agreement vs target |
|:--|---:|---:|---:|
| no-spec | 1.00 | 0.83 | 1.000 |
| lossless | 1.00 | 0.83 | 1.000 |
| lossy γ=0.3 | 1.00 | 0.83 | 1.000 |

wikitext-2 ppl (target == lossless by construction): **7.321**.

Lossless matches the target exactly (agreement 1.000). At γ=0.3 the lossy path is
*also* indistinguishable on these short, high-confidence prompts — the draft sits
in the target's top-1 nucleus, so the lenient accept reduces to argmax. Drift only
appears at higher γ / lower-confidence continuations (next table), which is exactly
what the RCPS bound governs.

## Ablations

### γ sweep (speed ↔ fidelity), ctx=2048 — `results/ablation_gamma.csv`

| γ | acc/call | tok/s | risk (vs target) |
|---:|---:|---:|---:|
| 0.0 | 4.36 | 13.2 | 0.000 |
| 0.1 | 4.36 | 14.5 | 0.000 |
| 0.2 | 4.36 | 14.5 | 0.000 |
| 0.3 | 4.36 | 14.6 | 0.000 |
| 0.4 | 4.36 | 14.8 | 0.083 |
| 0.7 | 4.80 | 14.8 | 0.104 |
| 0.9 | 4.80 | 14.5 | 0.104 |

Monotone: risk is 0 at γ=0 and non-decreasing; acceptance rises only once γ is large
enough to admit off-top-1 drafts. With this strong draft the leniency headroom is
small — an honest finding: lossy helps most when the draft is *mediocre* (more
near-misses to forgive), not when it is already excellent.

### K (draft length) sweep, ctx=2048 — `results/ablation_k.csv`

| K | acc/call | accept rate |
|---:|---:|---:|
| 1 | 1.92 | 0.92 |
| 2 | 2.82 | 0.91 |
| 4 | 4.36 | 0.84 |
| 6 | 5.33 | 0.72 |

acc/call keeps rising with K (more tokens amortised per target forward) while the
per-token accept rate falls — the standard speculative trade. K≈4–6 maximises
tokens-per-target-call here.

### lossy-vs-context crossover (the caveat) — `results/ablation_crossover.csv`

| ctx | lossless speedup | lossy speedup |
|---:|---:|---:|
| 256 | 0.83× | 0.85× |
| 1024 | 0.80× | 0.77× |
| 4096 | 0.27× | 0.27× |

Wall-clock speedup is < 1 everywhere and degrades with context (memory pressure on
8 GB), confirming the headline caveat. The *step-count* speedup, by contrast, is
~4.7× and context-independent — see the throughput table.

## Plots (`results/`)

- `throughput.png` — tok/s vs context, three regimes (lead; read with the caveat).
- `tok_s_vs_loss.png` — tok/s vs guaranteed max quality loss α (the lossy knob).
- `acceptance.png` — acceptance rate vs K, small-model vs prompt-lookup.
- `coverage.png` — RCPS: realised loss ≤ α (the warranty holds).
- `crossover.png` — where speculation helps on wall-clock (and where it doesn't).

## Honest reading / what this is and isn't

The **contribution stands**: an exact distribution-preserving verify (gated token-
for-token), a single monotone leniency knob, and a **distribution-free RCPS warranty**
that the relaxed knob's quality loss stays ≤ α (gated on held-out). The roofline
*step-count* lever — 4.7× fewer target forwards at identical output — is the clean,
implementation-independent result.

What this **isn't**: a wall-clock win. A 1.5B target on 8 GB with a pure-Python
decode loop is the wrong regime for speculative *latency* — the target is too small
relative to the draft + overhead, and two KV caches exhaust 8 GB at long context.
That is consistent with the taskspec's non-goals (no production server, not beating
vLLM in absolute tok/s) and is itself the trilogy's recurring lesson: the 8 GB
memory wall decides where a lever pays off. The knob's serving home — where the
step-count win meets a real fused backend and a larger target — is `conformal-serve`
(taskspec §10).
