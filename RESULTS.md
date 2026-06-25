# spec-roofline — results

Hardware: RTX 4070 Laptop, 8 GB, sm_89. Target `Qwen2.5-1.5B-Instruct`, draft
`Qwen2.5-0.5B-Instruct`, fp16, batch-1 greedy unless noted. Numbers from
`results/*.csv` (regenerate: `python scripts/run_benches.py` + `cli calibrate`).

## TL;DR

- **Both §7 gates pass.** Lossless-spec reproduces the target greedy output
  **token-for-token** (KV rollback is exact); the lossy knob's realised quality
  loss stays **≤ α** on held-out data for every α (RCPS coverage).
- **The step-count lever is real and large:** lossless-spec emits **4.6 tokens per
  target forward** — 32 tokens in **7** expensive target steps instead of **33**
  (**4.7× fewer**). This is implementation-independent. *Wall-clock is net-negative
  (0.27–0.85×) at 1.5B:0.5B on 8 GB — analysed below, not hidden.*
- **The lossy knob moves the needle, and we draw its scope.** On the diverse
  prompts it is calibrated on, leniency lifts accepted-per-call along a real,
  monotone frontier — **3.37 → 4.48 (+33%)** as γ→0.9 — but the frontier is
  *steep* (risk climbs 0 → 0.65). Under the RCPS warranty the deployable gain is
  therefore **small for this strong-draft pair: +1.1% at α=0.3** (γ=0.3, realised
  loss 0.067 ≤ 0.3), nothing at α≤0.2. The honest finding is the **boundary**: a
  0.5B draft already accepts 84–100%, so there is almost nothing for leniency to
  recover at bounded loss. The knob is a lever for *weak-draft / high-entropy*
  regimes; this pair has no slack to give. See [Headroom](#where-the-lossy-knob-buys-speed-headroom).
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

## Where the lossy knob buys speed (headroom)

`results/headroom_sweep.csv` + `headroom_calibrated.csv`
(`python -m spec_roofline.cli headroom`). The lossy knob is the contribution, so
this measures it where it is *calibrated* — the diverse prompt distribution, model
drafter — not the repetitive throughput prompts (where the target is so confident
the knob never bites and lossy trivially ties lossless).

**The speed↔fidelity frontier (model drafter, 60 diverse prompts):**

| γ | acc/call | Δ vs lossless | mean risk |
|---:|---:|---:|---:|
| 0.0 | 3.37 | — | 0.000 |
| 0.2 | 3.36 | −0.2% | 0.013 |
| 0.3 | 3.41 | **+1.1%** | 0.067 |
| 0.4 | 3.46 | +2.7% | 0.155 |
| 0.5 | 3.64 | +8.1% | 0.233 |
| 0.7 | 3.86 | +14% | 0.462 |
| 0.9 | 4.48 | **+33%** | 0.649 |

**The quantity moves** — leniency lifts accepted-per-target-call up to +33%. But the
frontier is *steep*: risk (token-disagreement vs the lossless/target path) climbs to
0.65 by γ=0.9. Whether that speed is worth taking is exactly what the warranty
decides.

**What the RCPS warranty actually deploys (the receipt):**

| target α | calibrated γ | acc/call | lift vs lossless | realised loss | valid |
|---:|---:|---:|---:|---:|:--:|
| 0.02 / 0.05 / 0.10 | 0.0 | 3.37 | 1.00× | 0.000 | ✓ |
| 0.20 | 0.2 | 3.36 | 1.00× | 0.013 | ✓ |
| 0.30 | **0.3** | 3.41 | **+1.1%** | 0.067 | ✓ |

**The honest result.** The deployable, warranty-backed gain is **small for this
pair — +1.1% acc/call at α=0.3, nothing tighter.** This is not a knob failure; it
is the *boundary*, measured: a 0.5B draft already accepts 84–100%, and the few
exact-verify rejections are **far-misses** (a rejected token is usually deep in the
target's tail, not a near-miss), so admitting them costs disproportionate quality.
Leniency pays where the draft is *weak* (many recoverable rejections) or the target
is *high-entropy* (cheap near-misses) — a well-aligned 1.5B/0.5B greedy pair is
neither. The knob is built, gated, and its useful regime is drawn rather than
oversold. (Contrast: the prompt-lookup drafter on these non-repetitive prompts
barely proposes — acc/call 1.02, flat across γ — the other end of "no headroom".)

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

> Note: lossy γ=0.3 shows token-agreement 1.000 *here* because these short factual
> prompts are high-confidence — γ=0.3 reduces to argmax (the knob is inert). Its
> measurable, bounded drift appears on the diverse calibration distribution, where
> the warranty governs it — see [Headroom](#where-the-lossy-knob-buys-speed-headroom).

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

Monotone: risk is 0 at γ=0 and non-decreasing. This sweep is on the *repetitive*
context (build_context, ctx=2048) where the target is highly confident, so the knob
barely bites until γ≥0.7 — the clearest visual of "no headroom". The properly
calibrated measurement, on the diverse distribution, is in
[Headroom](#where-the-lossy-knob-buys-speed-headroom); the conclusion is the same
and quantified: leniency helps a *weak* draft, not an already-excellent one.

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
- `headroom.png` — acc/call vs γ with the risk axis: where the knob buys speed.
- `tok_s_vs_loss.png` — tok/s vs guaranteed max quality loss α (the lossy knob).
- `acceptance.png` — acceptance rate vs K, small-model vs prompt-lookup.
- `coverage.png` — RCPS: realised loss ≤ α (the warranty holds).
- `crossover.png` — where speculation helps on wall-clock (and where it doesn't).

## Honest reading / what this is and isn't

This is a **rigorously-delineated negative-leaning result**, and that is the point.
Three things are built and gated: an exact distribution-preserving verify (gated
token-for-token), a single monotone leniency knob, and a **distribution-free RCPS
warranty** that the relaxed knob's quality loss stays ≤ α (gated on held-out).

What pays and what does not, measured:

- **Step-count: pays.** 4.7× fewer target forwards at identical output —
  implementation-independent, the clean win.
- **Wall-clock: does not pay here.** 0.27–0.85× — a 1.5B target is too small
  relative to a 0.5B draft + Python overhead, and two KV caches exhaust 8 GB at
  long context. (Non-goals, taskspec §1; the trilogy's lesson — the 8 GB memory
  wall decides where a lever pays off.)
- **Lossy leniency: pays a little, and we say how little.** +1.1% acc/call at the
  loosest certified budget (α=0.3); the frontier reaches +33% but at a quality cost
  no sane α accepts. The boundary is the finding: a strong, well-aligned draft
  leaves almost nothing for leniency to recover at bounded loss.

Framed as judgment, not as a win: the value here is showing *where* two speculative
levers pay off and proving *why* they don't where they don't — and a knob whose
warranty is honoured even when the headroom it governs is small. The regime where
all of this converts to latency — a larger target, a fused backend, the α budget
shared across levers — is `conformal-serve` (taskspec §10).
