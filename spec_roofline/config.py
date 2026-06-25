"""spec_roofline/config.py — pinned decisions (taskspec §3, §4) + tunables.

The trilogy's third repo. Same discipline as kv-roofline: the two DECISIONS the
spec forces are resolved here, in code, not prose.

  * §3 MODELS  = target  Qwen/Qwen2.5-1.5B-Instruct  (continuity with
                 decode-roofline / kv-roofline; ~3 GB fp16)
                 draft   Qwen/Qwen2.5-0.5B-Instruct  (~1 GB fp16)
                 Same family => identical tokenizer/vocab (151936), which the
                 verify step *requires* (target & draft probs over one vocab).
                 Both co-reside on 8 GB with room for KV. A zero-weight
                 prompt-lookup drafter rides behind a flag (DraftConfig.method).

  * §4 LOSSY   = a single scalar leniency knob `gamma` on the per-token accept
                 test. gamma=0 reduces to the exact rejection-sampling rule
                 (lossless). Increasing gamma accepts more drafts (more speed,
                 monotone more quality drift) — exactly one knob for RCPS to
                 calibrate against a quality risk. See lossy.py / conformal.py.

Everything downstream reads these. Nothing else hard-codes a model id, K, or the
relaxation rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

# ----------------------------------------------------------------------------
# §3 pinned models
# ----------------------------------------------------------------------------
TARGET_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"
VOCAB_SIZE: int = 151936  # shared tokenizer; asserted at load (verify needs it)

DraftMethod = Literal["model", "prompt_lookup"]
VerifyMode = Literal["greedy", "sample"]


@dataclass(frozen=True)
class DraftConfig:
    """The drafter (taskspec §9.1)."""

    method: DraftMethod = "model"   # "model" = 0.5B small model; "prompt_lookup" = n-gram
    k: int = 4                      # K: draft length (tokens proposed per round)
    # prompt-lookup: match the last `pl_ngram` tokens against history, copy what
    # followed. Greedy-verify only (no draft distribution to rejection-sample).
    pl_ngram: int = 3
    temperature: float = 1.0        # draft sampling temperature (sample mode)


@dataclass(frozen=True)
class LossyConfig:
    """§4 relaxed acceptance. One monotone leniency knob.

    Accept rule (greedy regime, the calibrated path): accept drafted token d at a
    target position iff ``p_target(d) >= (1 - gamma) * max_t p_target(t)``.
      * gamma = 0  -> accept only the target argmax == exact greedy verify (lossless).
      * gamma -> 1 -> accept anything the draft proposes (max speed, max drift).
    Monotone non-decreasing acceptance (and risk) in gamma => RCPS applies.

    Sample regime (for completeness): accept prob ``min(1, p(d)/((1-gamma)*q(d)))``;
    residual resample from ``norm((p - (1-gamma) q)_+)``. gamma=0 -> exact Leviathan.
    """

    gamma: float = 0.0
    # RCPS calibration grid (ascending; risk increases with gamma so we take the
    # largest gamma whose risk UCB <= alpha).
    gamma_grid: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)
    delta: float = 0.1                          # RCPS confidence (1 - delta)
    alphas: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.3)  # target quality-loss budgets


@dataclass(frozen=True)
class ModelConfig:
    target_id: str = TARGET_ID
    draft_id: str = DRAFT_ID
    dtype: str = "float16"
    device: str = "cuda"
    attn_impl: str = "sdpa"


@dataclass(frozen=True)
class BenchConfig:
    # taskspec §6: tok/s vs context length is the lead plot. 128k is gated by the
    # 8 GB budget (OOM is a reported data point, not a crash) — default sweep
    # stops where two models + KV fit; longer contexts are opt-in.
    context_lengths: tuple[int, ...] = (1024, 4096, 16384, 32768)
    decode_steps: int = 128       # new tokens generated per measured run
    warmup_steps: int = 16
    vram_budget_bytes: int = 8 * 1024**3   # 8 GB (RTX 4070 Laptop)
    n_prompts: int = 4


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    draft: DraftConfig = field(default_factory=DraftConfig)
    lossy: LossyConfig = field(default_factory=LossyConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)
    seed: int = 0

    # convenience constructors for sweeps -------------------------------------
    def with_k(self, k: int) -> "Config":
        return replace(self, draft=replace(self.draft, k=k))

    def with_method(self, method: DraftMethod) -> "Config":
        return replace(self, draft=replace(self.draft, method=method))

    def with_gamma(self, gamma: float) -> "Config":
        return replace(self, lossy=replace(self.lossy, gamma=gamma))


DEFAULT = Config()
