"""spec_roofline/engine.py — the SpeculativeDecoder loop + KV rollback.

The systems core (taskspec §5 / §9). One round:

  1. DRAFT   the small model (or prompt-lookup) proposes K tokens.
  2. VERIFY  the target runs *one* batched forward over [last_committed, d_1..d_K],
     producing K+1 distributions (the only expensive step in the round).
  3. ACCEPT  verify.py / lossy.py accept the longest valid prefix and emit one
     correction/bonus token.
  4. ROLLBACK both KV caches are cropped back to the committed length, discarding
     the rejected drafts' KV (model.sync_cache).

Cache convention: each cache holds ``committed[:-1]`` at the top of a round; the
last committed token is re-fed as the first verify/draft input, so the models
produce a distribution for the *first* drafted slot. After a round the caches are
over-extended by the speculated tokens; the next round's ``sync_cache`` crops them
(and, on a full-accept that left the draft cache one short, feeds the gap).

The roofline accounting (taskspec's through-line): the expensive unit is a
*target forward*. ``n_target_forwards == n_rounds``; mean accepted-tokens-per-
target-call = generated / rounds is the speedup lever. The baseline runs the
target one token per forward (the measuring stick).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache

import torch.nn.functional as F

from .config import Config
from .model import TwoModels, forward_logits, sync_cache
from .drafter import Drafter
from .lossy import verify_lossy_greedy, verify_lossy_sample, emission_tv


@dataclass
class GenResult:
    tokens: list                  # generated token ids (<= max_new_tokens)
    n_target_forwards: int        # expensive steps (== rounds for spec; == tokens for baseline)
    n_draft_forwards: int
    n_rounds: int
    n_accept_total: int           # drafts accepted (excludes corrections/bonus)
    accept_counts: list = field(default_factory=list)   # accepted drafts per round
    risk_tv: float = 0.0          # mean per-step emission TV vs target (sample mode)

    @property
    def n_generated(self) -> int:
        return len(self.tokens)

    @property
    def accepted_per_call(self) -> float:
        """Mean emitted tokens per expensive target forward (taskspec §6)."""
        return self.n_generated / max(1, self.n_target_forwards)


class SpeculativeDecoder:
    def __init__(self, models: TwoModels, cfg: Config | None = None):
        self.m = models
        self.cfg = cfg or Config()
        self.drafter = Drafter(self.cfg.draft)
        self.device = models.device
        self.eos = models.tokenizer.eos_token_id

    # -- baseline: plain target autoregressive decode (the measuring stick) ----
    @torch.no_grad()
    def generate_baseline(self, input_ids: torch.Tensor, max_new_tokens: int,
                          mode: str = "greedy", gen: torch.Generator | None = None) -> GenResult:
        committed = input_ids.view(-1).to(self.device)
        cache = DynamicCache()
        # prefill the prompt.
        logits = forward_logits(self.m.target, committed.view(1, -1), cache, 0)
        seen = committed.shape[0]
        generated, nfwd = [], 1
        last = logits[-1]
        for _ in range(max_new_tokens):
            tok = self._pick(last, mode, gen)
            generated.append(tok)
            if self._is_eos(tok):
                break
            cur = torch.tensor([[tok]], device=self.device)
            logits = forward_logits(self.m.target, cur, cache, seen)
            seen += 1
            nfwd += 1
            last = logits[-1]
        return GenResult(generated, nfwd, 0, len(generated), 0, [])

    # -- speculative: lossless (gamma=0) or lossy (gamma>0) --------------------
    @torch.no_grad()
    def generate_spec(self, input_ids: torch.Tensor, max_new_tokens: int,
                      mode: str = "greedy", gamma: float = 0.0,
                      gen: torch.Generator | None = None) -> GenResult:
        device = self.device
        committed = input_ids.view(-1).to(device)
        tgt_cache, drf_cache = DynamicCache(), DynamicCache()
        tgt_seen = drf_seen = 0
        use_pl = self.cfg.draft.method == "prompt_lookup"
        temp = self.cfg.draft.temperature

        generated, accept_counts = [], []
        n_rounds = n_accept = n_draft_fwd = 0
        tv_sum = tv_count = 0     # distributional risk accumulator (sample mode)

        while len(generated) < max_new_tokens:
            C = committed.shape[0]
            tgt_seen = sync_cache(self.m.target, tgt_cache, tgt_seen, committed[:-1])
            if not use_pl:
                drf_seen = sync_cache(self.m.draft, drf_cache, drf_seen, committed[:-1])

            # 1. DRAFT
            draft_tokens, draft_q, drf_seen = self.drafter.propose(
                self.m.draft, drf_cache, committed, drf_seen, mode, gen)
            K = int(draft_tokens.shape[0])
            if not use_pl:
                n_draft_fwd += K

            # 2. VERIFY forward (the one expensive step): [committed[-1], d_1..d_K]
            verify_input = torch.cat([committed[-1:], draft_tokens]).view(1, -1)
            tgt_logits = forward_logits(self.m.target, verify_input, tgt_cache, tgt_seen)
            tgt_seen += verify_input.shape[1]
            n_rounds += 1

            # 3. ACCEPT
            if mode == "greedy":
                res = verify_lossy_greedy(draft_tokens, tgt_logits, gamma)
            else:
                res = verify_lossy_sample(draft_tokens, draft_q, tgt_logits, gamma, gen, temp)
                # distributional risk: per-drafted-position TV(p_gamma || p) at temp.
                if K > 0:
                    p_all = F.softmax(tgt_logits[:K] / temp, dim=-1)
                    for i in range(K):
                        tv_sum += emission_tv(p_all[i], draft_q[i], gamma)
                    tv_count += K

            a = res.n_accept
            new_tokens = draft_tokens[:a].tolist() + [res.correction]
            committed = torch.cat(
                [committed, torch.tensor(new_tokens, device=device, dtype=torch.long)])
            n_accept += a
            accept_counts.append(a)

            # EOS: append up to and including the first eos, then stop.
            stop = False
            for j, t in enumerate(new_tokens):
                generated.append(t)
                if self._is_eos(t) or len(generated) >= max_new_tokens:
                    new_tokens = new_tokens[:j + 1]
                    stop = True
                    break
            if stop:
                break
            # 4. ROLLBACK happens lazily via sync_cache at the top of next round.

        return GenResult(generated[:max_new_tokens], n_rounds, n_draft_fwd,
                         n_rounds, n_accept, accept_counts,
                         risk_tv=(tv_sum / tv_count if tv_count else 0.0))

    # -- helpers ---------------------------------------------------------------
    def _pick(self, logits_row: torch.Tensor, mode: str, gen) -> int:
        if mode == "greedy":
            return int(logits_row.argmax())
        probs = torch.softmax(logits_row / self.cfg.draft.temperature, dim=-1)
        return int(torch.multinomial(probs, 1, generator=gen).item())

    def _is_eos(self, tok: int) -> bool:
        return self.eos is not None and tok == self.eos
