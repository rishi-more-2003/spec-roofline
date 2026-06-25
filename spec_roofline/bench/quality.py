"""spec_roofline/bench/quality.py — passkey + wikitext ppl + task accuracy (§6).

The quality axis the gates defend: fp16 no-spec vs lossless (must match) vs lossy
(degrades, bounded). Three probes:

  * passkey retrieval — bury a number in long filler, ask for it back. Tests that
    speculation does not break long-context behaviour.
  * task-suite accuracy — short prompts with checkable answers (keyword match).
  * token agreement vs the target's greedy continuation — 0 for lossless (it *is*
    the target), bounded > 0 for lossy. This is the risk the warranty covers.

wikitext ppl is reported once for the target as the measuring-stick number;
lossless decoding does not change the model, so its ppl is the target's by
construction.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict

import torch

from ..config import Config
from ..engine import SpeculativeDecoder
from ..data import passkey_prompt, TASK_PROMPTS, chat_ids, wikitext_slice
from .common import RESULTS

# checkable task suite: (prompt, expected lowercase substring in the answer).
TASK_SUITE = [
    ("What is the capital of France? Answer with one word.", "paris"),
    ("What is 7 multiplied by 8? Answer with the number only.", "56"),
    ("What color do you get by mixing blue and yellow? One word.", "green"),
    ("What planet is known as the Red Planet? One word.", "mars"),
    ("How many continents are there on Earth? Answer with the number.", "7"),
    ("What gas do plants absorb from the air for photosynthesis?", "carbon"),
]


@dataclass
class QualityRow:
    regime: str
    gamma: float
    passkey_acc: float
    task_acc: float
    token_agreement: float       # vs target greedy (1.0 for lossless)


def _regimes(lossy_gamma):
    return [("no_spec", 0.0), ("lossless", 0.0), ("lossy", lossy_gamma)]


@torch.no_grad()
def _gen(dec, ids, n_new, regime, gamma):
    if regime == "no_spec":
        return dec.generate_baseline(ids, n_new, mode="greedy").tokens
    return dec.generate_spec(ids, n_new, mode="greedy", gamma=gamma).tokens


@torch.no_grad()
def run(models, cfg: Config | None = None, lossy_gamma: float = 0.3,
        passkey_contexts=(512, 2048)) -> dict:
    cfg = cfg or Config()
    dec = SpeculativeDecoder(models, cfg)
    tok = models.tokenizer
    rows = []
    # references (lossless-spec greedy == target greedy per §7 gate) for token-
    # agreement; gamma=0 spec path so lossless reads exactly 1.000.
    task_ids = [chat_ids(tok, p).to(models.device) for p, _ in TASK_SUITE]
    task_refs = [dec.generate_spec(x, 24, mode="greedy", gamma=0.0).tokens for x in task_ids]

    for regime, gamma in _regimes(lossy_gamma):
        # passkey
        hits = tot = 0
        for ctx in passkey_contexts:
            pk = 73219
            ids = passkey_prompt(tok, ctx, pk, depth=0.5).to(models.device)
            out = _gen(dec, ids, 12, regime, gamma)
            txt = tok.decode(out, skip_special_tokens=True)
            hits += int(str(pk) in txt)
            tot += 1
            del ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        passkey_acc = hits / max(1, tot)
        # task suite + token agreement vs target greedy
        t_hits = 0
        agree_num = agree_den = 0
        for (prompt, expect), ids, ref in zip(TASK_SUITE, task_ids, task_refs):
            out = _gen(dec, ids, 24, regime, gamma)
            txt = tok.decode(out, skip_special_tokens=True).lower()
            t_hits += int(expect in txt)
            L = min(len(out), len(ref))
            agree_num += sum(1 for i in range(L) if out[i] == ref[i])
            agree_den += max(len(out), len(ref))
        rows.append(QualityRow(
            regime=regime, gamma=gamma,
            passkey_acc=round(passkey_acc, 3),
            task_acc=round(t_hits / len(TASK_SUITE), 3),
            token_agreement=round(agree_num / max(1, agree_den), 3)))
    _write(rows)
    return {"rows": rows}


@torch.no_grad()
def wikitext_ppl(models, n_tokens: int = 2048) -> float:
    """Target-model perplexity on a wikitext-2 slice (the measuring stick).

    Lossless decoding preserves the target distribution, so this is the lossless
    ppl by construction; reported once."""
    ids = wikitext_slice(models.tokenizer, n_tokens).to(models.device)
    out = models.target(input_ids=ids, use_cache=False)
    logits = out.logits[0, :-1].float()
    tgt = ids[0, 1:]
    nll = torch.nn.functional.cross_entropy(logits, tgt, reduction="mean")
    return float(math.exp(nll))


def _write(rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "quality.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return path
