"""End-to-end engine + rollback (loads the real models; needs CUDA + weights).

Skipped automatically where CUDA is unavailable. This is the §7 lossless gate in
test form: lossless-spec greedy must equal the target's own greedy decode token-
for-token, which only holds if KV rollback after each rejection is exact.
"""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def models():
    from spec_roofline.model import load_models
    return load_models()


@cuda
def test_lossless_greedy_matches_target(models):
    from spec_roofline.config import Config
    from spec_roofline.engine import SpeculativeDecoder
    from spec_roofline.data import gate_prompts
    dec = SpeculativeDecoder(models, Config())
    for ids in gate_prompts(models.tokenizer, n=3):
        ids = ids.to(models.device)
        base = dec.generate_baseline(ids, 40, mode="greedy")
        spec = dec.generate_spec(ids, 40, mode="greedy", gamma=0.0)
        assert base.tokens == spec.tokens          # token-for-token (rollback exact)
        assert spec.n_target_forwards < base.n_target_forwards   # and it sped up


@cuda
def test_prompt_lookup_matches_target_greedy(models):
    from spec_roofline.config import Config
    from spec_roofline.engine import SpeculativeDecoder
    from spec_roofline.data import gate_prompts
    dec_pl = SpeculativeDecoder(models, Config().with_method("prompt_lookup"))
    dec = SpeculativeDecoder(models, Config())
    ids = gate_prompts(models.tokenizer, n=1)[0].to(models.device)
    base = dec.generate_baseline(ids, 40, mode="greedy")
    pl = dec_pl.generate_spec(ids, 40, mode="greedy", gamma=0.0)
    assert base.tokens == pl.tokens                # lossless regardless of drafter


@cuda
def test_lossy_monotone_acceptance(models):
    from spec_roofline.config import Config
    from spec_roofline.engine import SpeculativeDecoder
    from spec_roofline.data import gate_prompts
    dec = SpeculativeDecoder(models, Config())
    ids = gate_prompts(models.tokenizer, n=1)[0].to(models.device)
    prev = 0.0
    for g in (0.0, 0.3, 0.6, 0.9):
        r = dec.generate_spec(ids, 48, mode="greedy", gamma=g)
        assert r.accepted_per_call >= prev - 1e-6
        prev = r.accepted_per_call
