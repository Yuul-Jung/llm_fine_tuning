# 제29장 검증 테스트
#
# 실행: python -m pytest examples/ch29 -q

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch19", HERE.parent / "ch24", HERE.parent / "ch25",
          HERE.parent / "ch28"):
    sys.path.insert(0, str(p))

from lora_linear import LoRALinear, apply_lora, merge_lora, trainable_count  # noqa: E402


def test_matches_formula_of_ch28():
    from lora_math import lora_forward
    torch.manual_seed(0)
    base = nn.Linear(5, 4, bias=False)
    lora = LoRALinear(base, r=2, alpha=8)
    with torch.no_grad():
        lora.lora_B.normal_()
    x = torch.randn(3, 5)
    expect = lora_forward(x, base.weight, lora.lora_A, lora.lora_B, alpha=8, r=2)
    torch.testing.assert_close(lora(x), expect)


def test_starts_as_identity_and_only_ab_get_grad():
    base = nn.Linear(6, 3)
    lora = LoRALinear(base, r=2)
    x = torch.randn(4, 6)
    torch.testing.assert_close(lora(x), base(x))
    lora(x).sum().backward()
    assert base.weight.grad is None and base.bias.grad is None
    assert lora.lora_B.grad is not None and lora.lora_A.grad is not None


def test_merge_keeps_output():
    torch.manual_seed(1)
    lora = LoRALinear(nn.Linear(6, 3), r=2, alpha=4)
    with torch.no_grad():
        lora.lora_B.normal_()
    x = torch.randn(4, 6)
    torch.testing.assert_close(lora.merged()(x), lora(x), atol=1e-6, rtol=1e-5)


@pytest.fixture()
def tiny():
    from hf_compat import to_hf
    from tiny_model import Qwen2Tiny, init_weights
    return to_hf(init_weights(Qwen2Tiny(), seed=0))


def test_apply_on_tiny_counts_and_reproducible_init(tiny):
    names = apply_lora(tiny, seed=0)
    assert len(names) == 8
    assert trainable_count(tiny) == 4 * (8 * (256 + 256) + 8 * (256 + 64))     # 26,624
    a1 = tiny.model.layers[0].self_attn.q_proj.lora_A.detach().clone()
    other = nn.Linear(256, 256)
    torch.manual_seed(0)
    assert torch.equal(LoRALinear(other).lora_A, a1)             # 같은 seed면 같은 A


def test_train_merge_tiny(tiny):
    from trainer import train
    ids = torch.randint(0, 2000, (4, 24))
    w_before = tiny.model.layers[1].self_attn.q_proj.weight.detach().clone()
    apply_lora(tiny)
    train(tiny, ids, {}, steps=5, lr=1e-2, warmup=1, log_every=5)
    assert torch.equal(tiny.model.layers[1].self_attn.q_proj.base.weight, w_before)   # 원본 불변
    with torch.no_grad():
        lora_logits = tiny(ids[:1]).logits
    assert merge_lora(tiny) == 8
    with torch.no_grad():
        torch.testing.assert_close(tiny(ids[:1]).logits, lora_logits, atol=1e-4, rtol=1e-4)
    assert not torch.equal(tiny.model.layers[1].self_attn.q_proj.weight, w_before)    # ΔW가 더해짐
