# 제23장 검증 테스트
#
# 실행: python -m pytest examples/ch23 -q

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch19"):
    sys.path.insert(0, str(p))

from grad_basics import B0, W0, grad_autograd, grad_by_hand, grad_numeric, loss_fn  # noqa: E402


def test_three_methods_agree():
    hw, hb = grad_by_hand(W0, B0)
    nw, nb = grad_numeric(W0.clone(), B0.clone())
    aw, ab = grad_autograd(W0, B0)
    torch.testing.assert_close(aw, hw, atol=1e-12, rtol=0)
    torch.testing.assert_close(ab, hb, atol=1e-12, rtol=0)
    torch.testing.assert_close(nw, hw, atol=1e-8, rtol=0)
    torch.testing.assert_close(nb, hb, atol=1e-8, rtol=0)


def test_gradcheck_passes():
    w, b = W0.clone().requires_grad_(True), B0.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(loss_fn, (w, b))


def test_bias_grad_sums_to_zero():
    # softmax − 원-핫의 합은 1 − 1 = 0
    _, hb = grad_by_hand(W0, B0)
    assert hb.sum().item() == pytest.approx(0.0, abs=1e-12)


def test_grad_accumulates_until_reset():
    w = W0.clone().requires_grad_(True)
    loss_fn(w, B0).backward()
    once = w.grad.clone()
    loss_fn(w, B0).backward()
    torch.testing.assert_close(w.grad, 2 * once)
    w.grad = None
    loss_fn(w, B0).backward()
    assert torch.equal(w.grad, once)


def test_tiny_backward_reaches_every_parameter():
    from hf_compat import to_hf
    from tiny_model import Qwen2Tiny, init_weights
    model = to_hf(init_weights(Qwen2Tiny(), seed=0))
    ids = torch.tensor([[9707, 11, 1879, 0, 1234, 42, 7, 7]])
    model(input_ids=ids, labels=ids).loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)
    # 공유 텐서(LM Head)를 통해 어휘 전체 행이 Gradient를 받는다
    emb = model.model.embed_tokens.weight.grad
    assert (emb.abs().sum(dim=1) > 0).all()
