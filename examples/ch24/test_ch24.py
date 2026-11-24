# 제24장 검증 테스트
#
# 실행: python -m pytest examples/ch24 -q

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch19"):
    sys.path.insert(0, str(p))

from optim_memory import full_ft_budget, state_bytes, warmup_cosine  # noqa: E402


def test_sgd_step_by_hand():
    w = torch.tensor([1.0, -2.0], requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)
    (w ** 2).sum().backward()                     # 기울기 2w = [2, −4]
    opt.step()
    torch.testing.assert_close(w.detach(), torch.tensor([0.8, -1.6]))


def test_adamw_state_is_8_bytes_per_param():
    lin = torch.nn.Linear(10, 7)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    lin(torch.randn(3, 10)).sum().backward()
    opt.step()
    n = sum(p.numel() for p in lin.parameters())
    assert state_bytes(opt) == 8 * n


def test_weight_decay_shrinks_weights_without_gradient():
    w = torch.nn.Parameter(torch.ones(4))
    opt = torch.optim.AdamW([w], lr=0.1, weight_decay=0.5)
    w.grad = torch.zeros(4)                        # 손실 기울기가 0이어도
    opt.step()
    torch.testing.assert_close(w.detach(), torch.full((4,), 1 - 0.1 * 0.5))


def test_full_ft_budget():
    b = full_ft_budget(494_032_768, "AdamW")
    assert sum(b.values()) == 16 * 494_032_768
    assert sum(b.values()) > 2 * 2 ** 30           # GT 1030 2GiB를 넘는다


def test_warmup_cosine_shape():
    assert warmup_cosine(0, 10, 100) == pytest.approx(0.1)
    assert warmup_cosine(9, 10, 100) == pytest.approx(1.0)
    assert warmup_cosine(55, 10, 100) == pytest.approx(0.5)
    assert warmup_cosine(100, 10, 100) == pytest.approx(0.0, abs=1e-12)


def test_accumulation_equals_big_batch():
    torch.manual_seed(0)
    lin, x, y = torch.nn.Linear(5, 3), torch.randn(8, 5), torch.randint(0, 3, (8,))
    F.cross_entropy(lin(x), y).backward()
    big = lin.weight.grad.clone()
    lin.weight.grad = None
    for chunk in range(4):                         # 묶음 2개씩 4번, 손실을 4로 나눠 더함
        s = slice(2 * chunk, 2 * chunk + 2)
        (F.cross_entropy(lin(x[s]), y[s]) / 4).backward()
    torch.testing.assert_close(lin.weight.grad, big)


def test_clipping_limits_norm():
    w = torch.nn.Parameter(torch.zeros(3))
    w.grad = torch.tensor([3.0, 4.0, 0.0])         # 노름 5
    before = torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)
    assert before.item() == pytest.approx(5.0)
    assert w.grad.norm().item() == pytest.approx(1.0, abs=1e-6)


def test_adam_beats_small_sgd_on_bigram():
    try:
        from common.fixed_data import load_blocks
        from common.qwen import load_tokenizer
        tok = load_tokenizer(True)
    except Exception:
        pytest.skip("Qwen 토크나이저가 아직 내려받아지지 않음")
    import optim_compare as oc
    blocks, evals, v = oc.compact(load_blocks(tok, "train"), load_blocks(tok, "eval"))
    sgd, _, _ = oc.train(blocks, evals, v, lambda p: torch.optim.SGD(p, lr=0.1), "cpu")
    adam, _, _ = oc.train(blocks, evals, v, lambda p: torch.optim.Adam(p, lr=0.1), "cpu")
    assert adam[300] < sgd[300] - 3
