# 제26장 검증 테스트
#
# 실행: python -m pytest examples/ch26 -q

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch19", HERE.parent / "ch24", HERE.parent / "ch25"):
    sys.path.insert(0, str(p))

from freeze_demo import build, one_step  # noqa: E402
from partial_ft import SCOPES, freeze_except, snapshot  # noqa: E402


def test_frozen_has_no_grad_and_no_state():
    model = build()
    for p in model[0].parameters():
        p.requires_grad_(False)
    opt = one_step(model, model.parameters())
    assert model[0].weight.grad is None
    assert all(p not in opt.state for p in model[0].parameters())


@pytest.fixture()
def tiny():
    from hf_compat import to_hf
    from tiny_model import Qwen2Tiny, init_weights
    return to_hf(init_weights(Qwen2Tiny(), seed=0))


def test_trainable_counts(tiny):
    full, total = freeze_except(tiny, SCOPES["전체(Full FT)"])
    assert full == total == 41_914_112
    last, _ = freeze_except(tiny, SCOPES["마지막 Block"])
    assert last == 754_560                       # 제19장 손계산: 계층 하나
    norms, _ = freeze_except(tiny, SCOPES["RMSNorm만"])
    assert norms == (2 * 4 + 1) * 256            # 계층마다 2개 + Final Norm


def test_last_block_training_keeps_others(tiny):
    from trainer import train
    freeze_except(tiny, SCOPES["마지막 Block"])
    before = snapshot(tiny)
    ids = torch.randint(0, 1000, (4, 16))
    train(tiny, ids, {}, steps=3, warmup=1, log_every=3, lr=1e-3)
    after = snapshot(tiny)
    for name in before:
        same = torch.equal(before[name], after[name])
        assert same == (not name.startswith("model.layers.3.")), name
