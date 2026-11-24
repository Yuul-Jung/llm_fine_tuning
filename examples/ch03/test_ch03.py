# 제03장 검증 테스트
#
# 실행: python -m pytest examples/ch03 -q

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from head_split import merge_heads, split_heads  # noqa: E402


def sample(batch=2, seq=4, hidden=8):
    return torch.arange(batch * seq * hidden, dtype=torch.float32).reshape(batch, seq, hidden)


# ── 1. 모양 예측 ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "shape, heads, expected",
    [
        ((2, 4, 8), 2, (2, 2, 4, 4)),       # 교재 예제
        ((1, 16, 896), 14, (1, 14, 16, 64)),  # Qwen2.5-0.5B Query
        ((3, 5, 256), 4, (3, 4, 5, 64)),     # Qwen2-Tiny Query
    ],
)
def test_split_shape(shape, heads, expected):
    assert tuple(split_heads(torch.zeros(shape), heads).shape) == expected


def test_axis_sum_shapes():
    x = sample()
    assert x.sum(dim=0).shape == (4, 8)
    assert x.sum(dim=1).shape == (2, 8)
    assert x.sum(dim=-1).shape == (2, 4)


def test_unsqueeze_mask_shape():
    # [batch, seq] 마스크를 Attention 점수 [batch, heads, seq, seq]에 맞추는 모양
    mask = torch.ones(2, 4)
    assert mask.unsqueeze(1).unsqueeze(2).shape == (2, 1, 1, 4)


# ── 2. 왕복 변환과 원소 위치 ─────────────────────────────────────────────

def test_round_trip_exact():
    x = sample()
    assert torch.equal(merge_heads(split_heads(x, 2)), x)


def test_round_trip_random_qwen_size():
    torch.manual_seed(0)
    x = torch.randn(2, 7, 896)
    assert torch.equal(merge_heads(split_heads(x, 14)), x)


def test_element_mapping():
    x = sample()
    y = split_heads(x, 2)
    head_dim = 4
    for b in range(2):
        for s in range(4):
            for h in range(2):
                for d in range(head_dim):
                    assert y[b, h, s, d] == x[b, s, h * head_dim + d]


# ── 3. 메모리 공유와 연속 저장 ───────────────────────────────────────────

def test_slice_shares_memory():
    x = sample()
    view = x[:, -1, :]
    view[0, 0] = -1.0
    assert x[0, -1, 0] == -1.0


def test_view_fails_after_transpose():
    t = sample().transpose(1, 2)
    assert not t.is_contiguous()
    with pytest.raises(RuntimeError):
        t.view(2, -1)
    assert t.reshape(2, -1).shape == (2, 32)
