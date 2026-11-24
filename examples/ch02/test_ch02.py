# 제02장 검증 테스트
#
# 실행: python -m pytest examples/ch02 -q

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.device import select_device  # noqa: E402
import bench_matmul  # noqa: E402
import memory_budget  # noqa: E402


# ── 1. 계산식 ───────────────────────────────────────────────────────────

def test_matmul_flops():
    # 2×2 행렬곱: 결과 4칸 × (곱셈 2 + 덧셈 2) = 16 = 2 × 2³
    assert bench_matmul.matmul_flops(2) == 16
    assert bench_matmul.matmul_flops(1024) == 2 * 1024 ** 3


def test_qwen_param_count_matches_official_size():
    # Qwen2.5-0.5B: 공식 표기 0.49B, Embedding은 공유 텐서로 한 번만 계산
    params = memory_budget.count_qwen_params(memory_budget.QWEN05)
    assert params == 494_032_768


def test_qwen_param_count_matches_torch_modules():
    # 같은 구조를 실제 PyTorch 계층으로 만들어 파라미터 수를 세어 대조한다(1개 계층).
    c = memory_budget.QWEN05
    h, i, kv = c["hidden"], c["intermediate"], c["kv_heads"] * (c["hidden"] // c["heads"])
    layer = torch.nn.ModuleList([
        torch.nn.Linear(h, h), torch.nn.Linear(h, kv), torch.nn.Linear(h, kv),   # q, k, v (+bias)
        torch.nn.Linear(h, h, bias=False),                                         # o
        torch.nn.Linear(h, i, bias=False), torch.nn.Linear(h, i, bias=False),      # gate, up
        torch.nn.Linear(i, h, bias=False),                                         # down
    ])
    per_layer = sum(p.numel() for p in layer.parameters()) + 2 * h               # + RMSNorm 2개
    expected = c["vocab"] * h + c["layers"] * per_layer + h
    assert memory_budget.count_qwen_params(c) == expected


def test_logits_bytes():
    # 시퀀스 256 × 어휘 151,936 × 4바이트
    assert memory_budget.logits_bytes(256, 151936) == 155_582_464


def test_fp32_weights_exceed_2gib_budget_with_context():
    # FP32 가중치만 약 1.84GiB → CUDA 컨텍스트를 더하면 GT 1030 2GB에 들어가지 않는다.
    params = memory_budget.count_qwen_params(memory_budget.QWEN05)
    assert 1.8 < params * 4 / 2 ** 30 < 1.9


# ── 2. 실제 장치 측정 ───────────────────────────────────────────────────

def test_time_matmul_returns_positive():
    device, dtype = select_device(verbose=False)
    assert bench_matmul.time_matmul(128, device, dtype, repeat=2) > 0


@pytest.mark.parametrize("device, dtype", bench_matmul.run_configs())
def test_relative_error_within_tolerance(device, dtype):
    # FP32는 1e-5, 16비트는 유효 숫자가 적어 1e-2를 기준으로 한다.
    limit = 1e-5 if dtype == torch.float32 else 1e-2
    assert bench_matmul.relative_error(256, device, dtype) < limit


def test_allocation_matches_calculation():
    device, _ = select_device(verbose=False)
    expected, actual = memory_budget.measure_allocation(device, mib=16)
    assert expected == actual
