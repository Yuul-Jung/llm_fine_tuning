# 제25장 검증 테스트
#
# 실행: python -m pytest examples/ch25 -q
#
# 전체 학습(수 분)은 pretrain_tiny.py로 실행하고, 테스트는 짧은 학습과 저장된 기록을 확인한다.

import json
import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch19", HERE.parent / "ch24"):
    sys.path.insert(0, str(p))

from tiny_model import TINY, Qwen2Tiny, init_weights  # noqa: E402
from trainer import eval_loss, train  # noqa: E402


def small_tiny():
    """테스트용으로 계층 1개, 어휘 앞부분만 쓰는 축소판(같은 Qwen2 구조)"""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(**{**TINY, "num_hidden_layers": 1, "vocab_size": 512}, tie_word_embeddings=True)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg)


def test_short_training_reduces_loss():
    torch.manual_seed(0)
    blocks = torch.randint(0, 64, (6, 32))            # 어휘 64개만 쓰는 무작위 데이터
    blocks[:, 1::2] = blocks[:, ::2] + 1                # 규칙: 짝수 위치 다음은 +1
    model = small_tiny()
    hist = train(model, blocks, {"평가": blocks[:2]}, steps=40, batch=2, accum=2, lr=3e-3,
                 warmup=5, log_every=20)
    assert hist[-1]["평가"] < hist[0]["평가"] - 1.0
    assert all(math.isfinite(r["grad_norm"]) for r in hist[1:])


def test_scheduler_ends_at_zero_and_clipping_recorded():
    blocks = torch.randint(0, 64, (4, 16))
    hist = train(small_tiny(), blocks, {}, steps=10, warmup=2, log_every=5, clip=0.5)
    assert hist[-1]["lr"] == pytest.approx(0.0, abs=1e-12)
    assert hist[1]["grad_norm"] > 0                     # 자르기 전 노름을 기록


def test_frozen_parameters_are_not_trained():
    model = small_tiny()
    emb_before = model.model.embed_tokens.weight.detach().clone()
    model.model.embed_tokens.weight.requires_grad_(False)
    train(model, torch.randint(0, 64, (4, 16)), {}, steps=5, warmup=1, log_every=5)
    assert torch.equal(model.model.embed_tokens.weight, emb_before)


def test_save_reload_identical(tmp_path):
    from transformers import Qwen2ForCausalLM
    model = small_tiny().eval()
    model.save_pretrained(tmp_path)
    again = Qwen2ForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).eval()
    ids = torch.randint(0, 512, (1, 12))
    with torch.no_grad():
        assert torch.equal(model(ids).logits, again(ids).logits)


def test_recorded_history():
    path = HERE / "history.json"
    if not path.exists():
        pytest.skip("pretrain_tiny.py를 먼저 실행")
    h = json.loads(path.read_text(encoding="utf-8"))
    pre, ft = h["pretrain"], h["full_ft"]
    assert pre[0]["일반평가"] == pytest.approx(math.log(151936), abs=0.2)
    assert min(r["일반평가"] for r in pre) < 6.0                    # 사전학습으로 크게 감소
    assert ft[-1]["도메인평가"] < ft[0]["도메인평가"] - 1.0           # 도메인 학습 효과
    assert ft[-1]["일반평가"] > ft[0]["일반평가"]                    # 일반 능력 일부 손실(망각)
