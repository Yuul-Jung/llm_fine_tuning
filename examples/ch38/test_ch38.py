# 제38장 검증 테스트
#
# 실행: python -m pytest examples/ch38 -q

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch19", HERE.parent / "ch22", HERE.parent / "ch37"):
    sys.path.insert(0, str(p))

pytest.importorskip("peft")
pytest.importorskip("bitsandbytes")

from qlora_memory import saved_bytes_of_step  # noqa: E402


def tiny():
    """어휘를 512개로 줄인 Qwen2-Tiny 구조(Logits가 Activation 비교를 가리지 않게)"""
    from tiny_model import TINY
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    return Qwen2ForCausalLM(Qwen2Config(**{**TINY, "vocab_size": 512}, tie_word_embeddings=True))


def test_checkpointing_saves_memory_same_loss():
    ids = torch.randint(0, 512, (1, 128))
    a = tiny()
    saved0, _, loss0 = saved_bytes_of_step(a, ids)
    b = tiny()
    b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    saved1, _, loss1 = saved_bytes_of_step(b, ids)
    assert saved1 < 0.6 * saved0
    assert loss1 == pytest.approx(loss0, abs=1e-5)
    for pa, pb in zip(a.parameters(), b.parameters()):        # 같은 Gradient
        torch.testing.assert_close(pa.grad, pb.grad, atol=1e-5, rtol=1e-4)


def test_parameters_not_counted_as_activation():
    lin = torch.nn.Linear(1000, 1000)
    x = torch.randn(1, 1000, requires_grad=True)

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = lin

        def forward(self, input_ids, labels):
            return type("O", (), {"loss": self.lin(x).sum()})()

    saved, _, _ = saved_bytes_of_step(Wrap(), None)
    assert saved < 1000 * 1000 * 4                           # 가중치(4MB)는 세지 않는다


def test_qlora_trainable_count():
    try:
        from qlora_memory import build_qlora
        model, parts = build_qlora("cpu", checkpointing=True)
    except Exception:
        pytest.skip("Qwen 모델이 아직 내려받아지지 않음")
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 540_672   # 제28장 계산
    assert parts["Embedding(= LM Head)"] == 151_936 * 896 * 4
