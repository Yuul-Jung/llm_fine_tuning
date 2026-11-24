# 제39장 검증 테스트
#
# 실행: python -m pytest examples/ch39 -q
#
# 0.5B QLoRA 학습(수 분)은 qlora_train.py로 실행하고, 테스트는 사전학습한 Qwen2-Tiny(제25장)를
# 4비트로 불러와 같은 절차를 짧게 확인한다.

import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch22", HERE.parent / "ch24", HERE.parent / "ch25",
          HERE.parent / "ch37", HERE.parent / "ch38"):
    sys.path.insert(0, str(p))

pytest.importorskip("peft")
pytest.importorskip("bitsandbytes")

from qlora_train import frozen_4bit, keep_training_layout  # noqa: E402

TINY = HERE.parents[1] / "models" / "qwen2-tiny-pretrained"


@pytest.fixture()
def tiny_qlora():
    if not TINY.exists():
        pytest.skip("제25장 pretrain_tiny.py를 먼저 실행")
    from load_4bit import nf4_config
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(TINY, quantization_config=nf4_config(), dtype=torch.float32,
                                             device_map="cpu")
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True,
                                        gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.manual_seed(0)
    m = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                     lora_dropout=0.0, task_type="CAUSAL_LM"))
    keep_training_layout(m)
    return m


def test_qlora_steps_keep_4bit_frozen(tiny_qlora, tmp_path):
    from trainer import eval_loss, train
    ids = torch.randint(0, 5000, (4, 32))
    before = frozen_4bit(tiny_qlora)
    assert len(before) == 2 * 4 * 7                       # 4계층 × Linear 7종 × (번호, absmax)
    l0 = eval_loss(tiny_qlora, ids, "cpu")                 # 평가 모드를 거쳐도 학습이 가능해야 한다
    train(tiny_qlora, ids, {}, steps=3, batch=1, accum=2, lr=1e-2, warmup=1, log_every=3)
    after = frozen_4bit(tiny_qlora)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert math.isfinite(eval_loss(tiny_qlora, ids, "cpu")) and math.isfinite(l0)
    tiny_qlora.save_pretrained(tmp_path)
    names = {f.name for f in tmp_path.iterdir()}
    assert "adapter_model.safetensors" in names and "model.safetensors" not in names
    assert (tmp_path / "adapter_model.safetensors").stat().st_size < 200_000   # LoRA 26,624개 × 4바이트 + 헤더


def test_keep_training_layout_counts(tiny_qlora):
    assert keep_training_layout(tiny_qlora) == 4 * 7
