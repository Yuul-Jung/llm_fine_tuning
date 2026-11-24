# 제30장 검증 테스트
#
# 실행: python -m pytest examples/ch30 -q

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch19", HERE.parent / "ch24", HERE.parent / "ch25",
          HERE.parent / "ch29"):
    sys.path.insert(0, str(p))

peft = pytest.importorskip("peft")

from lora_linear import ALL_LINEAR, apply_lora, trainable_count  # noqa: E402
from peft_lora import config, copy_A_from  # noqa: E402


@pytest.fixture()
def tiny_pair():
    from hf_compat import to_hf
    from tiny_model import Qwen2Tiny, init_weights
    return to_hf(init_weights(Qwen2Tiny(), seed=0)), to_hf(init_weights(Qwen2Tiny(), seed=0))


def test_trainable_counts_match_scratch(tiny_pair):
    a, b = tiny_pair
    pm = peft.get_peft_model(a, config(ALL_LINEAR))
    apply_lora(b, ALL_LINEAR)
    assert trainable_count(pm) == trainable_count(b) == 151_552


def test_names_and_shapes_correspond(tiny_pair):
    a, b = tiny_pair
    pm = peft.get_peft_model(a, config())
    apply_lora(b)
    ours = {n: tuple(p.shape) for n, p in b.named_parameters() if p.requires_grad}
    theirs = {n.replace("base_model.model.", "").replace(".default.weight", ""): tuple(p.shape)
              for n, p in pm.named_parameters() if p.requires_grad}
    assert ours == theirs                       # 같은 이름(접두어·접미어 제외), 같은 Shape


def test_same_start_same_training(tiny_pair):
    from trainer import train
    a, b = tiny_pair
    pm = peft.get_peft_model(a, config())
    apply_lora(b, seed=0)
    copy_A_from(b, pm)
    ids = torch.randint(0, 3000, (4, 24))
    with torch.no_grad():
        torch.testing.assert_close(pm(ids[:1]).logits, b(ids[:1]).logits)
    train(pm, ids, {}, steps=5, lr=1e-2, warmup=1, log_every=5)
    train(b, ids, {}, steps=5, lr=1e-2, warmup=1, log_every=5)
    with torch.no_grad():
        torch.testing.assert_close(pm(ids[:1]).logits, b(ids[:1]).logits, atol=1e-4, rtol=1e-4)


def test_adapter_save_reload(tiny_pair, tmp_path):
    a, b = tiny_pair
    pm = peft.get_peft_model(a, config())
    with torch.no_grad():
        pm.get_submodule("base_model.model.model.layers.0.self_attn.q_proj").lora_B["default"].weight.normal_()
    pm.save_pretrained(tmp_path)
    assert {"adapter_config.json", "adapter_model.safetensors"} <= {f.name for f in tmp_path.iterdir()}
    again = peft.PeftModel.from_pretrained(b, tmp_path)
    ids = torch.randint(0, 3000, (1, 10))
    with torch.no_grad():
        assert torch.equal(pm(ids).logits, again(ids).logits)
