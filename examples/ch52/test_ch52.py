# 제52장 검증 테스트
#
# 실행: python -m pytest examples/ch52 -q

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from merge import OUT  # noqa: E402

TOL = 1e-3                                         # 병합 허용 오차(Logits 최대 차이)


def test_merge_formula_on_toy_layer():
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(16, 12))
    peft = get_peft_model(net, LoraConfig(r=4, lora_alpha=8, target_modules=["0"], init_lora_weights=False))
    layer = peft.base_model.model[0]
    expect = layer.base_layer.weight + 2.0 * layer.lora_B["default"].weight @ layer.lora_A["default"].weight
    x = torch.randn(3, 16)
    y = peft(x)
    merged = peft.merge_and_unload()
    assert torch.allclose(merged[0].weight, expect, atol=1e-6) and torch.allclose(merged(x), y, atol=1e-5)


def test_recorded_merge_errors():
    r = json.loads(OUT.read_text())
    assert r["tiny"]["weight_diff"] < 1e-6 and r["tiny"]["logits_diff"] < TOL
    assert r["qlora"]["B_C"] < TOL and r["final"]["logits_diff"] < TOL              # B↔C 허용 오차 이내
    assert abs(r["qlora"]["ppl"]["B"] - r["qlora"]["ppl"]["C"]) < 0.01
    assert r["qlora"]["A_B"] > 100 * r["qlora"]["B_C"]                              # A↔B는 따로 기록할 큰 차이
    assert r["final"]["params"] == 494_032_768
