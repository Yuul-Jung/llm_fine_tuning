# 제49장 검증 테스트
#
# 실행: python -m pytest examples/ch49 -q

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MiB = 2 ** 20


def rec():
    return json.loads((HERE / "qlora.json").read_text(encoding="utf-8"))


def test_no_nonfinite_and_loss_decreases():
    r = rec()
    assert r["nonfinite"] == 0
    assert r["curve"][-1][1] < r["curve"][0][1]


def test_budget_within_2gb():
    b = rec()["budget"]
    total = b["weights"] + b["lora_states"] + b["act_real"]
    assert total < 2048 * MiB
    assert b["act_real"] < b["act_fixed256"] and b["longest"] < 256       # 실제 데이터는 256보다 짧다


def test_same_trainable_and_eval_as_lora():
    r = rec()
    lora = json.loads((HERE.parent / "ch48" / "domain.json").read_text(encoding="utf-8"))
    assert r["trainable"] == lora["trainable"]                             # 같은 LoRA 구성
    assert r["after"]["generation"] == lora["after"]["generation"]
    assert sorted(r["after"]["answers"]) == sorted(lora["after"]["answers"])
    assert set(r["verdict"]) == set(lora["verdict"])
