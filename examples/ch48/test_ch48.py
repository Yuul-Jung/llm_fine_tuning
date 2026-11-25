# 제48장 검증 테스트
#
# 실행: python -m pytest examples/ch48 -q

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def lora_params(c: dict, targets: list, r: int = 8) -> int:
    """LoRA 파라미터 = 층 수 × Σ r × (입력 + 출력)"""
    h, kv, i = c["hidden_size"], c["num_key_value_heads"] * (c["hidden_size"] // c["num_attention_heads"]), \
        c["intermediate_size"]
    shape = {"q_proj": (h, h), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (h, h),
             "gate_proj": (h, i), "up_proj": (h, i), "down_proj": (i, h)}
    return c["num_hidden_layers"] * sum(r * sum(shape[t]) for t in targets)


def test_scope_results():
    s = json.loads((HERE / "scope.json").read_text(encoding="utf-8"))
    n = [v["trainable"] for v in s.values()]
    assert n == sorted(n) and len(set(n)) == 3                       # 범위가 넓을수록 파라미터가 많다
    assert all(v["val_after"] < v["val_before"] for v in s.values())
    assert [v["val_after"] for v in s.values()] == sorted((v["val_after"] for v in s.values()), reverse=True)


def test_trainable_count_matches_formula():
    qwen = {"hidden_size": 896, "num_attention_heads": 14, "num_key_value_heads": 2,
            "intermediate_size": 4864, "num_hidden_layers": 24}
    d = json.loads((HERE / "domain.json").read_text(encoding="utf-8"))
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert d["trainable"] == lora_params(qwen, targets) == 4_399_104


def test_domain_verdict_recorded():
    d = json.loads((HERE / "domain.json").read_text(encoding="utf-8"))
    base = json.loads((HERE.parent / "ch47" / "baseline.json").read_text(encoding="utf-8"))
    assert set(d["verdict"]) == {"domain_ppl", "keyword_hit", "format_ok", "general_ppl"}
    assert d["after"]["domain_ppl"] < base["domain_ppl"]               # 검증 손실 감소
    assert sorted(d["after"]["answers"]) == sorted(base["answers"])     # 같은 test 문항으로 비교
    assert d["after"]["generation"] == base["generation"]               # 같은 Greedy 설정
