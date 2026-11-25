# 제58장 검증 테스트
#
# 실행: python -m pytest examples/ch58 -q

import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ondevice import OUT, fits, kv_mib, weight_mib  # noqa: E402

QWEN = SimpleNamespace(hidden_size=896, num_attention_heads=14, num_key_value_heads=2, num_hidden_layers=24)


def test_kv_cache_formula():
    assert kv_mib(QWEN, 1, 4) * 2 ** 20 == 2 * 24 * 2 * 64 * 4
    assert kv_mib(QWEN, 32768, 2) == 384.0


def test_block_format_sizes():
    n = 32 * 1000
    assert weight_mib(n, "Q8_0") * 2 ** 20 == 1000 * 34 and weight_mib(n, "Q4_0") * 2 ** 20 == 1000 * 18


def test_fit_rule_and_records():
    assert fits(1536, 2048) and not fits(1537, 2048)
    r = json.loads(OUT.read_text(encoding="utf-8"))
    assert not r["fits"]["GT 1030"]["FP32@2048"] and r["fits"]["GT 1030"]["Q4_0@32768"]
    for c in r["cpu"]:
        assert abs(c["kv_mib"] - c["kv_formula_mib"]) < 1e-6                  # 실제 KV Cache = 계산
