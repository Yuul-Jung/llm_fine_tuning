# 제55장 검증 테스트
#
# 실행: python -m pytest examples/ch55 -q

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("gguf")

from gguf import GGMLQuantizationType, quants  # noqa: E402

from quant_compare import OUT, q4_0_by_hand, q8_0_by_hand  # noqa: E402


@pytest.mark.parametrize("name, fn, nbytes", [("Q8_0", q8_0_by_hand, 34), ("Q4_0", q4_0_by_hand, 18)])
def test_hand_implementation_matches_library(name, fn, nbytes):
    x = np.random.default_rng(1).standard_normal((4, 64)).astype(np.float32)
    q = GGMLQuantizationType[name]
    packed = quants.quantize(x, q)
    assert packed.shape == (4, 2 * nbytes)                                  # 64개 = 블록 2개
    assert np.array_equal(fn(x), quants.dequantize(packed, q))


def test_recorded_comparison():
    r = json.loads(OUT.read_text(encoding="utf-8"))
    sizes = [r[f]["size_mib"] for f in ("F32", "F16", "Q8_0", "Q4_0")]
    assert sizes == sorted(sizes, reverse=True)
    assert abs(r["F32"]["domain_ppl"] - 12.0587) < 0.01                    # F32 GGUF = 병합 모델(제52장)
    assert r["F16"]["passed"] == r["Q8_0"]["passed"] == r["F32"]["passed"] > r["Q4_0"]["passed"]
    assert r["Q4_0"]["general_ppl"] > 1.3 * r["F32"]["general_ppl"]
