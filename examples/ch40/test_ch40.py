# 제40장 검증 테스트
#
# 실행: python -m pytest examples/ch40 -q

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.json"


@pytest.fixture(scope="module")
def results():
    if not RESULTS.exists():
        pytest.skip("compare_methods.py를 먼저 실행")
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_comparison_a_table(results):
    a = {r["방식"]: r for r in results["A"]}
    assert a["Full FT"]["학습 파라미터"] == 41_914_112
    assert a["LoRA"]["학습 파라미터"] == a["QLoRA"]["학습 파라미터"] == 151_552   # 제29·30장
    assert a["Full FT"]["고정 메모리MiB"] > 3 * a["LoRA"]["고정 메모리MiB"]
    assert a["QLoRA"]["고정 메모리MiB"] < a["LoRA"]["고정 메모리MiB"]
    assert a["Full FT"]["저장 MiB"] > 100 * a["LoRA"]["저장 MiB"]
    assert a["Full FT"]["도메인 PPL"] < a["LoRA"]["도메인 PPL"] < 884.9          # 모두 학습됨


def test_comparison_b_table(results):
    if "B" not in results:
        pytest.skip("비교 B를 실행하지 않음(--only-a)")
    b = {r["방식"]: r for r in results["B"]}
    assert b["QLoRA(NF4)"]["학습 파라미터"] == b["LoRA(FP32)"]["학습 파라미터"] == 540_672
    assert b["QLoRA(NF4)"]["고정 메모리MiB"] < 0.5 * b["LoRA(FP32)"]["고정 메모리MiB"]
    for r in b.values():
        assert r["평가 PPL"] < 21.45                                               # 제22장 기준보다 낮아짐
