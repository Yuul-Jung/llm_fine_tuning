# 제50장 검증 테스트
#
# 실행: python -m pytest examples/ch50 -q

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from improve import ERRORS, FINAL, NGRAM_T, RUNS, augmented_rows, leak_check  # noqa: E402


def test_augmented_data_is_train_only_and_leak_free():
    rows = augmented_rows()
    split = json.loads((HERE.parents[1] / "datasets/fpga_qa/split_v1.json").read_text(encoding="utf-8"))["split"]
    assert len(rows) == 76 and len({r["id"] for r in rows}) == 76
    assert all(split[r["id"].removesuffix("-p1")] == "train" for r in rows)      # 원본이 모두 train
    assert leak_check(rows) < NGRAM_T


def test_error_table_covers_validation():
    split = json.loads((HERE.parents[1] / "datasets/fpga_qa/split_v1.json").read_text(encoding="utf-8"))["split"]
    assert sorted(ERRORS) == sorted(i for i, s in split.items() if s == "validation")


def test_candidates_recorded_and_final_once():
    runs = json.loads(RUNS.read_text(encoding="utf-8"))
    cands = ["원본-15", "원본-30", "보강-15", "보강-30"]
    assert all(k in runs and set(runs[k]["verdict"]) == {"domain_ppl", "keyword_hit", "format_ok", "general_ppl"}
               for k in cands)
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    best = max(cands, key=lambda k: (runs[k]["passed"], runs[k]["keyword"], -runs[k]["domain_ppl"]))
    assert final["chosen"] == best                                             # validation 규칙으로 고른 모델
    assert final["result"]["split"] == "test" and len(final["result"]["answers"]) == 6


def test_reproduces_ch48():
    runs = json.loads(RUNS.read_text(encoding="utf-8"))
    ch48 = json.loads((HERE.parent / "ch48" / "domain.json").read_text(encoding="utf-8"))
    assert abs(runs["원본-30"]["domain_ppl"] - ch48["after"]["domain_ppl"]) < 0.05   # 같은 설정 = 같은 결과
