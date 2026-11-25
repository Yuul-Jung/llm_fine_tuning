# 제43장 검증 테스트
#
# 실행: python -m pytest examples/ch43 -q

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch41"):
    sys.path.insert(0, str(p))

from check_dataset import load  # noqa: E402
from clean_split import (NGRAM_T, group_split, groups_from_pairs, jaccard,  # noqa: E402
                         ngrams, text_hash)
from criteria import format_ok, keyword_hit, load_criteria  # noqa: E402

ROOT = HERE.parents[1]


def test_hash_and_ngram_detect_copies():
    a = "논블로킹 할당은 언제 쓰나요?"
    assert text_hash(a) == text_hash("논블로킹  할당은 언제 쓰나요 ?")
    assert jaccard(ngrams(a), ngrams("논블로킹 할당은 언제 쓰나요? 예를 들어")) >= NGRAM_T
    assert jaccard(ngrams(a), ngrams("임계 경로란 무엇인가요?")) < 0.1


def test_group_split_keeps_groups_together():
    rows = load()
    ids = [r["id"] for r in rows]
    group = groups_from_pairs(ids, [("fpga-010", "fpga-001"), ("fpga-001", "fpga-002")])
    split = group_split(rows, group)
    assert len({split[i] for i in ("fpga-010", "fpga-001", "fpga-002")}) == 1
    assert group_split(rows, group) == split


def test_committed_split_is_balanced_and_leak_free():
    rows = load()
    split = json.loads((ROOT / "datasets/fpga_qa/split_v1.json").read_text(encoding="utf-8"))["split"]
    assert sorted(split) == sorted(r["id"] for r in rows)
    for s in ("validation", "test"):
        cats = [r["category"] for r in rows if split[r["id"]] == s]
        assert len(cats) == len(set(cats)) == 6              # 분류마다 정확히 하나
    grams = {r["id"]: ngrams(r["instruction"] + " " + r["input"]) for r in rows}
    train = [i for i in split if split[i] == "train"]
    assert all(jaccard(grams[i], grams[t]) < NGRAM_T for i in split if split[i] != "train" for t in train)


def test_criteria_file_and_scorers():
    c = load_criteria()
    split = json.loads((ROOT / c["data"]["split"]).read_text(encoding="utf-8"))["split"]
    assert sorted(c["keywords"]) == sorted(i for i, s in split.items() if s == "test")
    assert keyword_hit("상태 레지스터와 다음 상태 논리", ["상태 레지스터", "다음 상태", "default"]) == 2 / 3
    assert format_ok("임계 경로는 가장 긴 경로입니다.", 4)
    assert not format_ok("가장 긴 경로입니다!", 4)
    assert not format_ok("가. 나. 다. 라. 마입니다.", 4)
