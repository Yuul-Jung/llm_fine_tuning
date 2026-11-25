# 제41장 검증 테스트
#
# 실행: python -m pytest examples/ch41 -q

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_dataset import PLAN, load, problems, to_structures  # noqa: E402

ROWS = load()


def test_dataset_passes_all_rules():
    assert len(ROWS) == 50 and problems(ROWS) == []


def test_checker_catches_violations():
    bad = copy.deepcopy(ROWS)
    del bad[0]["output"]                              # 필수 항목 누락
    bad[1]["category"] = "기타"                       # 설계표에 없는 분류
    bad[2]["output"] = "답변 끝맺음 없음"             # 문체 위반
    bad[3]["id"] = bad[4]["id"]                       # id 중복
    found = " ".join(problems(bad))
    for word in ("항목 없음", "설계표에 없는 분류", "끝나지 않음", "id 중복"):
        assert word in found


def test_three_structures_carry_same_content():
    r = next(x for x in ROWS if x["input"])
    s = to_structures(r)
    assert s["prompt_completion"]["completion"] == s["messages"]["messages"][1]["content"] == r["output"]
    assert r["input"] in s["prompt_completion"]["prompt"]
    assert [m["role"] for m in s["messages"]["messages"]] == ["user", "assistant"]


def test_license_and_card_recorded():
    root = Path(__file__).resolve().parents[2]
    lic = (root / "datasets" / "LICENSES.md").read_text(encoding="utf-8")
    card = (root / "datasets" / "fpga_qa" / "README.md").read_text(encoding="utf-8")
    assert "fpga_qa.json" in lic and "출처" in card and "라이선스" in card
    assert all(c in card for c in PLAN)
