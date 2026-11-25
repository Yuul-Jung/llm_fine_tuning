# 제60장 검증 테스트
#
# 실행: python -m pytest examples/ch60 -q

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from project import FILES, OUT, ROOT, STAGES, sha  # noqa: E402


def rec():
    return json.loads(OUT.read_text(encoding="utf-8"))


def test_input_hashes_match_repository():
    r = rec()
    assert all(r["hashes"][k] == sha(ROOT / v) for k, v in FILES.items())     # 기록 이후 데이터·기준 변경 없음


def test_reproduced_and_recorded():
    r = rec()
    assert r["reproduced"]["training_loss_same"] is True
    assert abs(r["reproduced"]["merged_ppl"] - 12.0587) < 1e-3
    assert r["seed"] == 0 and r["base_revision"] and r["versions"]["trl"]


def test_final_and_unverified_items_are_explicit():
    r = rec()
    assert r["final_test"]["adopted"] is False and r["final_test"]["chosen"] == "원본-15"
    states = " ".join(s for _, _, s in STAGES)
    assert "미검증" in states and "허가 대기" in states
