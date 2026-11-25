# 제56장 검증 테스트
#
# 실행: python -m pytest examples/ch56 -q

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "ch41"):
    sys.path.insert(0, str(p))

import pytest  # noqa: E402

from check_dataset import load, to_structures  # noqa: E402
from modelfile import MERGED, SYSTEM, modelfile, parse_parameters, render, render_go_like  # noqa: E402

pytestmark = pytest.mark.skipif(not MERGED.exists(), reason="제52장의 병합 모델 필요")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MERGED)


def test_template_matches_training_for_all_questions(tok):
    for r in load():
        q = to_structures(r)["messages"]["messages"][:1]
        assert render_go_like(SYSTEM, q) == render(tok, [{"role": "system", "content": SYSTEM}] + q, gen=True)


def test_parameters_follow_generation_config():
    gen = json.loads((MERGED / "generation_config.json").read_text())
    p = parse_parameters(modelfile("x.gguf", gen))
    assert p["temperature"] == ["0"] and float(p["repeat_penalty"][0]) == gen["repetition_penalty"]
    assert p["repeat_last_n"] == ["-1"] and set(p["stop"]) == {"<|im_end|>", "<|endoftext|>"}


def test_modelfile_written_and_points_to_q8_0():
    text = (MERGED.parent / "gguf" / "Modelfile").read_text(encoding="utf-8")
    assert text.startswith("FROM ./fpga-q8_0.gguf") and (MERGED.parent / "gguf" / "fpga-q8_0.gguf").exists()
