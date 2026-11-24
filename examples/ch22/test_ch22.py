# 제22장 검증 테스트
#
# 실행: python -m pytest examples/ch22 -q

import json
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch19", HERE.parent / "ch21"):
    sys.path.insert(0, str(p))

from loss_basics import IGNORE, lm_loss, nll_by_hand, shift  # noqa: E402

BASELINE = json.loads((HERE / "baseline_ppl.json").read_text(encoding="utf-8"))


def test_nll_by_hand_matches_pytorch():
    logits = [2.0, 1.0, 3.0, 0.5]
    for t in range(4):
        expect = F.cross_entropy(torch.tensor([logits]), torch.tensor([t])).item()
        assert nll_by_hand(logits, t) == pytest.approx(expect, abs=1e-6)


def test_shift_aligns_next_token():
    inputs, labels = shift(torch.tensor([[1, 2, 3, 4]]))
    assert inputs.tolist() == [[1, 2, 3]] and labels.tolist() == [[2, 3, 4]]


def test_ignore_index_excludes_positions():
    torch.manual_seed(1)
    lg, lab = torch.randn(2, 5, 7), torch.randint(0, 7, (2, 5))
    masked = lab.clone()
    masked[:, 3:] = IGNORE
    s, n = lm_loss(lg, masked)
    assert n == 6
    expect = F.cross_entropy(lg[:, :3].reshape(-1, 7), lab[:, :3].reshape(-1))
    assert s.item() / n == pytest.approx(expect.item(), abs=1e-5)


def test_uniform_model_ppl_equals_vocab():
    lg, lab = torch.zeros(1, 10, 50), torch.randint(0, 50, (1, 10))
    s, n = lm_loss(lg, lab)
    assert math.exp(s.item() / n) == pytest.approx(50, rel=1e-5)


def test_baseline_record():
    assert BASELINE["data"]["fingerprint"] == "1d8635627d660631"
    r = BASELINE["results"]
    assert all(v["tokens"] == 5 * 127 for v in r.values())
    assert r["Qwen2.5-0.5B Base"]["ppl"] < r["Qwen2.5-0.5B-Instruct"]["ppl"] < 100
    assert r["Qwen2-Tiny(무작위)"]["loss"] == pytest.approx(math.log(151936), abs=0.2)


@pytest.fixture(scope="module")
def tok():
    try:
        from common.qwen import load_tokenizer
        return load_tokenizer(True)
    except Exception:
        pytest.skip("Qwen 토크나이저가 아직 내려받아지지 않음")


def test_tiny_baseline_reproduces(tok):
    from common.fixed_data import load_blocks
    from hf_compat import to_hf
    from qwen_ppl import evaluate
    from tiny_model import Qwen2Tiny, init_weights
    r = evaluate(to_hf(init_weights(Qwen2Tiny(), seed=0)), load_blocks(tok, "eval"))
    assert round(r["loss"], 4) == BASELINE["results"]["Qwen2-Tiny(무작위)"]["loss"]
    for hf, mine in r["hf_vs_mine"]:
        assert hf == pytest.approx(mine, abs=1e-4)


def test_instruct_loss_matches_manual(tok):
    try:
        from common.qwen import load_model
        model = load_model(True)
    except Exception:
        pytest.skip("Qwen 모델이 아직 내려받아지지 않음")
    from common.fixed_data import load_blocks
    from qwen_ppl import evaluate
    r = evaluate(model, load_blocks(tok, "eval")[:1])
    hf, mine = r["hf_vs_mine"][0]
    assert hf == pytest.approx(mine, abs=1e-4)
