# 제46장 검증 테스트
#
# 실행: python -m pytest examples/ch46 -q

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pytest  # noqa: E402
import torch  # noqa: E402

from sft_loop import IGNORE, LOG, TINY, loader, setup, train_sft, validate  # noqa: E402

pytestmark = pytest.mark.skipif(not TINY.exists(), reason="제25장의 Qwen2-Tiny가 필요")


@pytest.fixture(scope="module")
def parts():
    return setup()


def test_collator_pads_and_masks(parts):
    _, _, tr, _ = parts
    b = next(iter(loader(tr, 4, 0, 0)))
    pad = b["attention_mask"] == 0
    assert b["input_ids"].shape == b["labels"].shape and (b["labels"][pad] == IGNORE).all()
    assert [x.tolist() for x in next(iter(loader(tr, 4, 0, 0)))["input_ids"]] == b["input_ids"].tolist()


def test_validation_is_token_weighted(parts):
    _, model, _, va = parts
    v = validate(model, va)
    assert v["val_tokens"] == sum(sum(t != IGNORE for t in e["labels"][1:]) for e in va.items)
    assert abs(v["val_ppl"] - torch.tensor(v["val_loss"]).exp().item()) < 1e-6 * v["val_ppl"]


def test_resume_continues_exactly(tmp_path):
    _, m1, tr, va = setup()
    full = train_sft(m1, tr, va, steps=4, log_every=1, ckpt=tmp_path / "c", save_at=(2,))
    _, m2, _, _ = setup(tmp_path / "c")
    resumed = train_sft(m2, tr, va, steps=4, log_every=1, ckpt=tmp_path / "c", resume=True)
    assert [round(h["train_loss"], 5) for h in full[2:]] == [round(h["train_loss"], 5) for h in resumed]
    assert full[-1]["train_loss"] < full[0]["train_loss"]


def test_recorded_log_decreases():
    rows = [json.loads(x) for x in LOG.read_text().splitlines()]
    assert [r["step"] for r in rows] == [5, 10, 15, 20, 25, 30]
    assert rows[-1]["val_loss"] < rows[0]["val_loss"] and rows[-1]["train_loss"] < rows[0]["train_loss"]
