# 제47장 검증 테스트
#
# 실행: python -m pytest examples/ch47 -q

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "ch46"):
    sys.path.insert(0, str(p))

import pytest  # noqa: E402

pytest.importorskip("trl")

from sft_loop import TINY, setup  # noqa: E402
from sft_trainer import SAME_AS_CH46, removed_names, to_hf_dataset  # noqa: E402

ROOT = HERE.parents[1]


def test_config_matches_ch46_and_fp32():
    assert SAME_AS_CH46["bf16"] is False and SAME_AS_CH46["fp16"] is False
    assert (SAME_AS_CH46["per_device_train_batch_size"], SAME_AS_CH46["gradient_accumulation_steps"],
            SAME_AS_CH46["max_steps"], SAME_AS_CH46["completion_only_loss"]) == (4, 2, 30, True)


def test_removed_apis_are_flagged():
    assert all(state == "제거됨" for _, state in removed_names())


@pytest.mark.skipif(not TINY.exists(), reason="제25장의 Qwen2-Tiny가 필요")
def test_trl_batches_have_same_labels(tmp_path):
    from trl import SFTConfig, SFTTrainer
    tok, model, tr, va = setup()
    args = SFTConfig(output_dir=str(tmp_path), use_cpu=True, **SAME_AS_CH46)
    trainer = SFTTrainer(model=model, args=args, processing_class=tok,
                         train_dataset=to_hf_dataset(tr), eval_dataset=to_hf_dataset(va))
    ours = {tuple(e["input_ids"]): e["labels"] for e in tr.items}
    batch = next(iter(trainer.get_train_dataloader()))
    for ids, lab, att in zip(batch["input_ids"], batch["labels"], batch["attention_mask"]):
        n = int(att.sum())
        assert lab[:n].tolist() == ours[tuple(ids[:n].tolist())]       # TRL이 만든 labels = 제45장 labels
        assert (lab[n:] == -100).all()


def test_recorded_comparison_and_baseline():
    log = json.loads((HERE / "trainer_log.json").read_text())
    ours = [json.loads(x) for x in (HERE.parent / "ch46" / "log.jsonl").read_text().splitlines()]
    trl_val, our_val = log["eval"]["30"], ours[-1]["val_loss"]
    assert log["eval"]["30"] < log["eval"]["5"] and abs(trl_val - our_val) / our_val < 0.05   # 5% 안에서 일치
    base = json.loads((HERE / "baseline.json").read_text(encoding="utf-8"))
    ch22 = json.loads((ROOT / "examples/ch22/baseline_ppl.json").read_text(encoding="utf-8"))
    assert abs(base["general_ppl"] - 21.45) < 0.01 and "21.45" in json.dumps(ch22)
    assert sorted(base["answers"]) == sorted(base["keyword_hit"]) and len(base["answers"]) == 6
