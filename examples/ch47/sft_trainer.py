# 제47장 핵심 실습: 제46장과 같은 조건으로 TRL SFTTrainer를 실행해 두 결과를 비교한다
#
# 실행: python examples/ch47/sft_trainer.py   (제25장의 models/qwen2-tiny-pretrained 필요, CPU 약 3분)

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch46"))

import torch  # noqa: E402

from sft_loop import IGNORE, LOG, setup, validate  # noqa: E402      (제46장)

OUT = HERE / "trainer_log.json"

# 제46장의 코드 → SFTConfig 항목(같은 값)
SAME_AS_CH46 = {
    "per_device_train_batch_size": 4,     # loader(ds, batch=4)
    "gradient_accumulation_steps": 2,     # for _ in range(accum)
    "max_steps": 30,                      # steps
    "learning_rate": 1e-3,                # AdamW(lr)
    "weight_decay": 0.1,                  # AdamW(weight_decay)
    "max_grad_norm": 1.0,                 # clip_grad_norm_
    "lr_scheduler_type": "cosine",        # warmup_cosine
    "warmup_steps": 6,                    # warmup
    "max_length": 256,                    # build_example(max_len)
    "completion_only_loss": True,         # labels의 -100(completion_mask)
    "eval_strategy": "steps",             # 5단계마다 validate
    "eval_steps": 5,
    "logging_steps": 5,                   # log_every
    "per_device_eval_batch_size": 1,      # 검증은 샘플 하나씩
    "dataloader_drop_last": True,         # DataLoader(drop_last)
    "seed": 0,
    # CPU·GT 1030 경로: FP32. 이 버전의 기본값은 bf16 켜짐, Gradient Checkpointing 켜짐
    "bf16": False,
    "fp16": False,
    "gradient_checkpointing": False,
    "save_strategy": "no",
    "report_to": "none",
}


def to_hf_dataset(ds):
    """제45장의 labels를 TRL이 읽는 열로: -100이 아닌 자리 = completion_mask 1"""
    from datasets import Dataset
    return Dataset.from_list([{"input_ids": e["input_ids"],
                               "completion_mask": [int(t != IGNORE) for t in e["labels"]]} for e in ds.items])


def defaults() -> dict:
    from trl import SFTConfig
    c = SFTConfig(output_dir="/tmp/ch47_probe", use_cpu=True)
    return {k: getattr(getattr(c, k), "value", getattr(c, k)) for k in ("bf16", "fp16", "gradient_checkpointing", "learning_rate",
                                       "lr_scheduler_type", "per_device_train_batch_size", "max_length")}


def removed_names() -> list:
    """예전 자료에 흔한 이름이 이 버전에서 그대로 쓰이는지 확인한다"""
    import trl
    from trl import SFTConfig, SFTTrainer
    import inspect
    checks = []
    for key in ("max_seq_length", "warmup_ratio", "evaluation_strategy"):
        try:
            SFTConfig(output_dir="/tmp/ch47_probe", bf16=False, **{key: 1 if key != "evaluation_strategy" else "no"})
            checks.append((f"SFTConfig({key}=…)", "사용 가능"))
        except TypeError:
            checks.append((f"SFTConfig({key}=…)", "제거됨"))
    checks.append(("SFTTrainer(tokenizer=…)",
                   "사용 가능" if "tokenizer" in inspect.signature(SFTTrainer.__init__).parameters else "제거됨"))
    checks.append(("trl.DataCollatorForCompletionOnlyLM",
                   "사용 가능" if hasattr(trl, "DataCollatorForCompletionOnlyLM") else "제거됨"))
    return checks


def run_trainer(model, tok, tr, va, out_dir: str = "/tmp/ch47_run") -> list:
    from trl import SFTConfig, SFTTrainer
    args = SFTConfig(output_dir=out_dir, use_cpu=True, **SAME_AS_CH46)
    trainer = SFTTrainer(model=model, args=args, processing_class=tok,
                         train_dataset=to_hf_dataset(tr), eval_dataset=to_hf_dataset(va))
    trainer.train()
    return trainer.state.log_history


def main() -> None:
    import transformers
    import trl
    print(f"── 1. 버전과 기본값(TRL {trl.__version__}, Transformers {transformers.__version__}) ──")
    for k, v in defaults().items():
        print(f"  {k:<28}기본 {v!s:<8}→ 이 교재 {SAME_AS_CH46.get(k, v)}")

    print("\n── 2. 예전 API 확인 ──")
    for name, state in removed_names():
        print(f"  {name:<38}{state}")

    print("\n── 3. SFTTrainer로 30단계(제46장과 같은 모델·데이터·설정) ──")
    tok, model, tr, va = setup()
    logs = run_trainer(model, tok, tr, va)
    train = {r["step"]: r["loss"] for r in logs if "loss" in r}
    evals = {r["step"]: r["eval_loss"] for r in logs if "eval_loss" in r}
    ours = {r["step"]: r for r in map(json.loads, LOG.read_text().splitlines())}
    print("  단계  train(직접)  train(TRL)  val(직접)  val(TRL)")
    for s in sorted(evals):
        print(f"  {s:>4}  {ours[s]['train_loss']:>10.3f}  {train[s]:>10.3f}  {ours[s]['val_loss']:>9.3f}  {evals[s]:>8.3f}")
    v = validate(model, va)
    model.eval()
    with torch.no_grad():
        per = [model(input_ids=torch.tensor([e["input_ids"]]), labels=torch.tensor([e["labels"]])).loss.item()
               for e in va.items]
    print(f"  학습 후 같은 모델: 토큰 가중 평균 {v['val_loss']:.3f}(제46장 validate), "
          f"샘플 평균 {sum(per) / len(per):.3f}, TRL eval_loss {evals[30]:.3f}")
    OUT.write_text(json.dumps({"train": train, "eval": evals, "validate_token_weighted": v["val_loss"]},
                              indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
