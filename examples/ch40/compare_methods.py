# 제40장 핵심 실습: Full FT·LoRA·QLoRA를 같은 조건으로 비교한다
#
# 실행: python examples/ch40/compare_methods.py          (비교 A + 비교 B, 교재 검증 환경 CPU에서 약 15분)
#       python examples/ch40/compare_methods.py --only-a  (비교 A만, 약 4분)
#
# 비교 A: 사전학습 Qwen2-Tiny(제25장) + 자전거 정비 데이터, 60단계
# 비교 B: Qwen2.5-0.5B-Instruct + 고정 학습 데이터(시퀀스 256), 30단계 — QLoRA(제39장)와 FP32 LoRA

import json
import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "ch22", HERE.parent / "ch24", HERE.parent / "ch25", HERE.parent / "ch29",
          HERE.parent / "ch37", HERE.parent / "ch38", HERE.parent / "ch39"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from transformers.utils import logging  # noqa: E402

from common.fixed_data import fingerprint, load_blocks  # noqa: E402
from common.models import QWEN_ID, QWEN_REVISION  # noqa: E402
from common.qwen import load_tokenizer  # noqa: E402
from lora_linear import ALL_LINEAR  # noqa: E402                     (제29장)
from load_4bit import memory_parts, nf4_config  # noqa: E402        (제37장)
from qlora_memory import saved_bytes_of_step  # noqa: E402          (제38장)
from qlora_train import keep_training_layout  # noqa: E402          (제39장)
from trainer import eval_loss, train  # noqa: E402                  (제25장)

logging.disable_progress_bar()
TINY = HERE.parents[1] / "models" / "qwen2-tiny-pretrained"
MiB = 2 ** 20


def lora(model, targets=ALL_LINEAR):
    torch.manual_seed(0)                                         # A 초기값 고정
    return get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=list(targets),
                                            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))


def load(path_or_id, quant: bool, **kw):
    q = {"quantization_config": nf4_config(), "device_map": "cpu"} if quant else {}
    return AutoModelForCausalLM.from_pretrained(path_or_id, dtype=torch.float32, **q, **kw)


def weight_bytes(model) -> int:
    """파라미터 메모리(4비트는 번호 + absmax). LoRA 파라미터 포함."""
    return sum(memory_parts(model).values())


def artifact_bytes(model) -> int:
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        return sum(f.stat().st_size for f in Path(d).iterdir() if f.suffix == ".safetensors")


def run(label, model, data, ev, steps, lr, accum, batch, seq_probe) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    act, _, _ = saved_bytes_of_step(model, seq_probe)            # 한 단계 저장 텐서(제38장)
    model.zero_grad(set_to_none=True)
    train(model, data, {}, steps=steps, batch=batch, accum=accum, lr=lr, warmup=5, log_every=steps)
    res = {"방식": label, "학습 파라미터": trainable,
           "고정 메모리MiB": round((weight_bytes(model) + trainable * 12) / MiB, 1),  # + Gradient·AdamW
           "Activation MiB": round(act / MiB, 1), "저장 MiB": round(artifact_bytes(model) / MiB, 2)}
    for k, blocks in ev.items():
        res[k] = round(math.exp(eval_loss(model, blocks, "cpu")), 2)
    return res


def show(rows: list) -> None:
    keys = list(rows[0])
    for k in keys:
        print(f"  {k:<14}" + "".join(f"{str(r[k]):>16}" for r in rows))


def compare_a(tok) -> list:
    d = {s: load_blocks(tok, s) for s in ("domain_train", "domain_eval", "eval")}
    ev = {"도메인 PPL": d["domain_eval"], "일반 PPL": d["eval"]}
    probe = d["domain_train"][:1]
    print(f"── 비교 A: Qwen2-Tiny, 학습 {tuple(d['domain_train'].shape)}, 60단계, Batch 2 × Accumulation 2 ──")
    base = load(TINY, quant=False)
    print("  학습 전" + "".join(f"  {k} {math.exp(eval_loss(base, b, 'cpu')):.2f}" for k, b in ev.items()))
    rows = [run("Full FT", load(TINY, False), d["domain_train"], ev, 60, 3e-4, 2, 2, probe)]
    rows.append(run("LoRA", lora(load(TINY, False)), d["domain_train"], ev, 60, 2e-3, 2, 2, probe))
    q = prepare_model_for_kbit_training(load(TINY, True), use_gradient_checkpointing=False)
    q = lora(q)
    keep_training_layout(q)
    rows.append(run("QLoRA", q, d["domain_train"], ev, 60, 2e-3, 2, 2, probe))
    show(rows)
    return rows


def compare_b(tok) -> list:
    train_blocks = load_blocks(tok, "train", seq_len=256)
    ev = {"평가 PPL": load_blocks(tok, "eval")}
    probe = train_blocks[:1]
    print(f"\n── 비교 B: Qwen2.5-0.5B-Instruct, 학습 {tuple(train_blocks.shape)}, 30단계, "
          f"Batch 1 × Accumulation 4, Checkpointing, q·v LoRA ──")
    rows = []
    for label, quant in (("QLoRA(NF4)", True), ("LoRA(FP32)", False)):
        m = load(QWEN_ID, quant, revision=QWEN_REVISION)
        m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False}) if quant \
            else m
        if not quant:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            m.enable_input_require_grads()
        m = lora(m, ("q_proj", "v_proj"))
        if quant:
            keep_training_layout(m)
        rows.append(run(label, m, train_blocks, ev, 30, 5e-4, 4, 1, probe))
        del m
    show(rows)
    return rows


def main() -> None:
    tok = load_tokenizer()
    print("── 비교 조건 ──")
    print(f"  토크나이저 Qwen2.5(제08장), Label: 전체 토큰(첫 토큰 제외), 평가: 128토큰 블록 독립(제22장)")
    print(f"  데이터 지문 domain_train {fingerprint(load_blocks(tok, 'domain_train'))}, "
          f"eval {fingerprint(load_blocks(tok, 'eval'))}")
    print(f"  LoRA r 8, α 16, seed 0, AdamW, Warmup + Cosine, Clipping 1.0(제25장 train())\n")
    out = {"A": compare_a(tok)}
    if "--only-a" not in sys.argv:
        out["B"] = compare_b(tok)
    (HERE / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
