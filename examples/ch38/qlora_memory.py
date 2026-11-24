# 제38장 핵심 실습: QLoRA(4비트 원본 + LoRA)의 학습 메모리를 분해하고 Gradient Checkpointing 효과를 잰다
#
# 실행: python examples/ch38/qlora_memory.py
#
# Activation은 "역전파를 위해 저장되는 텐서"의 바이트를 saved_tensors_hooks로 직접 센다(장치와 무관).

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "ch22", HERE.parent / "ch37"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402

from common.fixed_data import load_blocks  # noqa: E402
from common.qwen import load_tokenizer  # noqa: E402
from load_4bit import hardware, load_nf4, memory_parts  # noqa: E402   (제37장)

MiB = 2 ** 20
SEQ = 256                                                   # GT 1030 QLoRA 기준 시퀀스(이 교재의 메모리 예산)


def build_qlora(device: str, checkpointing: bool) -> tuple:
    """(QLoRA 모델, LoRA를 붙이기 전 원본의 메모리 분해)"""
    model = load_nf4(device)
    parts = memory_parts(model)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=checkpointing,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0,
                     bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg), parts


def saved_bytes_of_step(model, ids) -> tuple:
    """순전파 동안 역전파용으로 새로 저장된 텐서의 바이트와 한 단계(순전파+역전파) 시간.

    파라미터와 같은 저장 공간을 쓰는 텐서(가중치 자체)는 빼고, 같은 저장 공간은 한 번만 센다.
    """
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen, total = set(), [0]

    def pack(t):
        s = t.untyped_storage()
        if s.data_ptr() not in params and s.data_ptr() not in seen:
            seen.add(s.data_ptr())
            total[0] += s.nbytes()
        return t

    model.train()                                            # Checkpointing은 학습 모드에서만 동작한다
    t0 = time.perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    return total[0], time.perf_counter() - t0, loss.item()


def main() -> None:
    hw = hardware()
    tok = load_tokenizer()
    blocks = load_blocks(tok, "train", seq_len=SEQ)
    ids = blocks[:1].to(hw["device"])
    print(f"장치 {hw['name']}, 입력 [1, {SEQ}]\n")

    results = {}
    for ckpt in (False, True):
        torch.manual_seed(0)
        model, parts = build_qlora(hw["device"], ckpt)
        saved, sec, loss = saved_bytes_of_step(model, ids)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results[ckpt] = (saved, sec, loss, trainable, parts)
        del model

    saved0, sec0, loss0, trainable, parts = results[False]
    saved1, sec1, loss1, _, _ = results[True]
    base = sum(parts.values())
    lora_state = trainable * (4 + 4 + 8)                     # 가중치 + Gradient + AdamW 상태(FP32)
    logits = SEQ * 151_936 * 4

    print("── 1. QLoRA 구성 ──")
    print(f"  4비트 원본(동결) + Embedding 등: {base / MiB:7.1f}MiB, 학습 파라미터(LoRA) {trainable:,}개")

    print("\n── 2. 학습 메모리 분해(Batch 1, 시퀀스 256, FP32) ──")
    rows = [("원본 가중치(4비트 + Embedding)", base), ("LoRA 가중치·Gradient·AdamW", lora_state),
            ("Logits 하나(256 × 151,936 × 4)", logits),
            ("Activation(저장 텐서), Checkpointing 끔", saved0),
            ("Activation(저장 텐서), Checkpointing 켬", saved1)]
    for label, v in rows:
        print(f"  {label:<38}{v / MiB:8.1f}MiB")

    print("\n── 3. Gradient Checkpointing 비교 ──")
    print(f"  저장 텐서 {saved0 / MiB:.0f}MiB → {saved1 / MiB:.0f}MiB ({saved1 / saved0:.0%})")
    print(f"  한 단계 시간 {sec0:.2f}초 → {sec1:.2f}초 ({sec1 / sec0:.2f}배, 환경마다 다름)")
    print(f"  손실 같음: {abs(loss0 - loss1) < 1e-4}  ({loss0:.4f} / {loss1:.4f})")


if __name__ == "__main__":
    main()
