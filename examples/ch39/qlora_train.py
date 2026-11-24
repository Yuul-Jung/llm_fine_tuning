# 제39장 핵심 실습: Qwen2.5-0.5B-Instruct QLoRA 최소 학습 — 학습 전후 PPL, 동결 확인, Adapter 저장
#
# 실행: python examples/ch39/qlora_train.py
#
# 설정(이 교재의 GT 1030 기준): NF4 + FP32 계산, 시퀀스 256, Batch 1, Accumulation 4, Checkpointing
# GPU가 없으면 bitsandbytes CPU 경로로 같은 QLoRA를 실행한다(교재 검증 환경).

import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "ch22", HERE.parent / "ch24", HERE.parent / "ch25",
          HERE.parent / "ch37", HERE.parent / "ch38"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

from common.fixed_data import fingerprint, load_blocks  # noqa: E402
from common.qwen import load_tokenizer  # noqa: E402
from load_4bit import hardware  # noqa: E402             (제37장)
from qlora_memory import build_qlora  # noqa: E402       (제38장)
from trainer import eval_loss, train  # noqa: E402       (제25장)

ADAPTER = HERE.parents[1] / "models" / "qwen2.5-0.5b-qlora-fixed"
SEQ, STEPS, LR = 256, 30, 5e-4


def keep_training_layout(model) -> int:
    """bitsandbytes 0.50 CPU 경로는 AVX512-BF16 CPU에서 평가 모드로 계산할 때 4비트 가중치를 추론 전용
    배치로 바꾼다. 바뀐 뒤에는 학습이 실패하므로, 학습할 모델에서는 이 변환을 끈다(GPU 경로와는 무관)."""
    n = 0
    for m in model.modules():
        if hasattr(m, "support_avx512bf16_for_cpu"):
            m.support_avx512bf16_for_cpu = False
            n += 1
    return n


def frozen_4bit(model) -> dict:
    """4비트 가중치(번호, uint8)와 absmax를 복사해 둔다."""
    out = {}
    for name, p in model.named_parameters():
        if p.dtype == torch.uint8:
            out[name] = p.detach().clone()
            out[name + ".absmax"] = p.quant_state.absmax.detach().clone()
    return out


def main() -> None:
    hw = hardware()
    device = hw["device"]
    tok = load_tokenizer()
    train_blocks = load_blocks(tok, "train", seq_len=SEQ)
    eval_blocks = load_blocks(tok, "eval")                    # 평가는 제22장 기준(128토큰 블록)
    print(f"장치 {hw['name']}, 학습 블록 {tuple(train_blocks.shape)}, 평가 블록 {tuple(eval_blocks.shape)} "
          f"(지문 {fingerprint(eval_blocks)})\n")

    print("── 1. 학습 준비: 4비트 로딩 → prepare_model_for_kbit_training → LoRA ──")
    torch.manual_seed(0)
    model, parts = build_qlora(device, checkpointing=True)
    if device == "cpu":
        print(f"  CPU 추론용 가중치 재배치 끔: Linear4bit {keep_training_layout(model)}개")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  학습 파라미터 {trainable:,}개, 원본 메모리 {sum(parts.values()) / 2**20:.1f}MiB")
    before = frozen_4bit(model)
    ppl0 = math.exp(eval_loss(model, eval_blocks, device))
    print(f"  학습 전 PPL {ppl0:.2f}")

    print(f"\n── 2. 학습: Batch 1 × Accumulation 4, {STEPS}단계, lr {LR:.0e}, Warmup 3 ──")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    hist = train(model, train_blocks, {"평가": eval_blocks}, steps=STEPS, batch=1, accum=4, lr=LR,
                 warmup=3, device=device, log_every=10)
    for r in hist:
        print(f"  단계 {r['step']:>3}  학습 {r['train']:.3f}  Grad노름 {r['grad_norm']:.2f}  "
              f"평가 {r['평가']:.3f}(PPL {math.exp(r['평가']):.2f})")
    print(f"  걸린 시간 {time.perf_counter() - t0:.0f}초(환경마다 다름)")
    if device == "cuda":
        print(f"  GPU 최대 할당 {torch.cuda.max_memory_allocated() / 2**20:.0f}MiB")

    print("\n── 3. 동결 검증과 Adapter 저장 ──")
    after = frozen_4bit(model)
    same = all(torch.equal(before[k], after[k]) for k in before)
    print(f"  4비트 원본(번호 {sum(1 for k in before if not k.endswith('absmax'))}개 텐서 + absmax) 그대로: {same}")
    model.save_pretrained(ADAPTER)
    files = {f.name: f.stat().st_size for f in sorted(ADAPTER.iterdir()) if f.suffix in (".json", ".safetensors")}
    print(f"  저장 {files}")
    print(f"  PPL {ppl0:.2f} → {math.exp(hist[-1]['평가']):.2f}")


if __name__ == "__main__":
    main()
