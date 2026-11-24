# 제02장 핵심 실습 2: 모델 메모리 계산과 가용 메모리 측정
#
# 실행: python examples/ch02/memory_budget.py
# 내용: Qwen2.5-0.5B 가중치·Logits 메모리를 계산하고, 실제 장치에 텐서를 올려 계산값과 비교한다.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from common.device import select_device  # noqa: E402

MIB = 2 ** 20
GIB = 2 ** 30

# Qwen2.5-0.5B-Instruct 공식 config.json 값
QWEN05 = {
    "hidden": 896,
    "intermediate": 4864,
    "layers": 24,
    "heads": 14,
    "kv_heads": 2,
    "vocab": 151936,
}

# 자료형별 원소 하나의 크기(바이트). NF4는 4비트 = 0.5바이트(Scale 저장 비용 제외)
BYTES_PER_PARAM = {"FP32": 4, "FP16/BF16": 2, "INT8": 1, "NF4": 0.5}


def count_qwen_params(c: dict) -> int:
    """config 값만으로 전체 파라미터 수를 계산한다(Embedding·LM Head 공유는 한 번만)."""
    h, i = c["hidden"], c["intermediate"]
    kv = c["kv_heads"] * (h // c["heads"])                 # K·V 출력 차원 = 2 × 64 = 128
    attention = (h * h + h) + 2 * (h * kv + kv) + h * h    # q(+bias), k·v(+bias), o
    mlp = 3 * h * i                                        # gate, up, down
    norms = 2 * h                                          # Attention 앞·FFN 앞 RMSNorm
    embedding = c["vocab"] * h
    return embedding + c["layers"] * (attention + mlp + norms) + h   # 마지막 + h: Final Norm


def logits_bytes(seq_len: int, vocab: int, batch: int = 1, bytes_per: int = 4) -> int:
    """Logits 텐서 [batch, seq, vocab]의 크기(바이트)."""
    return batch * seq_len * vocab * bytes_per


def read_meminfo_gib(field: str) -> float:
    """/proc/meminfo의 항목(예: MemAvailable)을 GiB로 읽는다."""
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024 / GIB
    return -1.0


def measure_allocation(device: str, mib: int = 256) -> tuple:
    """지정한 크기의 FP32 텐서를 만들고 (계산값, 실제 증가량)을 바이트로 돌려준다."""
    numel = mib * MIB // 4
    expected = numel * 4
    if device == "cuda":
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        x = torch.empty(numel, dtype=torch.float32, device="cuda")
        actual = torch.cuda.memory_allocated() - before
    else:
        x = torch.empty(numel, dtype=torch.float32)
        actual = x.element_size() * x.numel()
    del x
    return expected, actual


def main() -> None:
    params = count_qwen_params(QWEN05)
    print(f"Qwen2.5-0.5B 파라미터 수: {params:,}")
    print("\n[가중치 메모리]")
    for name, size in BYTES_PER_PARAM.items():
        print(f"  {name:<10}: {params * size / GIB:6.2f} GiB")

    print("\n[Logits 메모리, Batch 1, FP32]")
    for seq in (128, 256, 512, 1024):
        print(f"  시퀀스 {seq:>4}: {logits_bytes(seq, QWEN05['vocab']) / MIB:7.1f} MiB")

    device, _ = select_device(verbose=False)
    print(f"\n[실측: {device}]")
    if device == "cuda":
        free_b, total_b = torch.cuda.mem_get_info(0)
        print(f"  전체 VRAM          : {total_b / MIB:8.0f} MiB")
        print(f"  사용 가능 VRAM     : {free_b / MIB:8.0f} MiB")
        print(f"  이미 사용 중       : {(total_b - free_b) / MIB:8.0f} MiB (CUDA 컨텍스트·화면 출력 등)")
    else:
        print(f"  전체 RAM           : {read_meminfo_gib('MemTotal'):6.1f} GiB")
        print(f"  사용 가능 RAM      : {read_meminfo_gib('MemAvailable'):6.1f} GiB")

    expected, actual = measure_allocation(device)
    verdict = "일치" if expected == actual else "불일치"
    print(f"  256MiB 텐서 할당   : 계산 {expected / MIB:.0f} MiB, 실제 {actual / MIB:.0f} MiB → {verdict}")


if __name__ == "__main__":
    main()
