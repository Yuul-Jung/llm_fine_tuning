# 제32장 핵심 실습: FP32·FP16·BF16을 비트로 분해하고, 같은 가중치를 세 자료형으로 저장해 비교한다
#
# 실행: python examples/ch32/float_bits.py

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.device import select_device  # noqa: E402

# (부호, 지수, 가수) 비트 수와 지수 치우침(bias)
LAYOUT = {torch.float32: (1, 8, 23, 127), torch.float16: (1, 5, 10, 15), torch.bfloat16: (1, 8, 7, 127)}
INT_VIEW = {torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16}


def bits_of(x: float, dtype) -> str:
    _, e, m, _ = LAYOUT[dtype]
    raw = torch.tensor([x], dtype=dtype).view(INT_VIEW[dtype]).item()
    total = 1 + e + m
    return format(raw & ((1 << total) - 1), f"0{total}b")


def decode(pattern: str, dtype) -> float:
    """비트열 → (−1)^부호 × 2^(지수 − bias) × (1 + 가수 ÷ 2^가수비트). 정규수만 다룬다."""
    _, e, m, bias = LAYOUT[dtype]
    s, ex, man = int(pattern[0]), int(pattern[1:1 + e], 2), int(pattern[1 + e:], 2)
    return (-1) ** s * 2.0 ** (ex - bias) * (1 + man / 2 ** m)


def show(x: float, dtype) -> str:
    _, e, m, _ = LAYOUT[dtype]
    p = bits_of(x, dtype)
    return f"{p[0]} {p[1:1 + e]} {p[1 + e:]}  → {decode(p, dtype):.10g}"


def matmul_speed(device: str, dtype, n: int = 512, reps: int = 10) -> float:
    a = torch.randn(n, n, device=device).to(dtype)
    b = torch.randn(n, n, device=device).to(dtype)
    a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    return 2 * n ** 3 * reps / (time.perf_counter() - t0) / 1e9


def main() -> None:
    names = {torch.float32: "FP32", torch.float16: "FP16", torch.bfloat16: "BF16"}

    print("── 1. 비트 분해: 부호 | 지수 | 가수 → 다시 계산한 값 ──")
    for x in (0.1, -2.5):
        print(f"  {x}")
        for dt, nm in names.items():
            print(f"    {nm}  {show(x, dt)}")

    print("\n── 2. 범위와 정밀도(torch.finfo) ──")
    for dt, nm in names.items():
        f = torch.finfo(dt)
        print(f"  {nm}: 최댓값 {f.max:.3e}, 가장 작은 정규수 {f.tiny:.3e}, 1 다음 간격 {f.eps:.3e}")

    print("\n── 3. 범위 밖의 값 ──")
    for x in (70000.0, 1e-8):
        row = ", ".join(f"{nm} {torch.tensor(x, dtype=dt).float().item():.4g}" for dt, nm in names.items())
        print(f"  {x:g} → {row}")

    print("\n── 4. 같은 가중치를 세 자료형으로 저장(Qwen2-Tiny 크기 행렬, N(0, 0.02)) ──")
    torch.manual_seed(0)
    w = torch.randn(256, 768) * 0.02
    for dt, nm in names.items():
        back = w.to(dt).float()
        rel = ((back - w).norm() / w.norm()).item()
        print(f"  {nm}: {w.numel() * torch.finfo(dt).bits // 8 / 1024:.0f}KiB, 상대 오차 {rel:.2e}")

    print("\n── 5. 저장은 16비트, 계산은 FP32 ──")
    x = torch.randn(4, 768)
    exact = x @ w.T
    for dt, nm in ((torch.float16, "FP16"), (torch.bfloat16, "BF16")):
        stored = w.to(dt)
        mixed = x @ stored.float().T                    # 꺼낼 때 FP32로 올려 계산
        print(f"  {nm} 저장 → FP32 계산: 결과 상대 오차 {((mixed - exact).norm() / exact.norm()).item():.2e}")

    print("\n── 6. FP16 Gradient의 언더플로와 Loss Scaling ──")
    g = torch.tensor([2e-8, 1e-6, 1e-3])
    print(f"  FP16 그대로      {g.half().float().tolist()}")
    scaled = (g * 65536).half()                          # 손실에 2^16을 곱한 효과
    print(f"  ×65536 후 FP16  {(scaled.float() / 65536).tolist()}  (되돌릴 때 FP32에서 나눔)")

    device, _ = select_device()
    print(f"\n── 7. 행렬곱 속도(512×512, {device}, 환경마다 다름) ──")
    for dt, nm in names.items():
        try:
            print(f"  {nm}: {matmul_speed(device, dt):8.1f} GFLOPS")
        except RuntimeError as e:
            print(f"  {nm}: 실행 불가({str(e)[:40]})")


if __name__ == "__main__":
    main()
