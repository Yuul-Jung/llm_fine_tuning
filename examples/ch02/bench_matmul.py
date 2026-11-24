# 제02장 핵심 실습 1: 행렬곱 크기에 따른 CPU·GPU 연산 속도 측정
#
# 실행: python examples/ch02/bench_matmul.py
# 결과: 화면 측정표 + examples/ch02/bench_matmul.csv (+ matplotlib이 있으면 bench_matmul.png)

import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from common.device import select_device  # noqa: E402

SIZES = [256, 512, 1024, 2048]   # 정사각 행렬의 한 변 크기 n
REPEAT = 5                        # 같은 측정을 반복해 중앙값을 사용


def matmul_flops(n: int) -> int:
    """n×n 행렬곱의 부동소수점 연산 수: 곱셈 n³ + 덧셈 n³ = 2n³."""
    return 2 * n ** 3


def synchronize(device: str) -> None:
    """GPU는 명령을 예약만 하고 바로 돌아오므로, 실제 완료까지 기다린다."""
    if device == "cuda":
        torch.cuda.synchronize()


def time_matmul(n: int, device: str, dtype: torch.dtype, repeat: int = REPEAT) -> float:
    """행렬곱 한 번에 걸린 시간(초)의 중앙값을 돌려준다."""
    torch.manual_seed(0)
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)

    a @ b                 # 워밍업: 첫 실행의 준비 시간을 측정에서 제외
    synchronize(device)

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        a @ b
        synchronize(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def relative_error(n: int, device: str, dtype: torch.dtype) -> float:
    """장치·자료형의 결과를 CPU FP64(기준값)와 비교한 최대 상대오차."""
    torch.manual_seed(0)
    a = torch.randn(n, n, dtype=torch.float64)
    b = torch.randn(n, n, dtype=torch.float64)
    reference = a @ b
    actual = (a.to(device, dtype) @ b.to(device, dtype)).to("cpu", torch.float64)
    return ((actual - reference).abs().max() / reference.abs().max()).item()


def run_configs() -> list:
    """측정할 (장치, 자료형) 조합: CPU FP32는 항상, GPU가 있으면 FP32·FP16 추가."""
    configs = [("cpu", torch.float32)]
    device, _ = select_device(verbose=False)
    if device == "cuda":
        configs += [("cuda", torch.float32), ("cuda", torch.float16)]
    return configs


def main() -> None:
    rows = []
    print(f"{'장치':<6}{'자료형':<10}{'n':>6}{'시간(ms)':>12}{'GFLOPS':>10}{'상대오차':>12}")
    for device, dtype in run_configs():
        name = str(dtype).replace("torch.", "")
        for n in SIZES:
            seconds = time_matmul(n, device, dtype)
            gflops = matmul_flops(n) / seconds / 1e9
            error = relative_error(n, device, dtype)
            rows.append([device, name, n, seconds * 1e3, gflops, error])
            print(f"{device:<6}{name:<10}{n:>6}{seconds * 1e3:>12.2f}{gflops:>10.1f}{error:>12.1e}")

    out = Path(__file__).with_name("bench_matmul.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["device", "dtype", "n", "ms", "gflops", "rel_error"])
        writer.writerows(rows)
    print(f"\n측정표 저장: {out.name}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib이 없어 그래프는 생략합니다.")
        return

    for device, name in sorted({(r[0], r[1]) for r in rows}):
        points = [(r[2], r[4]) for r in rows if r[0] == device and r[1] == name]
        plt.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                 label=f"{device} {name}")
    plt.xscale("log", base=2)
    plt.xlabel("matrix size n")
    plt.ylabel("GFLOPS")
    plt.ylim(bottom=0)   # 세로축을 0부터 시작해 차이를 과장하지 않는다
    plt.legend()
    plt.grid(True, alpha=0.3)
    png = Path(__file__).with_name("bench_matmul.png")
    plt.savefig(png, dpi=120, bbox_inches="tight")
    print(f"그래프 저장: {png.name}")


if __name__ == "__main__":
    main()
