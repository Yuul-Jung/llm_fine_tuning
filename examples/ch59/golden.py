# 제59장 핵심 실습(선택): k_proj 타일을 INT8로 만들고, RTL MatMul의 정수 기준값(Golden)과 HEX 파일을 만든다
#
# 실행: python examples/ch59/golden.py   (제52장의 병합 모델 필요, CPU 약 20초)
#
# 만드는 파일: examples/ch59/rtl/w.hex, x.hex, expected.hex (Testbench가 $readmemh로 읽는다)
# RTL 시뮬레이션과 Basys3 합성은 교재 검증 환경에 도구가 없어 실행하지 않았다(미검증).

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

MERGED = ROOT / "models" / "fpga_merged_v1"
RTL = HERE / "rtl"
ROWS, COLS = 8, 896                    # 타일: k_proj 출력 8개 × 입력 896개
BRAM_KIB = 50 * 36 / 8                 # XC7A35T: 36Kb 블록 50개 = 225KiB


def quant_rows(w: np.ndarray) -> tuple:
    """행(출력)마다 대칭 INT8: q = round(w / s), s = max|w| / 127 (제33장)"""
    s = np.abs(w).max(axis=1, keepdims=True) / 127
    return np.clip(np.round(w / s), -127, 127).astype(np.int8), s.astype(np.float32)


def quant_vec(x: np.ndarray) -> tuple:
    s = np.abs(x).max() / 127
    return np.clip(np.round(x / s), -127, 127).astype(np.int8), np.float32(s)


def acc_bits(n: int, a_max: int = 128, b_max: int = 128) -> int:
    """부호 있는 누산기가 넘치지 않는 최소 비트: |합| ≤ n × a_max × b_max"""
    worst = n * a_max * b_max
    return int(np.ceil(np.log2(worst + 1))) + 1


def to_hex(values, bits: int) -> list:
    mask = (1 << bits) - 1
    return [f"{int(v) & mask:0{bits // 4}x}" for v in values]          # 2의 보수(제31장)


def main() -> None:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MERGED, dtype=torch.float32).eval()
    layer = model.get_submodule("model.layers.0.self_attn.k_proj")      # 제53장과 같은 층

    print("── 1. Basys3 자원과 탑재 범위(INT8) ──")
    for name in ("q_proj", "k_proj"):
        w = model.get_submodule(f"model.layers.0.self_attn.{name}").weight
        kib = w.numel() / 1024
        print(f"  {name} {tuple(w.shape)}: {kib:.0f}KiB  BRAM {BRAM_KIB:.0f}KiB 안에 {'가능' if kib <= BRAM_KIB else '불가'}")

    print("\n── 2. 정수 MAC와 누산기 ──")
    print(f"  INT8 × INT8 곱 896개 합의 최악값 896 × 128 × 128 = {896 * 128 * 128:,} → 최소 {acc_bits(896)}비트"
          f"(부호 포함), 설계는 32비트")

    print(f"\n── 3. 타일 {ROWS} × {COLS}의 정수 기준값 ──")
    torch.manual_seed(0)
    x = torch.randn(COLS)
    wq, ws = quant_rows(layer.weight[:ROWS].detach().numpy())
    xq, xs = quant_vec(x.numpy())
    acc = wq.astype(np.int64) @ xq.astype(np.int64)                     # 정수 기준값(넘침 없는 64비트)
    assert np.abs(acc).max() < 2 ** 31
    y_int = acc * ws[:, 0] * xs                                          # 역양자화
    y_fp = (layer.weight[:ROWS] @ x).detach().numpy()                    # 편향 제외 W x
    rel = np.abs(y_int - y_fp).max() / np.abs(y_fp).max()
    print(f"  누산 결과 범위 {acc.min():,} ~ {acc.max():,}(32비트 안)")
    print(f"  INT8 결과(역양자화)와 FP32 W x의 최대 상대 오차 {rel:.2%}")

    RTL.mkdir(exist_ok=True)
    (RTL / "w.hex").write_text("\n".join(to_hex(wq.ravel(), 8)) + "\n")     # 행 우선, 한 줄에 8비트
    (RTL / "x.hex").write_text("\n".join(to_hex(xq, 8)) + "\n")
    (RTL / "expected.hex").write_text("\n".join(to_hex(acc, 32)) + "\n")
    print(f"  파일: w.hex {ROWS * COLS}줄, x.hex {COLS}줄, expected.hex {ROWS}줄(32비트 2의 보수)")
    print(f"  첫 기대값 {acc[0]:,} = {to_hex(acc[:1], 32)[0]}")


if __name__ == "__main__":
    main()
