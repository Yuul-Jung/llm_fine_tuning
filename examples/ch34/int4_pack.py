# 제34장 핵심 실습: INT4 양자화와 Bit Packing — 4비트 두 개를 한 바이트에 담아 파일로 저장한다
#
# 실행: python examples/ch34/int4_pack.py
#
# 바이트 하나 = [상위 4비트: 홀수 번째 값][하위 4비트: 짝수 번째 값]

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "ch33"))

import torch  # noqa: E402

from common.qwen import load_model  # noqa: E402
from int8_quant import NAME, dequant_symmetric, quant_symmetric  # noqa: E402  (제33장)


def quant_int4(w: torch.Tensor, group: int | None = None) -> tuple:
    """INT4 대칭 양자화(−7 ~ 7). group이 없으면 행마다, 있으면 행 안의 group개씩 Scale 하나."""
    rows = w.reshape(-1, group) if group else w
    scale = (rows.abs().amax(dim=1, keepdim=True) / 7).clamp(min=1e-12)
    q = torch.round(rows / scale).clamp(-7, 7).to(torch.int8)       # 담는 그릇은 int8, 값은 4비트
    return q.reshape(w.shape), scale


def dequant_int4(q: torch.Tensor, scale: torch.Tensor, group: int | None = None) -> torch.Tensor:
    rows = q.reshape(-1, group) if group else q
    return (rows.float() * scale).reshape(q.shape)


def pack(q: torch.Tensor) -> torch.Tensor:
    """−8 ~ 7 정수 두 개를 바이트 하나로: 하위 4비트 = 짝수 번째, 상위 4비트 = 홀수 번째"""
    flat = q.flatten().to(torch.int16)
    assert flat.numel() % 2 == 0 and flat.min() >= -8 and flat.max() <= 7
    low, high = flat[0::2] & 0xF, flat[1::2] & 0xF          # & 0xF: 2의 보수 하위 4비트만 남김
    return (low | (high << 4)).to(torch.uint8)


def sign_extend4(v: torch.Tensor) -> torch.Tensor:
    """4비트 2의 보수(0 ~ 15)를 −8 ~ 7로: 맨 위 비트(8)가 켜져 있으면 16을 뺀다"""
    return torch.where(v >= 8, v - 16, v)


def unpack(packed: torch.Tensor, shape) -> torch.Tensor:
    b = packed.to(torch.int16)
    low, high = sign_extend4(b & 0xF), sign_extend4((b >> 4) & 0xF)
    return torch.stack([low, high], dim=1).flatten().to(torch.int8).reshape(shape)


def rel_err(a, b) -> float:
    return ((a - b).norm() / b.norm()).item()


def main() -> None:
    print("── 1. 손계산: [3, −2] → 한 바이트 ──")
    p = pack(torch.tensor([3, -2]))
    print(f"  3 = 0011, −2 = 1110(2의 보수) → 상위 1110 | 하위 0011 = {int(p):08b} = 0x{int(p):02X} = {int(p)}")
    print(f"  되돌리기: 하위 {int(p) & 0xF} → {int(sign_extend4(torch.tensor(int(p) & 0xF)))}, "
          f"상위 {int(p) >> 4} → {int(sign_extend4(torch.tensor(int(p) >> 4)))}")
    every = torch.arange(-8, 8)
    print(f"  −8 ~ 7 전체 왕복 일치: {torch.equal(unpack(pack(every), every.shape), every.to(torch.int8))}")

    w = dict(load_model().named_parameters())[NAME].detach()
    print(f"\n── 2. {NAME} {tuple(w.shape)}: INT8 vs INT4 상대 오차 ──")
    q8, s8 = quant_symmetric(w, per_channel=True)
    print(f"  INT8 Per-Channel        {rel_err(dequant_symmetric(q8, s8), w):.3f}  (값 255가지)")
    for label, g in (("INT4 Per-Channel", None), ("INT4 Group 128", 128), ("INT4 Group 64", 64), ("INT4 Group 32", 32)):
        q4, s4 = quant_int4(w, g)
        extra = s4.numel() * 4 / w.numel() * 8                   # Scale(FP32)이 가중치 하나당 더하는 비트
        print(f"  {label:<24}{rel_err(dequant_int4(q4, s4, g), w):.3f}  (Scale {s4.numel():>6,}개, "
              f"가중치당 +{extra:.2f}비트)")

    print("\n── 3. 패킹과 파일 크기 ──")
    q4, s4 = quant_int4(w, 64)
    packed = pack(q4)
    same = torch.equal(unpack(packed, q4.shape), q4)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "w_int8.bin").write_bytes(q8.numpy().tobytes())
        (Path(d) / "w_int4.bin").write_bytes(packed.numpy().tobytes())
        (Path(d) / "w_int4.hex").write_text(
            "\n".join(packed[i:i + 4].flip(0).numpy().tobytes().hex() for i in range(0, packed.numel(), 4)) + "\n")
        sizes = {f.name: f.stat().st_size for f in sorted(Path(d).iterdir())}
        head = (Path(d) / "w_int4.hex").read_text().splitlines()[:2]
    print(f"  패킹 → 언패킹 왕복 완전 일치: {same}")
    print(f"  파일 크기 {sizes}")
    print(f"  INT4 / INT8 = {sizes['w_int4.bin'] / sizes['w_int8.bin']:.3f}")
    print(f"  w_int4.hex 앞 두 줄(32비트 워드, 가중치 8개씩): {head}")
    print(f"  워드 하나의 가중치 8개: {q4.flatten()[:8].tolist()}")


if __name__ == "__main__":
    main()
