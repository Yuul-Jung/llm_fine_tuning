# 제33장 핵심 실습: INT8 양자화를 직접 구현해 Qwen 가중치를 8비트로 저장·복원하고 오차를 비교한다
#
# 실행: python examples/ch33/int8_quant.py
#
# 대칭:   q = round(w ÷ s),             s = max|w| ÷ 127,          복원 ŵ = q × s
# 비대칭: q = round(w ÷ s) + z,         s = (max − min) ÷ 255,     복원 ŵ = (q − z) × s

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from common.qwen import load_model  # noqa: E402

NAME = "model.layers.0.self_attn.q_proj.weight"


def quant_symmetric(w: torch.Tensor, per_channel: bool = False, clip: float | None = None,
                    rounding=torch.round) -> tuple:
    """INT8 대칭 양자화. per_channel이면 출력 행마다 Scale을 따로 둔다. (코드 int8, Scale) 반환."""
    amax = w.abs().amax(dim=1, keepdim=True) if per_channel else w.abs().max()
    if clip is not None:                                       # 범위를 일부러 좁힌다(큰 값은 포화)
        amax = torch.quantile(w.abs().float().flatten(), clip) if not per_channel else amax
    scale = (amax / 127).clamp(min=1e-12)
    q = rounding(w / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequant_symmetric(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale


def quant_asymmetric(w: torch.Tensor) -> tuple:
    """UINT8 비대칭 양자화(Per-Tensor). (코드 uint8, Scale, Zero Point) 반환."""
    lo, hi = w.min(), w.max()
    scale = (hi - lo) / 255
    zero = torch.round(-lo / scale).clamp(0, 255)
    q = (torch.round(w / scale) + zero).clamp(0, 255).to(torch.uint8)
    return q, scale, zero


def dequant_asymmetric(q, scale, zero) -> torch.Tensor:
    return (q.float() - zero) * scale


def rel_err(a, b) -> float:
    return ((a - b).norm() / b.norm()).item()


def main() -> None:
    print("── 1. 손계산: w = [−0.5, 0.0, 0.25, 1.0] ──")
    w = torch.tensor([-0.5, 0.0, 0.25, 1.0])
    q, s = quant_symmetric(w)
    print(f"  대칭  s = 1.0 ÷ 127 = {s.item():.6f}, q = {q.tolist()}, 복원 {[round(v, 4) for v in dequant_symmetric(q, s).tolist()]}")
    qa, sa, za = quant_asymmetric(w)
    print(f"  비대칭 s = 1.5 ÷ 255 = {sa.item():.6f}, z = {int(za)}, q = {qa.tolist()}, "
          f"복원 {[round(v, 4) for v in dequant_asymmetric(qa, sa, za).tolist()]}")

    w = dict(load_model().named_parameters())[NAME].detach()
    print(f"\n── 2. {NAME} {tuple(w.shape)}의 값 분포 ──")
    a = w.abs().flatten()
    print(f"  |w| 평균 {a.mean():.4f}, 99% 지점 {torch.quantile(a, 0.99):.4f}, 99.99% 지점 "
          f"{torch.quantile(a, 0.9999):.4f}, 최댓값 {a.max():.4f}")
    row_max = w.abs().amax(dim=1)
    print(f"  행(출력 채널)별 최댓값: 최소 {row_max.min():.4f}, 중앙 {row_max.median():.4f}, 최대 {row_max.max():.4f}")

    print("\n── 3. Per-Tensor vs Per-Channel(대칭 INT8) ──")
    for label, pc in (("Per-Tensor", False), ("Per-Channel", True)):
        q, s = quant_symmetric(w, per_channel=pc)
        back = dequant_symmetric(q, s)
        worst = ((back - w).abs() / s.expand_as(w) if pc else (back - w).abs() / s).max().item()
        print(f"  {label:<12}Scale {s.numel():>3}개, 상대 오차 {rel_err(back, w):.2e}, "
              f"최대 오차 ÷ Scale = {worst:.3f}(≤ 0.5)")
    qa, sa, za = quant_asymmetric(w)
    print(f"  비대칭(Per-Tensor) 상대 오차 {rel_err(dequant_asymmetric(qa, sa, za), w):.2e}")

    print("\n── 4. Clipping과 Rounding(Per-Tensor) ──")
    for label, clip in (("최댓값 기준", None), ("99.99% 지점에서 자름", 0.9999), ("99% 지점에서 자름", 0.99)):
        q, s = quant_symmetric(w, clip=clip)
        print(f"  {label:<18}Scale {s.item():.2e}, 상대 오차 {rel_err(dequant_symmetric(q, s), w):.2e}")
    q, s = quant_symmetric(w, rounding=torch.floor)
    print(f"  반올림 대신 내림       상대 오차 {rel_err(dequant_symmetric(q, s), w):.2e}, "
          f"평균 치우침 {(dequant_symmetric(q, s) - w).mean():+.2e}")

    print("\n── 5. 저장 용량 실측(safetensors) ──")
    q, s = quant_symmetric(w, per_channel=True)
    with tempfile.TemporaryDirectory() as d:
        save_file({"w": w.contiguous()}, f"{d}/fp32.safetensors")
        save_file({"q": q.contiguous(), "scale": s.contiguous()}, f"{d}/int8.safetensors")
        a32, a8 = (Path(d) / "fp32.safetensors").stat().st_size, (Path(d) / "int8.safetensors").stat().st_size
    print(f"  FP32 {a32:,}바이트 → INT8 + Scale {a8:,}바이트 ({a8 / a32:.1%})")


if __name__ == "__main__":
    main()
