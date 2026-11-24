# 제27장 핵심 실습: 실제 Qwen 가중치를 Rank별로 근사해 복원 오차와 파라미터 수를 비교한다
#
# 실행: python examples/ch27/qwen_svd.py
#
# 대상: 0번 계층 q_proj [896, 896]. 비교: 같은 크기의 무작위 행렬,
#       그리고 Base → Instruct 사후 학습에서 생긴 변화량 ΔW = W_instruct − W_base(제21장)

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch21"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

from base_vs_instruct import load_base  # noqa: E402   (제21장)
from common.qwen import load_model  # noqa: E402
from rank_basics import low_rank, rel_error  # noqa: E402

NAME = "model.layers.0.self_attn.q_proj.weight"
RANKS = (1, 4, 16, 64, 128, 256, 512, 896)


def error_curve(w: torch.Tensor) -> list:
    return [rel_error(w, b @ a) for b, a in (low_rank(w, r) for r in RANKS)]


def main() -> None:
    inst, base = dict(load_model().named_parameters()), dict(load_base().named_parameters())
    w_inst, w_base = inst[NAME].detach(), base[NAME].detach()
    delta = w_inst - w_base
    torch.manual_seed(0)
    rand = torch.randn_like(w_inst) * w_inst.std()
    out_dim, in_dim = w_inst.shape

    print(f"── 1. {NAME} {tuple(w_inst.shape)}, 원소 {w_inst.numel():,}개 ──")
    print(f"  Rank(수치) {torch.linalg.matrix_rank(w_inst).item()}, "
          f"‖ΔW‖ / ‖W_base‖ = {(delta.norm() / w_base.norm()).item():.3f}")

    print("\n── 2. Rank r 근사: 파라미터 수와 상대 오차 ──")
    curves = {"Instruct W": error_curve(w_inst), "무작위": error_curve(rand), "ΔW": error_curve(delta)}
    print(f"  {'r':>4}{'파라미터 r(out+in)':>19}{'비율':>8}" + "".join(f"{k:>12}" for k in curves))
    for i, r in enumerate(RANKS):
        n = r * (out_dim + in_dim)
        print(f"  {r:>4}{n:>19,}{n / w_inst.numel():>8.1%}" + "".join(f"{c[i]:>12.3f}" for c in curves.values()))

    print("\n── 3. 특잇값 에너지가 상위 10%(90개)에 모인 비율 ──")
    for label, m in (("Instruct W", w_inst), ("무작위", rand), ("ΔW", delta)):
        s2 = torch.linalg.svdvals(m) ** 2
        print(f"  {label:<11}{(s2[:90].sum() / s2.sum()).item():.1%}")

    print("\n── 4. 다른 행렬의 ΔW: Rank 64 근사의 상대 오차 ──")
    for name in ("model.layers.0.self_attn.v_proj.weight", "model.layers.12.self_attn.q_proj.weight",
                 "model.layers.0.mlp.down_proj.weight"):
        d = (inst[name] - base[name]).detach()
        b, a = low_rank(d, 64)
        print(f"  {name:<42}{tuple(d.shape)!s:<13}{rel_error(d, b @ a):.3f}")


if __name__ == "__main__":
    main()
