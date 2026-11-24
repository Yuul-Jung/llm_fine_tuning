# 제18장 실습 1: LayerNorm과 RMSNorm 직접 구현, 잔차 스트림의 크기 관찰
#
# 실행: python examples/ch18/norm_residual.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """평균을 빼고 표준편차로 나눈 뒤 크기(weight)와 이동(bias)을 학습한다."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * weight + bias


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """평균은 빼지 않고, 제곱평균제곱근(RMS)으로만 나눈 뒤 크기(weight)를 곱한다."""
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8) * 50 + 3            # 크고 치우친 값

    print("── 1. 정규화 전후의 통계(마지막 축 기준) ──")
    ones, zeros = torch.ones(8), torch.zeros(8)
    ln, rms = layer_norm(x, ones, zeros), rms_norm(x, ones)
    for name, t in (("입력", x), ("LayerNorm", ln), ("RMSNorm", rms)):
        r = t.pow(2).mean(dim=-1).sqrt()
        print(f"  {name:<10} 평균 {t.mean(dim=-1)[0, 0].item():+8.3f}   RMS {r[0, 0].item():8.3f}")

    print("\n── 2. PyTorch 함수와 비교 ──")
    w = torch.randn(8)
    e1 = (layer_norm(x, w, zeros) - F.layer_norm(x, (8,), w, zeros)).abs().max().item()
    e2 = (rms_norm(x, w) - F.rms_norm(x, (8,), w, eps=1e-6)).abs().max().item()
    print(f"  LayerNorm 최대 차이 {e1:.1e},  RMSNorm 최대 차이 {e2:.1e}")

    print("\n── 3. 크기가 달라도 같은 결과(RMSNorm은 배율에 무관) ──")
    print(f"  rms_norm(x) = rms_norm(x × 100): {torch.allclose(rms_norm(x, w), rms_norm(x * 100, w), atol=1e-5)}")

    print("\n── 4. 실제 Qwen2.5의 잔차 스트림 크기(토큰 평균 L2 길이) ──")
    from common.qwen import load_model, load_tokenizer
    model, tok = load_model(), load_tokenizer()
    with torch.no_grad():
        hs = model(tok("언어 모델은 다음 토큰을 예측한다.", return_tensors="pt").input_ids,
                   output_hidden_states=True).hidden_states
    for i in (0, 1, 3, 12, 20):
        name = "Embedding 출력" if i == 0 else f"{i}번째 계층 통과 후"
        print(f"  {name:<16} {hs[i].norm(dim=-1).mean().item():8.1f}")
    normed = rms_norm(hs[12], torch.ones(896))          # 가중치를 곱하기 전: RMS = 1
    print(f"  12번째 계층 통과 후 값을 RMSNorm하면: 길이 {normed.norm(dim=-1).mean().item():.1f}"
          f" (= √896 = {896 ** 0.5:.1f})")


if __name__ == "__main__":
    main()
