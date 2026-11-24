# 제11장 실습 1: 순차 처리(RNN)와 병렬 처리(Self-Attention)의 계산 순서 비교
#
# 실행: python examples/ch11/rnn_vs_attention.py

import torch


def rnn_forward(x: torch.Tensor, w_x: torch.Tensor, w_h: torch.Tensor) -> tuple:
    """h_t = tanh(x_t W_xᵀ + h_(t-1) W_hᵀ): 이전 결과가 있어야 다음을 계산할 수 있다."""
    seq, hidden = x.shape[0], w_h.shape[0]
    h = torch.zeros(hidden)
    outputs, steps = [], 0
    for t in range(seq):                       # 위치마다 한 번씩, 순서대로
        h = torch.tanh(x[t] @ w_x.T + h @ w_h.T)
        outputs.append(h)
        steps += 1
    return torch.stack(outputs), steps


def attention_forward(x: torch.Tensor) -> tuple:
    """모든 위치 쌍의 점수를 행렬곱 한 번으로 계산하고, 미래 위치는 가린다."""
    seq = x.shape[0]
    scores = x @ x.T / x.shape[1] ** 0.5            # [seq, seq]: 한 번에 계산
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    weights = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
    return weights @ x, 1                            # 행렬곱 단계: 위치 수와 무관


def main() -> None:
    torch.manual_seed(0)
    for seq in (4, 64, 1024):
        x = torch.randn(seq, 8)
        _, rnn_steps = rnn_forward(x, torch.randn(8, 8) * 0.3, torch.randn(8, 8) * 0.3)
        _, attn_steps = attention_forward(x)
        print(f"시퀀스 {seq:>5}: RNN 순차 단계 {rnn_steps:>5}번, Attention 순차 단계 {attn_steps}번")

    print("\n앞 위치의 결과는 뒤 토큰이 바뀌어도 변하지 않는가?")
    x = torch.randn(6, 8)
    y = x.clone()
    y[-1] = torch.randn(8)                              # 마지막 토큰만 교체
    w_x, w_h = torch.randn(8, 8) * 0.3, torch.randn(8, 8) * 0.3
    rnn_same = torch.allclose(rnn_forward(x, w_x, w_h)[0][:-1], rnn_forward(y, w_x, w_h)[0][:-1])
    att_same = torch.allclose(attention_forward(x)[0][:-1], attention_forward(y)[0][:-1])
    print(f"  RNN: {rnn_same},  Attention(미래 가림): {att_same}")


if __name__ == "__main__":
    main()
