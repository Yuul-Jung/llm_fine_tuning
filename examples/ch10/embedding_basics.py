# 제10장 실습 1: One-hot 곱셈과 Embedding Lookup이 같은 계산임을 확인
#
# 실행: python examples/ch10/embedding_basics.py

import torch
import torch.nn.functional as F
from torch import nn


def one_hot_matmul(ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """One-hot 벡터 [.., vocab]에 Embedding 표 [vocab, hidden]를 곱한다."""
    one_hot = F.one_hot(ids, num_classes=table.shape[0]).float()
    return one_hot @ table


def lookup(ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """표에서 ID 번째 행을 꺼낸다(곱셈 없이 Indexing만)."""
    return table[ids]


def cosine(u: torch.Tensor, v: torch.Tensor) -> float:
    """코사인 유사도 = 내적 ÷ (길이 × 길이)"""
    return (u @ v / (u.norm() * v.norm())).item()


def main() -> None:
    print("── 1. 작은 Embedding 표(어휘 5, hidden 3) ──")
    table = torch.tensor([[0.1, 0.2, 0.3],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.9, 0.1, 0.0],
                          [0.0, 0.0, 1.0]])
    ids = torch.tensor([3, 1, 4])
    print("One-hot(ID 3):", F.one_hot(torch.tensor(3), 5).tolist())
    print("One-hot 곱셈:\n", one_hot_matmul(ids, table))
    print("Lookup:\n", lookup(ids, table))

    print("\n── 2. nn.Embedding ──")
    torch.manual_seed(0)
    emb = nn.Embedding(num_embeddings=10, embedding_dim=4)
    batch = torch.tensor([[1, 5, 7], [2, 2, 9]])           # [batch 2, seq 3]
    out = emb(batch)
    print(f"weight {tuple(emb.weight.shape)},  입력 {tuple(batch.shape)} → 출력 {tuple(out.shape)}")
    print(f"같은 ID(2)는 같은 벡터: {torch.equal(out[1, 0], out[1, 1])}")
    print(f"출력 = weight[ID]: {torch.equal(out, emb.weight[batch])}")

    print("\n── 3. 코사인 유사도 ──")
    for a, b in ((1, 3), (1, 2), (1, 1)):
        print(f"ID {a}·ID {b}: 직접 {cosine(table[a], table[b]):.3f}, "
              f"PyTorch {F.cosine_similarity(table[a], table[b], dim=0).item():.3f}")


if __name__ == "__main__":
    main()
