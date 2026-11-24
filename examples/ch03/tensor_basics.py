# 제03장 실습 1: 텐서의 차원·모양·꺼내기·모양 바꾸기
#
# 실행: python examples/ch03/tensor_basics.py

import torch


def show(name: str, t: torch.Tensor) -> None:
    """텐서 이름, 차원 수, 모양을 한 줄로 출력한다."""
    print(f"{name:<28} ndim={t.ndim}  shape={tuple(t.shape)}")


def main() -> None:
    print("── 1. 스칼라에서 텐서까지 ──")
    show("스칼라(값 하나)", torch.tensor(3.0))
    show("벡터(토큰 하나의 표현)", torch.zeros(8))
    show("행렬(문장 하나)", torch.zeros(4, 8))
    show("3차원(문장 묶음)", torch.zeros(2, 4, 8))

    # LLM의 기본 모양: [batch, seq, hidden] = [문장 2개, 토큰 4개, 표현 8차원]
    x = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)

    print("\n── 2. Axis(축)별 연산 ──")
    show("x", x)
    show("x.sum(dim=0) 문장끼리 합", x.sum(dim=0))
    show("x.sum(dim=1) 토큰끼리 합", x.sum(dim=1))
    show("x.sum(dim=-1) 표현끼리 합", x.sum(dim=-1))

    print("\n── 3. Indexing과 Slicing ──")
    show("x[0] 첫 번째 문장", x[0])
    show("x[:, -1, :] 마지막 토큰", x[:, -1, :])
    show("x[:, :2, :] 앞 두 토큰", x[:, :2, :])

    # Slicing 결과는 복사본이 아니라 같은 메모리를 보는 창(view)이다.
    first = x[0]
    first[0, 0] = -1.0
    print(f"first[0, 0]을 바꾼 뒤 x[0, 0, 0] = {x[0, 0, 0].item()}  (원본도 바뀜)")
    x[0, 0, 0] = 0.0

    print("\n── 4. Reshape·Squeeze·Unsqueeze ──")
    show("x.reshape(8, 8)", x.reshape(8, 8))
    show("x.reshape(2, -1)", x.reshape(2, -1))
    mask = torch.ones(2, 4)
    show("mask [batch, seq]", mask)
    show("mask.unsqueeze(1).unsqueeze(2)", mask.unsqueeze(1).unsqueeze(2))
    show("torch.zeros(1, 4, 8).squeeze(0)", torch.zeros(1, 4, 8).squeeze(0))

    print("\n── 5. Transpose와 Permute ──")
    t = x.transpose(1, 2)
    show("x.transpose(1, 2)", t)
    show("x.permute(2, 0, 1)", x.permute(2, 0, 1))
    print(f"x 연속 저장 여부: {x.is_contiguous()},  transpose 결과: {t.is_contiguous()}")
    try:
        t.view(2, -1)
    except RuntimeError:
        print("t.view(2, -1) → 오류: 연속 저장이 아닌 텐서는 view로 모양을 바꿀 수 없음")
    show("t.reshape(2, -1) → 가능", t.reshape(2, -1))
    show("t.contiguous().view(2, -1)", t.contiguous().view(2, -1))


if __name__ == "__main__":
    main()
